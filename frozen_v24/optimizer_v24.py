import numpy as np
from collections import defaultdict
from maps_llm.entities import Plan
from maps_llm.predict import quality, feasible
from maps_llm.pareto import nondominated, crowding

def _stable_plan_key(p):
    return (int(p.primary), int(p.verifier))

def _q(x):
    return float(np.round(float(x), 12))

class MAPSv2:
    """
    MAPS-LLM v2.4

    v2.4 keeps v2.3's debt-conditioned structural fairness, but adds a
    counterfactual safety invariant.

    Service risk:
        R(p) = 0.70 * predicted_quality_shortfall(p)
             + 0.30 * predicted_SLA_risk(p)

    Debt-conditioned admissibility:
        R(p) <= R_min + (R_max - R_min) * exp(-gamma * D_u)

    Safety invariant:
        if fairness changes the chosen plan relative to the no-fairness
        counterfactual p0, the chosen plan must satisfy
            R(p_fair) <= R(p0)

    Fairness therefore cannot knowingly move an indebted user to a plan with
    higher predicted service risk than the scheduler would otherwise choose.
    """

    def __init__(
        self,
        seed=1,
        beta=.10,
        ledger=None,
        fairness_gamma=.70,
        fairness_enabled=True,
        fairness_safe=True,
    ):
        self.rng = np.random.default_rng(seed)
        self.beta = beta
        self.n = defaultdict(int)
        self.mu = defaultdict(lambda: .72)
        self.user_bad = defaultdict(float)

        # Conditional aggregation learning
        self.esc_a = defaultdict(lambda: 1.0)
        self.esc_b = defaultdict(lambda: 1.0)
        self.plan_n = defaultdict(int)
        self.plan_mu = defaultdict(lambda: .78)

        self.ledger = ledger
        self.fairness_gamma = float(fairness_gamma)
        self.fairness_enabled = bool(fairness_enabled)
        self.fairness_safe = bool(fairness_safe)

    def _learn(self, records):
        if not records:
            return

        r = records[-1]

        # Primary model learns only its own realized quality.
        k = (r["kind"], r["model"])
        n = self.n[k] + 1
        obs = float(r.get("primary_quality", r["quality"]))
        self.mu[k] += (obs - self.mu[k]) / n
        self.n[k] = n

        # Planned-verifier posterior.
        if r.get("verifier", ""):
            pk = (r["kind"], r["model"], r["verifier"])
            if r["aggregated"]:
                self.esc_a[pk] += 1.0
            else:
                self.esc_b[pk] += 1.0

            pn = self.plan_n[pk] + 1
            self.plan_mu[pk] += (float(r["quality"]) - self.plan_mu[pk]) / pn
            self.plan_n[pk] = pn

        bad = (
            .5 * float(r["sla_violation"])
            + .3 * float(not r["quality_floor_met"])
            + .2 * float(not r["budget_met"])
        )
        self.user_bad[r["user_id"]] = .92 * self.user_bad[r["user_id"]] + bad

    def _expected_plan(self, t, p, ms):
        from maps_llm.predict import plan_prediction

        if p.verifier < 0:
            q, c, l = plan_prediction(t, p, ms)
            return float(q), float(c), float(l), 0.0

        q1, c1, l1 = plan_prediction(t, Plan(p.primary, -1), ms)
        q2, c2, l2 = plan_prediction(t, p, ms)
        pk = (t.kind, ms[p.primary].name, ms[p.verifier].name)

        pe = self.esc_a[pk] / (self.esc_a[pk] + self.esc_b[pk])

        expected_cost = c1 + pe * (c2 - c1)
        expected_latency = l1 + pe * (l2 - l1)

        predicted_quality = q1 + pe * (q2 - q1)
        pn = self.plan_n[pk]
        expected_quality = (
            predicted_quality
            if pn == 0
            else .45 * predicted_quality + .55 * self.plan_mu[pk]
        )

        return (
            float(expected_quality),
            float(expected_cost),
            float(expected_latency),
            float(pe),
        )

    def _candidate_plans(self, t, ms):
        feasible_models = sorted(i for i, m in enumerate(ms) if feasible(t, m))
        plans = [Plan(i) for i in feasible_models]

        # Two highest-predicted-quality primaries may pair with the strongest
        # distinct verifier.
        top = sorted(
            feasible_models,
            key=lambda i: (-_q(quality(t, ms[i])), i),
        )[:2]

        for i in top:
            others = [j for j in feasible_models if j != i]
            if others:
                j = sorted(
                    others,
                    key=lambda z: (-_q(quality(t, ms[z])), z),
                )[0]
                plans.append(Plan(i, j))

        return feasible_models, sorted(set(plans), key=_stable_plan_key)

    def _build_candidates(self, t, ms, plans):
        rows = []

        for p in plans:
            q, c, l, pe = self._expected_plan(t, p, ms)

            k = (t.kind, ms[p.primary].name)
            n = self.n[k]
            learned = q if n == 0 else .45 * q + .55 * self.mu[k]
            unc = 1 / np.sqrt(n + 1)

            q_short = max(0.0, t.quality_floor - learned)
            sla_risk = max(0.0, l / max(t.deadline, .1) - 1.0)
            service_risk = .70 * q_short + .30 * sla_risk

            rows.append(
                dict(
                    plan=p,
                    q=float(q),
                    c=float(c),
                    l=float(l),
                    pe=float(pe),
                    learned=float(learned),
                    unc=float(unc),
                    n=int(n),
                    q_short=float(q_short),
                    sla_risk=float(sla_risk),
                    service_risk=float(service_risk),
                )
            )

        return rows

    def _admissible_indices(self, rows, t, enabled=None):
        """
        v2.3 debt-conditioned admissibility only.
        v2.4's safety cap is added later relative to the no-fair counterfactual.
        """
        if not rows:
            return []

        use_fairness = self.fairness_enabled if enabled is None else bool(enabled)
        if not use_fairness:
            return list(range(len(rows)))

        debt = max(0.0, float(self.user_bad[t.user_id]))
        risks = np.array([r["service_risk"] for r in rows], dtype=float)

        rmin = float(risks.min())
        rmax = float(risks.max())

        if debt <= 1e-15 or abs(rmax - rmin) <= 1e-15:
            return list(range(len(rows)))

        width = (rmax - rmin) * float(np.exp(-self.fairness_gamma * debt))
        threshold = rmin + width

        keep = [i for i, r in enumerate(risks) if r <= threshold + 1e-12]

        if not keep:
            keep = [int(np.argmin(risks))]

        return keep

    def _select_from_indices(self, t, rows, indices):
        """
        Run the MAPS Pareto/crowding/exploration/exploitation selection on the
        supplied candidate subset.
        """
        vectors = []
        meta = []

        for original_idx in indices:
            r = rows[original_idx]
            v = np.array(
                [
                    r["c"] / max(t.budget, .001),
                    r["l"] / max(t.deadline, .1),
                    1 - r["learned"],
                    r["sla_risk"],
                    r["service_risk"],
                ],
                dtype=np.float64,
            )
            vectors.append(v)
            meta.append((original_idx, r))

        nd_local = sorted(nondominated(vectors))
        V = np.array([vectors[i] for i in nd_local], dtype=np.float64)
        cd = crowding(V)

        ranked = []
        details = []

        for z, li in enumerate(nd_local):
            original_idx, r = meta[li]

            div = .035 if r["n"] < 3 else 0.0
            crowd = (
                0.0
                if np.isinf(cd[z])
                else min(float(cd[z]), 3.0) * .01
            )

            exploit = (
                r["learned"]
                - .06 * V[z, 0]
                - .035 * V[z, 1]
                - .15 * V[z, 3]
                - .12 * V[z, 4]
            )

            score = _q(exploit + self.beta * r["unc"] + div + crowd)
            p = r["plan"]

            ranked.append((-score, _stable_plan_key(p), original_idx, p))

            details.append(
                dict(
                    original_index=int(original_idx),
                    plan=[p.primary, p.verifier],
                    vector=[float(x) for x in V[z]],
                    service_risk=float(r["service_risk"]),
                    crowding=None if np.isinf(cd[z]) else float(cd[z]),
                    uncertainty=float(r["unc"]),
                    count=int(r["n"]),
                    learned=float(r["learned"]),
                    predicted_quality=float(r["q"]),
                    predicted_cost=float(r["c"]),
                    predicted_latency=float(r["l"]),
                    escalation_probability=float(r["pe"]),
                    exploit=float(exploit),
                    diversification_bonus=float(div),
                    score=float(score),
                )
            )

        ranked.sort()
        _, _, chosen_idx, chosen_plan = ranked[0]

        return chosen_plan, int(chosen_idx), nd_local, details

    def choose(self, t, ms, records):
        self._learn(records)

        feasible_models, plans = self._candidate_plans(t, ms)
        rows = self._build_candidates(t, ms, plans)

        # 1) Compute no-fairness counterfactual from the exact same state.
        all_indices = list(range(len(rows)))
        p0, p0_idx, _, _ = self._select_from_indices(t, rows, all_indices)
        p0_risk = float(rows[p0_idx]["service_risk"])

        # 2) Apply debt-conditioned structural fairness.
        admissible = self._admissible_indices(rows, t, enabled=self.fairness_enabled)

        # 3) v2.4 safety cap: fairness may not select a plan with risk above the
        #    no-fair counterfactual choice.
        debt = max(0.0, float(self.user_bad[t.user_id]))
        fairness_intervened = (
            self.fairness_enabled
            and debt > 1e-15
            and len(admissible) < len(all_indices)
        )

        if fairness_intervened and self.fairness_safe:
            safe = [
                i for i in admissible
                if rows[i]["service_risk"] <= p0_risk + 1e-12
            ]
            if safe:
                admissible_safe = safe
            else:
                # Counterfactual plan is always feasible by construction.
                admissible_safe = [p0_idx]
        else:
            # At zero debt (or when fairness prunes nothing), preserve ordinary
            # MAPS behavior exactly.
            admissible_safe = admissible

        selected, selected_idx, nd_local, ledger_rows = self._select_from_indices(
            t, rows, admissible_safe
        )

        selected_risk = float(rows[selected_idx]["service_risk"])

        if self.ledger is not None:
            self.ledger.append(
                dict(
                    task_id=int(t.task_id),
                    user_id=int(t.user_id),
                    kind=t.kind,
                    arrival=float(t.arrival),
                    quality_floor=float(t.quality_floor),
                    budget=float(t.budget),
                    deadline=float(t.deadline),
                    user_bad=float(self.user_bad[t.user_id]),
                    fairness_gamma=float(self.fairness_gamma),
                    feasible_models=[int(i) for i in feasible_models],
                    all_plans=[[p.primary, p.verifier] for p in plans],
                    all_service_risks=[
                        float(r["service_risk"]) for r in rows
                    ],
                    counterfactual_no_fair_plan=[p0.primary, p0.verifier],
                    counterfactual_no_fair_risk=float(p0_risk),
                    admissible_indices=[int(i) for i in admissible],
                    safe_admissible_indices=[int(i) for i in admissible_safe],
                    nondominated_local_indices=[int(i) for i in nd_local],
                    candidates=ledger_rows,
                    selected_plan=[selected.primary, selected.verifier],
                    selected_risk=float(selected_risk),
                    fairness_intervened=bool(fairness_intervened),
                    fairness_changed=(selected != p0),
                    fairness_safety_ok=bool(
                        (not fairness_intervened)
                        or selected_risk <= p0_risk + 1e-12
                    ),
                )
            )

        return selected
