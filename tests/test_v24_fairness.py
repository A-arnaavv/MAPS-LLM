import unittest
from maps_llm.optimizer import MAPSv2
from maps_llm.defaults import models
from maps_llm.canonical import load_workload, load_trace
from maps_llm.simulator import execute

class TestV24FairnessSafety(unittest.TestCase):

    def test_zero_debt_keeps_no_fair_behavior(self):
        t = load_workload(0, limit=1)[0]

        fair = MAPSv2(1, fairness_enabled=True, fairness_safe=True)
        nofair = MAPSv2(1, fairness_enabled=False, fairness_safe=True)

        a = fair.choose(t, models(), [])
        b = nofair.choose(t, models(), [])

        self.assertEqual(a, b)

    def test_ledger_safety_invariant(self):
        ledger = []

        execute(
            load_workload(0, limit=100),
            models(),
            MAPSv2(51000, ledger=ledger, fairness_enabled=True, fairness_safe=True),
            trace=load_trace(0),
        )

        self.assertTrue(ledger)
        self.assertTrue(all(r["fairness_safety_ok"] for r in ledger))

        changed = [r for r in ledger if r["fairness_changed"]]
        self.assertGreater(len(changed), 0)

        for r in changed:
            self.assertLessEqual(
                r["selected_risk"],
                r["counterfactual_no_fair_risk"] + 1e-12,
            )

    def test_fairness_can_change_a_decision(self):
        ledger = []

        execute(
            load_workload(0, limit=100),
            models(),
            MAPSv2(51000, ledger=ledger, fairness_enabled=True, fairness_safe=True),
            trace=load_trace(0),
        )

        self.assertTrue(any(r["fairness_changed"] for r in ledger))

    def test_execution_runs(self):
        rec = execute(
            load_workload(1, limit=30),
            models(),
            MAPSv2(51001),
            trace=load_trace(1),
        )
        self.assertEqual(len(rec), 30)

if __name__ == "__main__":
    unittest.main()
