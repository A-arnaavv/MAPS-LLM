# MAPS-LLM v2 research specification

Primary vector (minimize):
F(S) = [cost, tail latency, quality loss, SLA violation risk, user outcome unfairness].

Exploration/exploitation and diversification/intensification are deliberately
not synonyms.

- Exploration: uncertainty bonus for poorly observed task-model mappings.
- Exploitation: learned quality and feasibility of established mappings.
- Diversification: crowding preference plus under-sampled plans.
- Intensification: non-dominated filtering concentrates selection on locally
  Pareto-efficient plans.

Conditional aggregation is a plan-level action: a verifier is acquired only
when the primary result remains near/below the task's quality requirement.

v2 limitation: decisions are still online per task. A genuine horizon-level
population scheduler is not yet implemented. Therefore do not describe v2 as
joint batch/horizon scheduling in a paper.
