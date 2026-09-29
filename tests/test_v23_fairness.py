import unittest, copy
from maps_llm.optimizer import MAPSv2
from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute
from maps_llm.entities import Plan

class TestV23Fairness(unittest.TestCase):
    def test_zero_debt_admits_all(self):
        t=load_workload(0,limit=1)[0]
        p=MAPSv2(1)
        f,plans=p._candidate_plans(t,models())
        rows=p._build_candidates(t,models(),plans)
        keep=p._admissible_indices(rows,t)
        self.assertEqual(keep,list(range(len(rows))))

    def test_positive_debt_can_prune(self):
        t=load_workload(0,limit=1)[0]
        p=MAPSv2(1)
        p.user_bad[t.user_id]=3.0
        f,plans=p._candidate_plans(t,models())
        rows=p._build_candidates(t,models(),plans)
        keep=p._admissible_indices(rows,t)
        self.assertLess(len(keep),len(rows))
        kept_r=[rows[i]["service_risk"] for i in keep]
        removed_r=[rows[i]["service_risk"] for i in range(len(rows)) if i not in keep]
        if removed_r:
            self.assertLessEqual(max(kept_r),max(removed_r)+1e-12)

    def test_execution_runs(self):
        rec=execute(load_workload(0,limit=30),models(),MAPSv2(51000),trace=load_trace(0))
        self.assertEqual(len(rec),30)

if __name__=="__main__":
    unittest.main()
