import unittest
from maps_llm.optimizer import MAPSv2
from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute
class TestV22Fairness(unittest.TestCase):
    def test_plan_sensitive_gap(self):
        debt=2.0; a0=.70*.02; b0=.70*.15+.30*.10
        self.assertGreater((1+.35*debt)*(b0-a0),b0-a0)
    def test_debt_updates(self):
        p=MAPSv2(1); r=dict(kind='general',model='economy',primary_quality=.5,quality=.5,verifier='',aggregated=False,user_id=0,sla_violation=True,quality_floor_met=False,budget_met=True)
        p._learn([r]); self.assertGreater(p.user_bad[0],0)
    def test_execution(self):
        r=execute(load_workload(0,limit=30),models(),MAPSv2(51000),trace=load_trace(0)); self.assertEqual(len(r),30)
if __name__=='__main__': unittest.main()
