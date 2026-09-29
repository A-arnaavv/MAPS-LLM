import unittest
from maps_llm.optimizer import MAPSv2
from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute

class TestV25Safety(unittest.TestCase):
    def test_stress_updates(self):
        p=MAPSv2(1)
        r=dict(kind="general",model="economy",primary_quality=.4,quality=.4,verifier="",
               aggregated=False,user_id=0,sla_violation=True,quality_floor_met=False,budget_met=False)
        p._learn([r])
        self.assertGreater(p.system_stress,0.0)

    def test_invariant(self):
        ledger=[]
        execute(load_workload(0,limit=100),models(),MAPSv2(51000,ledger=ledger),trace=load_trace(0))
        self.assertTrue(ledger)
        self.assertTrue(all(r["safety_invariant_ok"] for r in ledger))

    def test_execution(self):
        rec=execute(load_workload(1,limit=30),models(),MAPSv2(51001),trace=load_trace(1))
        self.assertEqual(len(rec),30)
if __name__=="__main__":unittest.main()
