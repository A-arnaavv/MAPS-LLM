import unittest,copy
from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute
from maps_llm.optimizer import MAPSv2

class TestV211Diagnostics(unittest.TestCase):
    def test_planned_verifier_is_recorded_even_when_not_executed(self):
        rec=execute(load_workload(0,limit=100),models(),MAPSv2(51000),trace=load_trace(0))
        planned=[r for r in rec if r.get("verifier","")]
        self.assertTrue(planned)
        self.assertTrue(any(not r["aggregated"] for r in planned))

    def test_escalation_posterior_gets_success_and_failure_updates(self):
        p=MAPSv2(51000)
        rec=execute(load_workload(0,limit=100),models(),p,trace=load_trace(0))
        succ=sum(v-1.0 for v in p.esc_a.values())
        fail=sum(v-1.0 for v in p.esc_b.values())
        self.assertGreater(succ,0)
        self.assertGreater(fail,0)

    def test_primary_quality_field_exists(self):
        rec=execute(load_workload(0,limit=10),models(),MAPSv2(51000),trace=load_trace(0))
        self.assertTrue(all("primary_quality" in r for r in rec))

if __name__=="__main__":
    unittest.main()
