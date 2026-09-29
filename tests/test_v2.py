import unittest,os,sys,hashlib,json
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from maps_llm.defaults import users,models
from maps_llm.canonical import load_workload,load_trace,subset_sha256
from maps_llm.simulator import execute,summarize
from maps_llm.optimizer import MAPSv2
from maps_llm.nsga2_control import NSGA2Control
from maps_llm.pareto import dominates,nondominated

class T(unittest.TestCase):
    def test_canonical_workload(self):
        x=load_workload(0,40); self.assertEqual(len(x),40)
        self.assertTrue(all(x[i].arrival<=x[i+1].arrival for i in range(39)))
    def test_maps(self):
        x=load_workload(1,30); r=execute(x,models(),MAPSv2(9),trace=load_trace(1))
        self.assertEqual(len(r),30); self.assertTrue(0<=summarize(r,5)["fairness"]<=1)
    def test_nsga(self):
        x=load_workload(2,20); r=execute(x,models(),NSGA2Control(),trace=load_trace(2))
        self.assertEqual(len(r),20)
    def test_pareto(self):
        self.assertTrue(dominates([1,1],[2,2]))
        self.assertEqual(nondominated([[1,2],[2,1],[3,3]]),[0,1])
    def test_exact_repeat(self):
        x=load_workload(0,100); tr=load_trace(0)
        r1=execute(x,models(),MAPSv2(51000),trace=tr)
        x=load_workload(0,100)
        r2=execute(x,models(),MAPSv2(51000),trace=tr)
        self.assertEqual(r1,r2)
    def test_input_hash_repeat(self):
        self.assertEqual(subset_sha256(0,100,False),subset_sha256(0,100,False))
        self.assertEqual(subset_sha256(0,100,True),subset_sha256(0,100,True))
    def test_trace_required(self):
        with self.assertRaises(ValueError):
            execute(load_workload(0,2),models(),MAPSv2(1))


def test_ledger_repeat(self):
    from maps_llm.optimizer import MAPSv2
    x=load_workload(0,25); tr=load_trace(0)
    l1=[]; r1=execute(x,models(),MAPSv2(51000,ledger=l1),trace=tr)
    x=load_workload(0,25)
    l2=[]; r2=execute(x,models(),MAPSv2(51000,ledger=l2),trace=tr)
    self.assertEqual(l1,l2)
    self.assertEqual(r1,r2)


def test_crowding_tied_extremes_all_boundary(self):
    import numpy as np
    from maps_llm.pareto import crowding
    # Three candidates share the minimum on objective 2.
    V=np.array([
        [0.10,0.20,0.30,0.0,0.0],
        [0.20,0.30,0.20,0.0,0.0],
        [0.30,0.40,0.10,0.0,0.0],
        [0.40,0.50,0.10,0.0,0.0],
        [0.50,0.60,0.10,0.0,0.0],
    ],dtype=float)
    d=crowding(V)
    self.assertTrue(np.isinf(d[2]))
    self.assertTrue(np.isinf(d[3]))
    self.assertTrue(np.isinf(d[4]))

def test_crowding_permutation_invariant(self):
    import numpy as np
    from maps_llm.pareto import crowding
    V=np.array([
        [0.060736363636363644,0.18,0.343012914471041,0.0,0.10104926118095059],
        [0.2705454545454546,0.34,0.243012914471041,0.0,0.0010492611809506114],
        [0.8447727272727273,0.66,0.20301291447104097,0.0,0.0],
        [1.4532272727272728,0.708,0.09569300634247946,0.0,0.0],
        [0.6084545454545455,0.5,0.223012914471041,0.0,0.0],
    ])
    base=crowding(V)
    perm=np.array([4,2,0,3,1])
    got=crowding(V[perm])
    restored=np.empty_like(got)
    restored[perm]=got
    for a,b in zip(base,restored):
        self.assertEqual(np.isinf(a),np.isinf(b))
        if not np.isinf(a):
            self.assertAlmostEqual(a,b,places=12)


def test_primary_learning_not_aggregate_contaminated(self):
    from maps_llm.optimizer import MAPSv2
    p=MAPSv2(1)
    rec=[dict(kind="general",model="m",primary_quality=.40,quality=.95,
              verifier="v",aggregated=True,user_id=0,sla_violation=False,
              quality_floor_met=True,budget_met=True)]
    p._learn(rec)
    self.assertAlmostEqual(p.mu[("general","m")],.40,places=12)

def test_escalation_posterior_updates(self):
    from maps_llm.optimizer import MAPSv2
    p=MAPSv2(1)
    r=dict(kind="general",model="m",primary_quality=.5,quality=.8,verifier="v",
           aggregated=True,user_id=0,sla_violation=False,quality_floor_met=True,budget_met=True)
    p._learn([r])
    self.assertEqual(p.esc_a[("general","m","v")],2.0)
    self.assertEqual(p.esc_b[("general","m","v")],1.0)

if __name__=="__main__": unittest.main()
