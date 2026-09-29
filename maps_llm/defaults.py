from .entities import User, Model

TASKS=["general","reasoning","coding","long_context","translation"]

def users():
    return [
        User(0,"standard",1.0,.022,1.00,.72,"balanced"),
        User(1,"urgent",1.5,.050,.62,.78,"latency-critical"),
        User(2,"budget",.8,.010,1.35,.66,"cost-critical"),
        User(3,"quality",1.4,.065,1.05,.86,"quality-critical"),
        User(4,"batch",.7,.018,1.75,.70,"throughput-oriented"),
    ]

def models():
    return [
        Model("economy",dict(zip(TASKS,[.74,.58,.57,.60,.72])),.45,.12,.0002,.0008,8000,8,.985),
        Model("balanced",dict(zip(TASKS,[.84,.76,.75,.78,.84])),.85,.18,.0010,.0030,16000,5,.990),
        Model("reasoner",dict(zip(TASKS,[.88,.94,.88,.80,.82])),1.65,.35,.0030,.0100,32000,3,.995),
        Model("specialist",dict(zip(TASKS,[.86,.84,.95,.91,.93])),1.25,.28,.0022,.0070,64000,3,.993),
    ]
