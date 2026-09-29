from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class User:
    user_id: int
    name: str
    priority: float
    budget: float
    deadline_scale: float
    quality_floor: float
    profile: str

@dataclass(frozen=True)
class Model:
    name: str
    quality: Dict[str,float]
    latency: float
    jitter: float
    in_cost: float
    out_cost: float
    context: int
    capacity: int
    reliability: float

@dataclass
class Task:
    task_id: int
    user_id: int
    kind: str
    arrival: float
    input_tokens: int
    output_tokens: int
    deadline: float
    quality_floor: float
    budget: float
    difficulty: float
    priority: float

@dataclass(frozen=True)
class Plan:
    primary: int
    verifier: int = -1   # -1 = single model
