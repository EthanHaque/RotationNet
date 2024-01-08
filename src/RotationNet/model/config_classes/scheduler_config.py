from dataclasses import dataclass
@dataclass
class SchedulerConfig:
    T_max: int
    eta_min: float
