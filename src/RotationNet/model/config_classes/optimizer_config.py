from dataclasses import dataclass
@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
