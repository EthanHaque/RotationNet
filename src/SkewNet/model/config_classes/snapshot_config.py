from dataclasses import dataclass

@dataclass
class SnapshotConfig:
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    epoch: int
    best_loss: float
    best_epoch: int
