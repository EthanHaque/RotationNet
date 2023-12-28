from dataclasses import dataclass


@dataclass
class TrainConfig:
    evaluate: bool = True
    test: bool = False
    start_epoch: int = 0
    max_epochs: int = 20
    batch_size: int = 64
    data_loader_prefetch_factor: int = 4
    data_loader_num_workers: int = 8
    snapshot_dir: str = ""
    snapshot_prefix: str = ""
    snapshot_path: str = ""
    resume: bool = False
    snapshot_interval: int = 1
    use_automatic_mixed_precision: bool = False
    grad_norm_clip: float = 1.0
    profile: bool = False
    logdir: str = "/scratch/gpfs/eh0560/SkewNet/logs"
