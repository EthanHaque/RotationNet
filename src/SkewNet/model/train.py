import argparse
import json
import os
from dataclasses import dataclass

import torch
import wandb
from torch.distributed import destroy_process_group, init_process_group, ReduceOp, all_reduce
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from SkewNet.model.rotated_images_dataset import DataConfig, RotatedImageDataset
from SkewNet.model.rotation_net import ModelRegistry


@dataclass
class TrainConfig:
    evaluate: bool = True
    start_epoch: int = 0
    max_epochs: int = 20
    batch_size: int = 64
    data_loader_prefetch_factor: int = 4
    data_loader_num_workers: int = 8
    snapshot_dir: str = ""
    snapshot_path: str = ""
    resume: bool = False
    snapshot_interval: int = 1
    use_automatic_mixed_precision: bool = False
    grad_norm_clip: float = 1.0
    profile: bool = False
    logdir: str = "/scratch/gpfs/eh0560/SkewNet/logs"


@dataclass
class SnapshotConfig:
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    epoch: int
    best_loss: float
    best_epoch: int


@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    step_size: int
    gamma: float


class Trainer:
    def __init__(self, trainer_config, model, criterion, optimizer, scheduler, train_dataset, runID, val_dataset=None):
        if trainer_config.evaluate and not val_dataset:
            raise ValueError("Validation dataset must be provided when in evaluation mode")
        if not trainer_config.evaluate and val_dataset:
            raise ValueError("Validation dataset provided but not in evaluation mode.")
        if not trainer_config.snapshot_dir:
            raise ValueError("No snapshot directory provided.")

        self.config = trainer_config

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))

        self.model = model
        self.model = model.to(self.local_rank)

        self.runID = f"{self.model.__class__.__name__}_{runID}"

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataset = train_dataset
        self.train_loader = self._prepare_distributed_dataloader(train_dataset)
        self.val_loader = self._prepare_distributed_dataloader(val_dataset) if val_dataset else None

        self.epochs_run = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.snapshot_interval = max(self.config.snapshot_interval, 1)

        if self.config.use_automatic_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        if not self.config.snapshot_path:
            self.config.snapshot_path = os.path.join(
                self.config.snapshot_dir, f"{self.runID}_best.pth"
            )

        if not os.path.exists(os.path.dirname(self.config.snapshot_path)):
            os.makedirs(os.path.dirname(self.config.snapshot_path))
        if self.config.resume:
            self._load_snapshot(self.config.snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

        if self.config.profile:
            group = self.runID
            self.run = wandb.init(
                project="SkewNet", entity="ethanhaque", config=self.config, dir=self.config.logdir, group=group
            )
            # self.run.enable_profiling()
            self.run.watch(self.model)

        print(f"GPU {self.global_rank} | Initialized trainer")
        if self.global_rank == 0:
            print(f"ID {self.runID}")
            print(f"GPUs {torch.cuda.device_count()}")
            print(f"Train samples {len(self.train_dataset)}")
            if self.val_loader:
                print(f"Val samples {len(self.val_loader.dataset)}")
            print(f"Batch size {self.config.batch_size}")
            print(f"Epochs {self.config.max_epochs}")

    def _prepare_distributed_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_num_workers,
            prefetch_factor=self.config.data_loader_prefetch_factor,
            sampler=DistributedSampler(dataset, shuffle=True),
        )

    def _load_snapshot(self, snapshot_path):
        try:
            snapshot = torch.load(snapshot_path)
        except FileNotFoundError:
            print(f"Snapshot not found at {snapshot_path}")
            return

        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.scheduler.load_state_dict(snapshot.scheduler_state)
        self.epochs_run = snapshot.epoch
        self.best_loss = snapshot.best_loss
        self.best_epoch = snapshot.best_epoch
        print(f"GPU {self.global_rank} | Loaded snapshot from {snapshot_path}")

    def _save_snapshot(self, snapshot_path):
        model = self.model.module if isinstance(self.model, DDP) else self.model
        snapshot = SnapshotConfig(
            model_state=model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            epoch=self.epochs_run,
            best_loss=self.best_loss,
            best_epoch=self.best_epoch,
        )
        torch.save(snapshot, snapshot_path)

    def _calculate_metrics(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        mae = mae_loss(y_pred, y_true)
        return loss, mae

    def _training_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True), torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.config.use_automatic_mixed_precision
        ):
            x, y = batch
            x, y = x.to(self.local_rank), y.to(self.local_rank)
            y_hat = self.model(x)
            loss, mae = self._calculate_metrics(y_hat, y)
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_automatic_mixed_precision:
                self.scaler.scale(loss).backward()
                clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

            self.scheduler.step()
            print(
                f"GPU {self.global_rank} | EPOCH {self.epochs_run} | Batch {batch_idx} | Train Loss: {loss:.5f} | Train MAE: {mae:.5f}"
            )

        return loss, mae

    def _train(self, epoch, data_loader):
        data_loader.sampler.set_epoch(epoch)
        self.model.train()
        running_metrics = torch.zeros(3, device=self.local_rank)
        for idx, batch in enumerate(data_loader):
            batch_loss, mae = self._training_step(batch, idx)
            running_metrics[0] += batch_loss.item() * len(batch[0])
            running_metrics[1] += mae.item() * len(batch[0])
            running_metrics[2] += len(batch[0])

        all_reduce(running_metrics, op=ReduceOp.SUM)
        average_loss = (running_metrics[0] / running_metrics[2]).item()
        average_mae = (running_metrics[0] / running_metrics[2]).item()
        if self.global_rank == 0:
            print(f"EPOCH {self.epochs_run} | Train Loss: {average_loss:.5f} | Train MAE: {average_mae:.5f}")

        return average_loss, average_mae

    def _eval_step(self, batch, batch_idx):
        with torch.set_grad_enabled(False):
            x, y = batch
            x, y = x.to(self.local_rank), y.to(self.local_rank)
            y_hat = self.model(x)
            loss, mae = self._calculate_metrics(y_hat, y)
            print(
                f"GPU {self.global_rank} | EPOCH {self.epochs_run} | Batch {batch_idx} | Val Loss: {loss:.5f} | Val MAE: {mae:.5f}"
            )
            return loss, mae

    def _validate(self, epoch, data_loader):
        data_loader.sampler.set_epoch(epoch)
        self.model.eval()
        running_metrics = torch.zeros(3, device=self.local_rank)
        for idx, batch in enumerate(data_loader):
            batch_loss, mae = self._eval_step(batch, idx)
            running_metrics[0] += batch_loss.item() * len(batch[0])
            running_metrics[1] += mae.item() * len(batch[0])
            running_metrics[2] += len(batch[0])

        all_reduce(running_metrics, op=ReduceOp.SUM)
        average_loss = (running_metrics[0] / running_metrics[2]).item()
        average_mae = (running_metrics[0] / running_metrics[2]).item()
        if self.global_rank == 0:
            print(f"EPOCH {self.epochs_run} | Val Loss: {average_loss:.5f} | Val MAE: {average_mae:.5f}")

        return average_loss, average_mae

    def fit(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            train_loss, train_mae = self._train(epoch, self.train_loader)
            val_loss = None
            if self.val_loader:
                val_loss, val_mae = self._validate(epoch, self.val_loader)
                # TODO: make sure everything is communicated properly. All reduce?
                # TODO: add save snapshot for rank 0 only and for periodic snapshots
                # TODO: add early stopping
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self._save_snapshot(self.config.snapshot_path)
            self.epochs_run += 1

            if self.config.profile:
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "epoch": epoch,
                }
                self.run.log(metrics)

        if self.config.profile:
            self.run.finish()


def mse_loss(y_pred, y_true, scale=1):
    return torch.mean((y_pred - y_true) ** 2) * scale


def mae_loss(y_pred, y_true, scale=1):
    return torch.mean(torch.abs(y_pred - y_true)) * scale


def setup_optimizer(model, learning_rate, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def setup_scheduler(optimizer, step_size, gamma):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler


def setup_criterion():
    criterion = mse_loss
    return criterion


def setup_data_loaders(data_config, transform=None, target_transform=None):
    train_dataset = RotatedImageDataset(
        data_config, subset="train", transform=transform, target_transform=target_transform
    )
    val_dataset = RotatedImageDataset(data_config, subset="val", transform=transform, target_transform=target_transform)
    test_dataset = RotatedImageDataset(
        data_config, subset="test", transform=transform, target_transform=target_transform
    )

    return train_dataset, val_dataset, test_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch SkewNet Training")
    parser.add_argument("runID", metavar="RUNID", help="wandb uuid")
    parser.add_argument("config", metavar="FILE", help="path to config file")
    return parser.parse_args()


def setup_config(config):
    train_config = TrainConfig(**config["train_config"])
    optimizer_config = OptimizerConfig(**config["optimizer_config"])
    scheduler_config = SchedulerConfig(**config["scheduler_config"])
    data_config = DataConfig(**config["data_config"])
    return train_config, optimizer_config, scheduler_config, data_config


def get_train_objects(model, optimizer_config, scheduler_config, data_config):
    model = ModelRegistry.get_model(model)
    criterion = setup_criterion()
    optimizer = setup_optimizer(model, optimizer_config.learning_rate, optimizer_config.weight_decay)
    scheduler = setup_scheduler(optimizer, scheduler_config.step_size, scheduler_config.gamma)
    train_dataset, val_dataset, _ = setup_data_loaders(data_config)
    return model, criterion, optimizer, scheduler, train_dataset, val_dataset


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    ddp_setup()

    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    runID = args.runID

    train_config, optimizer_config, scheduler_config, data_config = setup_config(config)

    model, criterion, optimizer, scheduler, train_dataset, val_dataset = get_train_objects(
        config["model"], optimizer_config, scheduler_config, data_config
    )

    trainer = Trainer(train_config, model, criterion, optimizer, scheduler, train_dataset, runID, val_dataset)
    trainer.fit()

    destroy_process_group()


if __name__ == "__main__":
    main()
