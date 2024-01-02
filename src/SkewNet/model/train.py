import torch
import math
from typing import Iterable

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler,
                    data_loader: Iterable,
                    device: torch.device,
                    epoch: int,
                    use_amp=False,
                    loss_scaler=None,
                    max_norm=None,
                    wandb_logger=None,
                    start_steps=None,
                    num_training_steps_per_epoch=None,):
    """Train the model for one epoch."""
    
    model.train()
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        if num_training_steps_per_epoch and data_iter_step > num_training_steps_per_epoch:
            continue

        global_step = start_steps + data_iter_step

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(f"Outputs: {outputs}")
            print(f"Targets: {targets}")
            assert math.isfinite(loss_value), "Loss is not finite. Stopping training."

        if use_amp:
            grad_norm = loss_scaler(loss, optimizer, max_norm, model.parameters(), update_grad=True)
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        torch.cuda.synchronize()

        print(f"Epoch: {epoch}, Step: {data_iter_step}, Loss: {loss_value}")

    scheduler.step()

    return 

