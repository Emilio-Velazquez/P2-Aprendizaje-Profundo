import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_b, y_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = len(x_b)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / total_samples if total_samples > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            out = model(x_b)
            loss = criterion(out, y_b)
            bs = len(x_b)
            total_loss += loss.item() * bs
            total_samples += bs
            all_preds.append(out.cpu().numpy())
            all_targets.append(y_b.cpu().numpy())
    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, 1))
    targets_arr = (
        np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 1))
    )
    return mean_loss, preds_arr, targets_arr
