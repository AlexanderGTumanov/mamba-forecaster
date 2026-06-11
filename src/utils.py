import os
import sys
import math
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

DATA_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
DATA_PATH = "../data/ETTm2.csv"
_LAST_PROGRESS_MESSAGE_LEN = 0

def load_data(normalize = True, overwrite = False):
    if overwrite or not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok = True)
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            f.write(response.content)
    df = pd.read_csv(DATA_PATH, parse_dates = ["date"])
    if normalize:
        cols = df.select_dtypes(include = "number").columns
        mean = df[cols].mean()
        std = df[cols].std()
        df[cols] = (df[cols] - mean) / std
    return df

class TSDataset(Dataset):
    def __init__(self, df, L = 512, H = 96, step = 1):
        self.df = df.select_dtypes(include = "number").values
        if len(self.df) < L + H:
            raise ValueError("L + H is greater than the total length of the series.")
        self.L = L
        self.H = H
        self.step = step
        
    def __len__(self):
        return 1 + (len(self.df) - self.L - self.H) // self.step
    
    def __getitem__(self, idx):
        start = idx * self.step
        x = self.df[start : start + self.L]
        y = self.df[start + self.L : start + self.L + self.H]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)

class DLinear(nn.Module):
    def __init__(self, L, H, n_channels):
        super().__init__()
        self.channels = nn.ModuleList([nn.Linear(L, H) for _ in range(n_channels)])

    def forward(self, x):
        return torch.stack([self.channels[i](x[:, :, i]) for i in range(len(self.channels))], dim = -1)

class RevIN(nn.Module):
    def __init__(self, n_channels, eps = 1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, n_channels))
        self.beta  = nn.Parameter(torch.zeros(1, 1, n_channels))

    def normalize(self, x):
        self._mean = x.mean(dim = 1, keepdim = True)
        self._std  = x.std(dim = 1, keepdim = True) + self.eps
        return (x - self._mean) / self._std * self.gamma + self.beta

    def denormalize(self, y):
        return (y - self.beta) / self.gamma * self._std + self._mean

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state = 32, kernel_size = 32, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.kernel_size = kernel_size
        self.W_in = nn.Linear(self.d_model, 2 * self.d_inner, bias = False)
        self.W_o = nn.Linear(self.d_inner, self.d_model, bias = False)
        A = torch.arange(1, self.d_state + 1, dtype = torch.float32)
        A = A.unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.W_BC = nn.Linear(self.d_inner, 2 * self.d_state, bias = False)
        self.W_delta = nn.Linear(self.d_inner, self.d_inner)
        dt = torch.exp(torch.rand(self.d_inner) * math.log(0.1 / 0.001) + math.log(0.001))
        self.W_delta.bias = nn.Parameter(torch.log(torch.exp(dt) - 1))
        self.norm = nn.RMSNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            kernel_size = self.kernel_size,
            padding = self.kernel_size - 1,
            groups = self.d_inner
        )

    def ssm(self, z):
        B, L, _ = z.shape
        h = torch.zeros(B, self.d_inner, self.d_state, device = z.device, dtype = z.dtype)
        A = - torch.exp(self.A_log)
        ys = torch.empty(B, L, self.d_inner, device = z.device, dtype = z.dtype)
        I = torch.ones(1, self.d_inner, self.d_state, device = z.device, dtype = z.dtype)
        for t in range(L):
            z_t = z[:, t, :]
            delta = F.softplus(self.W_delta(z_t)).unsqueeze(-1)
            A_delta = delta * A
            A_bar = torch.exp(A_delta)
            B_t, C_t = self.W_BC(z_t).chunk(2, dim = -1)
            B_bar = ((A_bar - I) / (A_delta + 1e-6)) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * z_t.unsqueeze(-1)
            y = (C_t.unsqueeze(1) * h).sum(-1) + self.D * z_t
            ys[:, t, :] = y
        return ys

    def forward(self, x):
        B, L, _ = x.shape
        x_norm = self.norm(x)
        z, r = self.W_in(x_norm).chunk(2, dim = -1)
        z = z.transpose(1, 2)
        z = self.conv(z)[:, :, :L]
        z = F.silu(z.transpose(1, 2))
        z = self.dropout(self.ssm(z))
        r = F.silu(r)
        return x + self.W_o(z * r)

class MambaModel(nn.Module):
    def __init__(
        self, L, H,
        n_channels,
        n_layers = 4,
        d_model = 64,
        d_inner = None,
        d_state = 32,
        kernel_size = 24,
        n_tail = 32,
        dropout = 0.1
    ):
        super().__init__()
        self.L = L
        self.H = H
        self.d_model = d_model
        self.d_inner = 2 * d_model if d_inner is None else d_inner
        self.d_state = d_state
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.n_tail = n_tail
        self.embedding = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model = d_model, d_inner = self.d_inner, d_state = d_state, kernel_size = kernel_size, dropout = dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.forecast = nn.ModuleList([nn.Linear(n_tail * d_model, H) for _ in range(n_channels)])

    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, -self.n_tail:, :].reshape(B * C, self.n_tail * self.d_model)
        x = x.reshape(B, C, self.n_tail * self.d_model)
        return torch.stack([self.forecast[i](x[:, i, :]) for i in range(C)], dim = -1)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    model = torch.load(model_path, map_location = "cpu", weights_only = False)
    return model
    
def prepare_dataloaders(df, L, H, batch_size = 32, valid_len = 0, test_len = 0, step = 1):
    T = len(df)
    train_loader = DataLoader(TSDataset(df.iloc[:T - valid_len - test_len], L = L, H = H, step = step), batch_size = batch_size, shuffle = True,  drop_last = True)
    valid_loader = DataLoader(TSDataset(df.iloc[T - valid_len - test_len : T - test_len], L = L, H = H), batch_size = batch_size, shuffle = False, drop_last = True) if valid_len > 0 else None
    test_loader  = DataLoader(TSDataset(df.iloc[T - test_len:], L = L, H = H), batch_size = batch_size, shuffle = False, drop_last = True) if test_len  > 0 else None
    return train_loader, valid_loader, test_loader

def evaluate(model, loader, revin = None):
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    if revin is not None:
        revin.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if revin is not None:
                x = revin.normalize(x)
            pred = model(x)
            if revin is not None:
                pred = revin.denormalize(pred)
            total_loss += F.mse_loss(pred, y).item()
            total_batches += 1
    model.train(training)
    if revin is not None:
        revin.train(training)
    return total_loss / total_batches

def show_progress(bi, total_batches, epoch = None, grad_norm = None):
    global _LAST_PROGRESS_MESSAGE_LEN
    pct = 100.0 * bi / max(1, total_batches)
    if epoch is None:
        base = f"progress: {pct:6.2f}%"
    else:
        base = f"epoch {epoch}: {pct:6.2f}%"
    if grad_norm is not None:
        GRAD_COL = 30
        spaces = " " * max(1, GRAD_COL - len(base))
        msg = f"{base}{spaces}gradient norm: {grad_norm:.4g}"
    else:
        msg = base
    padding = " " * max(0, _LAST_PROGRESS_MESSAGE_LEN - len(msg))
    sys.stdout.write("\r" + msg + padding)
    sys.stdout.flush()
    _LAST_PROGRESS_MESSAGE_LEN = len(msg)

def train_model(
    model,
    train_loader,
    valid_loader,
    epochs,
    lr = 1e-4,
    revin = None,
    patience = None,
    lr_patience = 2,
    weight_decay = 0.01,
    max_grad_norm = 1,
    clip_start_batch = None,
    checkpoint = "last",
    model_dir = "../model"
):
    global _LAST_PROGRESS_MESSAGE_LEN
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    os.makedirs(model_dir, exist_ok = True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = model.to(device)
    params = list(model.parameters())
    if revin is not None:
        revin = revin.to(device)
        params += list(revin.parameters())
    optimizer = torch.optim.AdamW(params, lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = lr_patience, factor = 0.5)
    history = {"train": [], "valid": []}
    total_train_batches = len(train_loader)
    best_valid_loss = float("inf")
    best_epoch = None
    best_model_state = None
    best_revin_state = None
    patience_counter = 0
    early_stopped = False
    for epoch in range(1, epochs + 1):
        _LAST_PROGRESS_MESSAGE_LEN = 0
        model.train()
        if revin is not None:
            revin.train()
        train_losses = []
        grad_norm_value = None
        show_progress(0, total_train_batches, epoch = epoch)
        for bi, (x, y) in enumerate(train_loader, start = 1):
            x = x.to(device)
            y = y.to(device)
            if revin is not None:
                x = revin.normalize(x)
            pred = model(x)
            if revin is not None:
                pred = revin.denormalize(pred)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            clip = False
            if max_grad_norm is not None:
                if epoch == 1:
                    if clip_start_batch is None:
                        clip = True
                    elif bi >= clip_start_batch:
                        clip = True
                else:
                    clip = True
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
            grad_norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
            if clip and grad_norm_value > max_grad_norm:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                sys.stdout.write("\n")
                sys.stdout.flush()
                print(f"gradient clipped: epoch {epoch}, batch {bi}: grad_norm = {grad_norm_value:.4g}")
            optimizer.step()
            train_losses.append(float(loss.item()))
            show_progress(bi, total_train_batches, epoch = epoch, grad_norm = min(grad_norm_value, max_grad_norm) if clip else grad_norm_value)
        train_loss = float(np.mean(train_losses))
        valid_loss = evaluate(model, valid_loader, revin = revin)
        history["train"].append(train_loss)
        history["valid"].append(valid_loss)
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr and best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            if revin is not None and best_revin_state is not None:
                revin.load_state_dict({k: v.to(device) for k, v in best_revin_state.items()})
        summary = f"epoch {epoch}: train_loss = {train_loss:.6f}, valid_loss = {valid_loss:.6f}, grad_norm = {grad_norm_value:.4g}, lr = {current_lr:.2e}"
        padding = " " * max(0, _LAST_PROGRESS_MESSAGE_LEN - len(summary))
        sys.stdout.write(f"\r{summary}{padding}\n")
        sys.stdout.flush()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_revin_state = {k: v.cpu().clone() for k, v in revin.state_dict().items()} if revin is not None else None
            patience_counter = 0
        else:
            if patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stopped = True
                    break
        if checkpoint == "each":
            torch.save(model, os.path.join(model_dir, f"model-{run_id}-e{epoch}.pt"))
            if revin is not None:
                torch.save(revin, os.path.join(model_dir, f"revin-{run_id}-e{epoch}.pt"))
    if patience is not None:
        model.load_state_dict(best_model_state)
        if revin is not None:
            revin.load_state_dict(best_revin_state)
        if early_stopped:
            print(f"Early stopping triggered at epoch {epoch}. Returning model from epoch {best_epoch}.")
    if checkpoint == "last":
        tag = f"e{best_epoch if patience is not None else epoch}"
        torch.save(model, os.path.join(model_dir, f"model-{run_id}-{tag}.pt"))
        if revin is not None:
            torch.save(revin, os.path.join(model_dir, f"revin-{run_id}-{tag}.pt"))
    elif checkpoint == "best":
        return_state = {k: v.clone() for k, v in model.state_dict().items()}
        return_revin_state = {k: v.clone() for k, v in revin.state_dict().items()} if revin is not None else None
        model.load_state_dict(best_model_state)
        if revin is not None:
            revin.load_state_dict(best_revin_state)
        torch.save(model, os.path.join(model_dir, f"model-{run_id}-e{best_epoch}.pt"))
        if revin is not None:
            torch.save(revin, os.path.join(model_dir, f"revin-{run_id}-e{best_epoch}.pt"))
        model.load_state_dict(return_state)
        if revin is not None:
            revin.load_state_dict(return_revin_state)
    torch.save(history, os.path.join(model_dir, f"history-{run_id}.pt"))
    return (model, revin) if revin is not None else model

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    model = torch.load(model_path, map_location = "cpu", weights_only = False)
    return model

def load_history(*history_paths):
    history = {"train": [], "valid": []}
    for path in history_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"history file not found: {path}")
        h = torch.load(path, map_location = "cpu")
        history["train"].extend(h.get("train", []))
        history["valid"].extend(h.get("valid", []))
    return history

def plot_history(history, log_x = False, log_y = False, batches_per_epoch = None, title = "Training and Validation Loss"):
    train = history["train"]
    valid = history["valid"]
    vx = [i for i, v in enumerate(valid) if np.isfinite(v)]
    vy = [v for v in valid if np.isfinite(v)] 
    plt.figure(figsize = (10, 5))
    plt.plot(train, label = "Train Loss")
    plt.plot(vx, vy, label = "Valid Loss")
    if batches_per_epoch is not None and batches_per_epoch > 0:
        total_batches = len(train)
        k = 1
        while True:
            x = k * batches_per_epoch
            if x > total_batches:
                break
            if not (log_x and x == 0):
                plt.axvline(x = x, linestyle = "--", linewidth = 1, alpha = 0.5)
            k += 1
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()