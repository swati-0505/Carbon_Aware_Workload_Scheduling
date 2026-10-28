import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 48

def available():
    try:
        import torch 
        return True
    except ImportError:
        return False

def prepare(ci, horizons, train_end, window=WINDOW):
    """Build windows and targets (numpy only, no torch needed).
    Returns a dict with X (N, window, 5), y (N, len(horizons)) in scaled units,
    the origin timestamps, train/test masks and the scaling constants."""
    values = ci.values.astype(float)
    n = len(values)
    idx = ci.index
    mu = values[idx <= train_end].mean()
    sd = values[idx <= train_end].std()
    scaled = (values - mu) / sd

    hour = idx.hour.values
    dow = idx.dayofweek.values
    feats = np.column_stack([
        scaled,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ])

    maxh = max(horizons)
    n_samples = n - maxh - window + 1
    win = sliding_window_view(feats, window, axis=0)       # (n-window+1, 5, window)
    X = win[:n_samples].transpose(0, 2, 1).astype(np.float32)
    origins = np.arange(window - 1, window - 1 + n_samples)
    y = np.stack([scaled[origins + h] for h in horizons], axis=1).astype(np.float32)

    origin_time = idx[origins]
    train_mask = origin_time <= train_end - pd.Timedelta(hours=maxh)
    test_mask = origin_time > train_end
    return {"X": X, "y": y, "origin_time": origin_time, "train": train_mask,
            "test": test_mask, "mu": mu, "sd": sd}


def fit_predict(ci, horizons, train_end, window=WINDOW, hidden=64, epochs=25,
                batch_size=256, lr=1e-3, seed=0, verbose=True):
    """Train on data up to train_end, predict every test origin.
    Returns {horizon: Series of forecasts indexed by origin time}."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    d = prepare(ci, horizons, train_end, window)
    Xtr, ytr = d["X"][d["train"]], d["y"][d["train"]]
    n_val = max(int(0.1 * len(Xtr)), 1)           # last 10% of training = validation
    Xva, yva = torch.tensor(Xtr[-n_val:]), torch.tensor(ytr[-n_val:])
    Xtr, ytr = torch.tensor(Xtr[:-n_val]), torch.tensor(ytr[:-n_val])

    class Net(nn.Module):
        def __init__(self, n_in, hid, n_out):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hid, batch_first=True)
            self.head = nn.Linear(hid, n_out)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    net = Net(Xtr.shape[2], hidden, len(horizons))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state = float("inf"), None

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xva), yva).item()
        if val < best:
            best = val
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if verbose and (ep + 1) % 5 == 0:
            print(f"      LSTM epoch {ep + 1}/{epochs}  val MSE (scaled) {val:.4f}")

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(d["X"][d["test"]])).numpy()
    pred = pred * d["sd"] + d["mu"]
    t_idx = d["origin_time"][d["test"]]
    return {h: pd.Series(pred[:, j], index=t_idx) for j, h in enumerate(horizons)}