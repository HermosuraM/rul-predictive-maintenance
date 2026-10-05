"""
Remaining Useful Life estimation on NASA C-MAPSS (FD001).

Pipeline:  Gaussian HMM health states -> BiLSTM RUL regressor
           -> split conformal intervals -> asymmetric cost decision layer

Data:
    NASA "Turbofan Engine Degradation Simulation" (C-MAPSS), FD001 subset.
    Available from the NASA Prognostics Data Repository, and mirrored on
    Kaggle as "NASA CMAPSS Jet Engine Simulated Data". You need three files:
        train_FD001.txt   test_FD001.txt   RUL_FD001.txt

Setup:
    pip install torch hmmlearn scikit-learn pandas numpy

Usage:
    python rul_pipeline.py --data-dir ./CMAPSSData

Prints test RMSE, NASA score, empirical conformal coverage, and cost
reduction versus a fixed-interval preventive schedule.
"""
import argparse, os, numpy as np, pandas as pd, torch, torch.nn as nn
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler

SEED = 42
RUL_CAP = 125          # standard piecewise-linear RUL clip for C-MAPSS
WINDOW = 30            # cycles per input sequence
N_STATES = 3           # healthy / degrading / critical

COLS = (['unit', 'cycle'] + [f'op{i}' for i in (1, 2, 3)]
        + [f's{i}' for i in range(1, 22)])
# Sensors that are constant in FD001 and carry no information.
DEAD = ['s1', 's5', 's6', 's10', 's16', 's18', 's19', 'op1', 'op2', 'op3']


def load(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    return df


def add_rul(df, cap=RUL_CAP):
    last = df.groupby('unit')['cycle'].transform('max')
    return df.assign(rul=(last - df['cycle']).clip(upper=cap))


def engineer(df, sensors):
    """Rolling means, EWMA trends, and per-unit drift from the first cycles."""
    out = [df]
    g = df.groupby('unit')[sensors]
    out.append(g.rolling(5, min_periods=1).mean()
               .reset_index(level=0, drop=True).add_suffix('_roll5'))
    out.append(g.apply(lambda x: x.ewm(span=10, adjust=False).mean())
               .reset_index(level=0, drop=True).add_suffix('_ewm10'))
    base = df.groupby('unit')[sensors].transform(lambda x: x.head(5).mean())
    out.append((df[sensors] - base).add_suffix('_drift'))
    return pd.concat(out, axis=1)


def fit_hmm(df, feats, seed=SEED):
    X = df[feats].to_numpy()
    lengths = df.groupby('unit', sort=False).size().to_numpy()
    hmm = GaussianHMM(n_components=N_STATES, covariance_type='diag',
                      n_iter=40, random_state=seed)
    hmm.fit(X, lengths)
    return hmm


def hmm_posteriors(hmm, df, feats, order):
    X = df[feats].to_numpy()
    lengths = df.groupby('unit', sort=False).size().to_numpy()
    post, i = [], 0
    for L in lengths:
        post.append(hmm.predict_proba(X[i:i + L]))
        i += L
    return np.vstack(post)[:, order]


def windows(df, feats, window=WINDOW, with_target=True):
    """Sliding windows within each unit. Short units are left-padded."""
    Xs, ys, units = [], [], []
    for u, g in df.groupby('unit', sort=False):
        arr = g[feats].to_numpy(dtype=np.float32)
        tgt = g['rul'].to_numpy(dtype=np.float32) if with_target else None
        if len(arr) < window:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr = np.vstack([pad, arr])
            if tgt is not None:
                tgt = np.concatenate([np.repeat(tgt[:1], window - len(tgt)), tgt])
        for e in range(window, len(arr) + 1):
            Xs.append(arr[e - window:e])
            units.append(u)
            if tgt is not None:
                ys.append(tgt[e - 1])
    X = np.stack(Xs)
    return (X, np.array(ys, dtype=np.float32), np.array(units)) if with_target \
        else (X, None, np.array(units))


class BiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def train(model, X, y, epochs=25, bs=256, lr=1e-3, val=None, device='cpu'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    lossf = nn.MSELoss()
    Xt = torch.tensor(X, device=device); yt = torch.tensor(y, device=device)
    n = len(Xt)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        msg = f'epoch {ep:3d}  train RMSE {np.sqrt(tot / n):7.3f}'
        if val is not None:
            vp = predict(model, val[0], device)
            vr = float(np.sqrt(np.mean((vp - val[1]) ** 2)))
            sched.step(vr)
            msg += f'   val RMSE {vr:7.3f}'
        print(msg, flush=True)
    return model


@torch.no_grad()
def predict(model, X, device='cpu', bs=512):
    model.eval()
    out = [model(torch.tensor(X[i:i + bs], device=device)).cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def nasa_score(err):
    """Official C-MAPSS score: late predictions penalised harder than early."""
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1,
                                 np.exp(err / 10) - 1)))


def conformal_q(cal_pred, cal_true, alpha=0.10):
    n = len(cal_true)
    scores = np.abs(cal_pred - cal_true)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


def fixed_interval_cost(interval, true, c_down, c_early):
    sched = np.full_like(true, interval, dtype=float)
    return float(np.sum(np.where(sched > true, c_down, c_early * (true - sched))))


def cost_analysis(pred, true, q, cal_true, ratio=100.0, c_early=1.0):
    """
    Model policy: schedule at the lower conformal bound (pred - q).
    Baseline:     one fixed interval for every unit, no model. The interval is
                  tuned on the CALIBRATION units, not on test, so the baseline
                  is fair rather than oracle.
    Costs:        every cycle of unused remaining life costs c_early; an
                  unplanned failure (scheduled later than true RUL) costs
                  `ratio` times that, i.e. ratio=100 means one unplanned
                  failure is as expensive as discarding 100 good cycles.
                  This ratio drives the headline number -- state it whenever
                  you quote the result.
    """
    c_down = ratio * c_early
    sched = np.maximum(pred - q, 0)
    policy = float(np.sum(np.where(sched > true, c_down,
                                   c_early * (true - sched))))
    grid = np.arange(0, int(cal_true.max()) + 1)
    best = min(grid, key=lambda g: fixed_interval_cost(g, cal_true, c_down, c_early))
    base = fixed_interval_cost(best, true, c_down, c_early)
    fails_model = int(np.sum(sched > true))
    fails_base = int(np.sum(best > true))
    return (policy, base, best, 100.0 * (base - policy) / base,
            fails_model, fails_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='CMAPSSData')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--alpha', type=float, default=0.10)
    a = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    tr = add_rul(load(os.path.join(a.data_dir, 'train_FD001.txt')))
    te = load(os.path.join(a.data_dir, 'test_FD001.txt'))
    rul_true = pd.read_csv(os.path.join(a.data_dir, 'RUL_FD001.txt'),
                           sep=r'\s+', header=None).iloc[:, 0].to_numpy()

    sensors = [c for c in COLS if c.startswith('s') and c not in DEAD]
    print(f'units: train {tr.unit.nunique()}  test {te.unit.nunique()}   '
          f'informative sensors: {len(sensors)}')

    scaler = MinMaxScaler().fit(tr[sensors])
    tr[sensors] = scaler.transform(tr[sensors])
    te[sensors] = scaler.transform(te[sensors])

    trf = engineer(tr, sensors)
    tef = engineer(te, sensors)
    feats = [c for c in trf.columns if c not in ('unit', 'cycle', 'rul')
             and c not in DEAD]

    # --- HMM health states -------------------------------------------------
    hmm = fit_hmm(trf, sensors)
    # Order states by mean RUL so index 0 = critical, 2 = healthy.
    st = hmm.predict(trf[sensors].to_numpy(),
                     trf.groupby('unit', sort=False).size().to_numpy())
    order = np.argsort([trf['rul'].to_numpy()[st == k].mean()
                        for k in range(N_STATES)])
    for name, frame in (('train', trf), ('test', tef)):
        p = hmm_posteriors(hmm, frame, sensors, order)
        for k in range(N_STATES):
            frame[f'state{k}'] = p[:, k]
    feats += [f'state{k}' for k in range(N_STATES)]
    print(f'feature count: {len(feats)}')

    # --- windows, with a unit-disjoint calibration split -------------------
    rng = np.random.default_rng(SEED)
    units = trf.unit.unique()
    cal_units = set(rng.choice(units, size=max(1, len(units) // 5), replace=False))
    fit_df = trf[~trf.unit.isin(cal_units)]
    cal_df = trf[trf.unit.isin(cal_units)]

    Xf, yf, _ = windows(fit_df, feats)
    Xc, yc, _ = windows(cal_df, feats)
    print(f'windows: fit {len(Xf)}  calibration {len(Xc)}')

    model = BiLSTM(len(feats)).to(dev)
    train(model, Xf, yf, epochs=a.epochs, val=(Xc, yc), device=dev)

    # --- conformal calibration --------------------------------------------
    q = conformal_q(predict(model, Xc, dev), yc, alpha=a.alpha)

    # --- test set: last window per unit ------------------------------------
    Xt, _, ut = windows(tef.assign(rul=0), feats)
    last = {u: i for i, u in enumerate(ut)}          # last occurrence wins
    idx = np.array([last[u] for u in sorted(last)])
    pred = np.clip(predict(model, Xt[idx], dev), 0, RUL_CAP)
    true = np.minimum(rul_true, RUL_CAP)

    os.makedirs('results', exist_ok=True)
    np.savez('results/preds.npz', cal_pred=predict(model, Xc, dev), cal_true=yc,
             test_pred=pred, test_true=true, rul_raw=rul_true)
    torch.save(model.state_dict(), 'model.pt')

    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    cover = float(np.mean(np.abs(err) <= q)) * 100
    (policy, base, best_iv, saved, f_mod, f_base) = cost_analysis(pred, true, q, yc)

    print('\n' + '=' * 62)
    print('RESULTS')
    print('=' * 62)
    print(f'  test RMSE                : {rmse:.1f} cycles')
    print(f'  NASA score               : {nasa_score(err):.0f}')
    print(f'  conformal target coverage: {100 * (1 - a.alpha):.0f}%')
    print(f'  empirical coverage       : {cover:.1f}%  (interval +/- {q:.1f} cycles)')
    print(f'  cost vs fixed-interval   : {saved:.0f}% lower  '
          f'({policy:.0f} vs {base:.0f}, at a 100:1 failure/early-swap ratio;'
          f' best fixed interval = {best_iv} cycles)')
    print(f'  unplanned failures       : {f_mod} (model) vs {f_base} (fixed) '
          f'out of {len(true)} engines')
    print('=' * 62)


if __name__ == '__main__':
    main()
