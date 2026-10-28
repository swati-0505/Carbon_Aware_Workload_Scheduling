import argparse
import os
import numpy as np
import pandas as pd
from src.utils import metrics
from . import baseline_scheduler, job_generator, rule_based_scheduler
from .baseline_scheduler import earliest_feasible
from .rl_env import N_ACTIONS, StoppingEnv, ci_percentile, encode_state

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
TRAIN_END = pd.Timestamp("2022-12-31 23:00")
REGION_NAMES = {"india": "India", "uk": "UK", "gb": "UK"}


def train_q(env, episodes=60000, alpha=0.1, eps_start=1.0, eps_end=0.05, seed=0):
    """Q-learning with decaying epsilon-greedy exploration (no discounting:
    episodes are short and end when the job starts)."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, N_ACTIONS))
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1.0 - ep / (0.7 * episodes)))
        s = env.reset()
        done = False
        while not done:
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, done = env.step(a)
            target = r if done else r + Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
    return Q


def schedule(jobs, ci, capacity_kw, Q):
    """Capacity-aware schedule using the learned start/wait policy.
    Returns (starts, load)."""
    ci = np.asarray(ci, dtype=float)
    n = len(ci)
    pct = ci_percentile(ci)
    load = np.zeros(n)
    arr = jobs["arrival"].values
    dur = jobs["duration"].values
    pw = jobs["power_kw"].values
    sl = jobs["slack"].values
    starts = np.full(len(jobs), -1, dtype=int)

    for i in range(len(jobs)):
        a, d, p, s = int(arr[i]), int(dur[i]), pw[i], int(sl[i])
        st = -1
        if s == 0:
            st = earliest_feasible(load, capacity_kw, p, d, a)
        else:
            for t in range(a, a + s + 1):
                if t + d > n:
                    break
                fits = load[t:t + d].max() + p <= capacity_kw + 1e-9
                if not fits:
                    continue
                state = encode_state(pct[t], a + s - t, t % 24)
                if t == a + s or Q[state].argmax() == 1:
                    st = t
                    break
            if st < 0:
                st = earliest_feasible(load, capacity_kw, p, d, a + s + 1)
        if st >= 0:
            load[st:st + d] += p
            starts[i] = st
    return starts, load


def _load(region):
    path = os.path.join(PROCESSED_DIR, f"{region.lower()}_intensity.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python main.py clean")
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    s = df.set_index("timestamp")["carbon_intensity"].astype(float)
    if s.isna().any() or len(s.index.to_series().diff().dropna().unique()) != 1:
        raise ValueError(f"{path} has gaps. Run: python src\\data\\fix_gaps.py")
    return s


def run(regions=None, episodes=60000, slack_hours=24, flexible_fraction=0.6,
        headroom=1.25, seed=42):
    if not regions:
        regions = ["India", "UK"]
    if isinstance(regions, str):
        regions = [regions]
    regions = [REGION_NAMES[str(r).strip().lower()] for r in regions]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for region in regions:
        s = _load(region)
        train_end = TRAIN_END
        if s.index[-1] <= TRAIN_END + pd.Timedelta(days=90):
            train_end = s.index[int(0.8 * len(s))]
        pos = s.index.get_loc(train_end)

        env = StoppingEnv(s.values[:pos + 1], slack_hours=slack_hours, seed=seed)
        print(f"\n[{region}] training Q-learning for {episodes} episodes "
              f"on data up to {train_end} ...")
        Q = train_q(env, episodes=episodes, seed=seed)

        
        ci_test = s.values[pos + 1 - 168:]
        jobs = job_generator.generate_jobs(
            len(ci_test), flexible_fraction=flexible_fraction,
            slack_hours=slack_hours, seed=seed)
        cap = headroom * baseline_scheduler.immediate_load(jobs, len(ci_test)).max()
        base_starts, _ = baseline_scheduler.schedule(jobs, len(ci_test), cap)
        base = metrics.summarize(jobs, base_starts, ci_test)

        results = {"baseline": base}
        for label, pol, kind in [("threshold", "threshold", "oracle"),
                                 ("forecast_seasonal24", "forecast", "seasonal24"),
                                 ("forecast_oracle", "forecast", "oracle")]:
            st, _ = rule_based_scheduler.schedule(jobs, ci_test, cap, policy=pol,
                                                  forecast_kind=kind, seed=seed)
            results[label] = metrics.summarize(jobs, st, ci_test)
        st, _ = schedule(jobs, ci_test, cap, Q)
        results["rl_q_learning"] = metrics.summarize(jobs, st, ci_test)

        print(f"  test jobs: {len(jobs)} | {int(flexible_fraction * 100)}% flexible, "
              f"{slack_hours} h slack")
        for label, res in results.items():
            sav = metrics.savings_pct(base["emissions_kg"], res["emissions_kg"])
            print(f"  {label:22s} saving {sav:6.2f}%   mean delay {res['mean_delay_h']:5.2f} h"
                  f"   violations {res['violations']}")
            rows.append({"region": region, "policy": label,
                         "savings_pct": round(sav, 3),
                         "emissions_kg": round(res["emissions_kg"], 1),
                         "mean_delay_h": round(res["mean_delay_h"], 2),
                         "violations": res["violations"], "n_jobs": res["n_jobs"]})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "rl_results.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/rl_results.csv")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=60000)
    args = ap.parse_args()
    run(args.regions, episodes=args.episodes)