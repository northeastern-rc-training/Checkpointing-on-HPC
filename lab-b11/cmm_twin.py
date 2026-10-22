"""Helpers for the CMM thread digital-twin checkpointing lab.

Data model
----------
One *run* = one CMM measurement of one thread with one configuration:
(tip, hole, method, direction, number of hits). Every point is converted to
cylindrical coordinates around the hole axis, so the thread is described by
  theta (angle), z (height), and the deviations of the measured point from
  the theoretical one: dr (radial) and dz (axial).

The "digital twin" is a model that predicts the healthy deviation pattern
(dr, dz) of a thread from its configuration and position. A measured point
that disagrees strongly with the twin, or an expected point that is missing,
marks a place where the thread was lost.
"""
import os
import json
import time
import subprocess

import joblib
import numpy as np
import pandas as pd

PITCH = 0.5              # mm, M3x0.5 thread in the measured part
TOLERANCE = 0.127        # mm, drawing tolerance used in the original analysis
METHODS = {1.1: "TT", 2.1: "SCAN"}      # 3.1 = thread locator (outer feature), excluded


# ---------------------------------------------------------------------------
# Time / checkpoint helpers
# ---------------------------------------------------------------------------
def seconds_left():
    """Remaining wall time of the current Slurm job (OnDemand or batch)."""
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return float("inf")
    try:
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return float("inf")
    if not out or out in ("UNLIMITED", "NOT_SET", "INVALID"):
        return float("inf")
    days, _, hms = out.rpartition("-")
    parts = [int(p) for p in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + s


class TimeGuard:
    """Decides when to stop: Slurm time left, and/or an explicit budget."""

    def __init__(self, safety_seconds=180, budget_minutes=None):
        self.safety = safety_seconds
        self.t0 = time.time()
        self.budget = budget_minutes * 60 if budget_minutes else None
        self.stopped = False
        self._last_check = 0.0
        self._cached_left = float("inf")

    def ok(self):
        if self.stopped:
            return False
        if self.budget is not None and time.time() - self.t0 > self.budget:
            self.stopped = True
            return False
        # squeue is cheap but not free; ask at most every 30 s
        now = time.time()
        if now - self._last_check > 30:
            self._cached_left = seconds_left()
            self._last_check = now
        if self._cached_left - (now - self._last_check) < self.safety:
            self.stopped = True
            return False
        return True


def atomic_dump(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    joblib.dump(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Loading and feature engineering
# ---------------------------------------------------------------------------
def load_points(csv_path):
    """Read the raw CMM export and add cylindrical coordinates and deviations."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Tip_Dim-s": "Tip", "HIT#": "HIT",
                            "HOLE#_ISO_mm": "HOLE", "TT_SCAN_TL": "METHOD_CODE"})
    df = df[df.METHOD_CODE.isin(METHODS)].copy()
    df["METHOD"] = df.METHOD_CODE.map(METHODS)
    df["IDX"] = df.HIT.str.split("_").str[2].astype(int)
    df["HOLE"] = df.HOLE.astype(int)
    return add_geometry(df)


def add_geometry(df):
    dx_t, dy_t = df.Theo_X - df.Feat_X, df.Theo_Y - df.Feat_Y
    dx_a, dy_a = df.Act_X - df.Feat_X, df.Act_Y - df.Feat_Y
    df["theta"] = np.arctan2(dy_t, dx_t)                 # rad, theoretical angle
    df["r_theo"] = np.hypot(dx_t, dy_t)
    df["dr"] = np.hypot(dx_a, dy_a) - df.r_theo          # radial deviation (mm)
    df["dz"] = df.Act_Z - df.Theo_Z                      # axial deviation (mm)
    cfg = "|" + df.METHOD + "|" + df.DIR + "|N" + df.NUMHITS.astype(str)
    df["RUN"] = df.Tip + "|H" + df.HOLE.astype(str) + cfg
    # measured run a point comes from (augmented copies keep their source run);
    # used to keep copies of the same measurement in the same CV fold
    src_hole = df["SRC_HOLE"] if "SRC_HOLE" in df else df.HOLE
    df["SRC_RUN"] = df.Tip + "|H" + src_hole.astype(int).astype(str) + cfg
    df = df.sort_values(["RUN", "IDX"]).reset_index(drop=True)
    # unwrapped angle along the probing path, starting at 0 for every run (deg)
    df["u"] = df.groupby("RUN")["theta"].transform(
        lambda t: np.abs(np.degrees(np.unwrap(t.values) - t.values[0])))
    return df


def augment_holes(df, n_new, seed=0):
    """Create extra *healthy* threads from the measured ones.

    Each new hole copies all runs of a measured hole, rotates the deviation
    pattern by a random angle, scales its amplitude by 0.8-1.2 and adds
    measurement noise. This scales the dataset (and the training time) while
    keeping the deviation shapes learned from real measurements.
    """
    if n_new <= 0:
        return df
    rng = np.random.default_rng(seed)
    holes = sorted(df.HOLE.unique())
    noise = {"TT": 0.004, "SCAN": 0.003}
    parts = [df]
    for k in range(n_new):
        src = holes[k % len(holes)]
        new = df[df.HOLE == src].copy()
        phi = rng.uniform(0, 2 * np.pi)
        scale = rng.uniform(0.8, 1.2)
        n = len(new)
        sig = new.METHOD.map(noise).values
        # rotate everything about the hole axis
        for p in ("Theo", "Act"):
            x, y = new[f"{p}_X"] - new.Feat_X, new[f"{p}_Y"] - new.Feat_Y
            new[f"{p}_X"] = new.Feat_X + x * np.cos(phi) - y * np.sin(phi)
            new[f"{p}_Y"] = new.Feat_Y + x * np.sin(phi) + y * np.cos(phi)
        # rescale radial deviation + noise, rebuild actual XYZ
        r_t = np.hypot(new.Theo_X - new.Feat_X, new.Theo_Y - new.Feat_Y)
        th_a = np.arctan2(new.Act_Y - new.Feat_Y, new.Act_X - new.Feat_X)
        dr0 = np.hypot(new.Act_X - new.Feat_X, new.Act_Y - new.Feat_Y) - r_t
        r_a = r_t + scale * dr0 + rng.normal(0, sig, n)
        new["Act_X"] = new.Feat_X + r_a * np.cos(th_a)
        new["Act_Y"] = new.Feat_Y + r_a * np.sin(th_a)
        new["Act_Z"] = new.Act_Z + rng.normal(0, sig / 3, n)
        new["SRC_HOLE"] = src
        new["HOLE"] = 100 + k
        parts.append(new)
    out = pd.concat(parts, ignore_index=True)
    out["SRC_HOLE"] = out["SRC_HOLE"].fillna(out.HOLE).astype(int)
    return add_geometry(out.drop(columns=["theta", "r_theo", "dr", "dz", "RUN", "SRC_RUN", "u"]))


class FeatureBuilder:
    """Turns points into model inputs. Categories are fixed at fit time."""

    def fit(self, df):
        self.tips = sorted(df.Tip.unique())
        self.methods = sorted(df.METHOD.unique())
        self.holes = sorted(df.HOLE.unique())
        self.z_mean, self.z_std = df.Theo_Z.mean(), df.Theo_Z.std()
        return self

    def transform(self, df):
        th = df.theta.values
        cols = [np.sin(k * th) for k in (1, 2, 3)] + [np.cos(k * th) for k in (1, 2, 3)]
        cols.append(((df.Theo_Z - self.z_mean) / self.z_std).values)
        cols.append(np.where(df.DIR.values == "CW", 1.0, -1.0))
        cols.append(np.log(df.NUMHITS.values.astype(float)))
        for name, cats in (("Tip", self.tips), ("METHOD", self.methods), ("HOLE", self.holes)):
            v = df[name].values
            cols += [(v == c).astype(float) for c in cats]
        return np.column_stack(cols)


def targets(df):
    return df[["dr", "dz"]].values


# ---------------------------------------------------------------------------
# Defect injection (ground truth for the study)
# ---------------------------------------------------------------------------
def inject_defect(run, seed, amp_range=(0.15, 0.35), mode=None, length=None, amp=None):
    """Return (defective_run, truth) for one healthy run.

    modes:
      lost_flank   - part of the thread is gone: the probe lands deeper, radial
                     deviation jumps by `amp` mm over the segment
      missing_data - the CMM recorded no points over the segment
      both         - lost flank, with the middle of the segment unrecorded
    mode, length (deg) and amp (mm) are random (seeded) unless given.
    """
    rng = np.random.default_rng(seed)
    if mode is None:
        mode = rng.choice(["lost_flank", "missing_data", "both"])
    span = run.u.max()
    if length is None:
        length = rng.uniform(45, 150)
    length = min(float(length), 0.9 * span)
    start = rng.uniform(0.05 * span, max(0.05 * span, span - length - 0.05 * span))
    end = start + length
    run = run.copy()
    inside = (run.u >= start) & (run.u <= end)
    if amp is None:
        amp = rng.uniform(*amp_range)
    if mode in ("lost_flank", "both"):
        w = np.sin(np.pi * (run.u - start) / length).clip(0) ** 0.5 * inside
        run["dr"] = run.dr + amp * w
        run["dz"] = run.dz + 0.2 * amp * w * rng.choice([-1, 1])
    if mode in ("missing_data", "both"):
        drop = inside if mode == "missing_data" else (
            (run.u >= start + 0.3 * length) & (run.u <= end - 0.3 * length))
        run = run[~drop]
    truth = {"mode": str(mode), "start": float(start), "end": float(end),
             "amp": float(amp), "length": float(length)}
    return run.reset_index(drop=True), truth


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect(run, pred, sigma, k_sigma=6.0, gap_factor=2.5, merge_deg=30.0):
    """Find segments where the thread deviates from the twin or data is missing.

    run   : points of one run (sorted by u)
    pred  : twin prediction (n, 2) for dr, dz
    sigma : (sigma_dr, sigma_dz) of healthy residuals for this method
    """
    res = run[["dr", "dz"]].values - pred
    score = np.sqrt((res[:, 0] / sigma[0]) ** 2 + (res[:, 1] / sigma[1]) ** 2)
    u = run.u.values
    flagged = score > k_sigma
    segs = []
    # deviation segments
    idx = np.flatnonzero(flagged)
    if len(idx):
        start = prev = idx[0]
        for i in idx[1:]:
            if u[i] - u[prev] > merge_deg:
                segs.append(("deviation", u[start], u[prev]))
                start = i
            prev = i
        segs.append(("deviation", u[start], u[prev]))
    # coverage gaps
    du = np.diff(u)
    if len(du) > 2:
        med = np.median(du)
        for i in np.flatnonzero(du > gap_factor * med):
            segs.append(("missing", u[i], u[i + 1]))
    return segs, score, flagged


def score_case(segs, truth, spacing):
    """Compare detected segments with the injected truth (angles in deg)."""
    if truth is None:
        return {"detected": None, "iou": None, "false_alarms": len(segs)}
    t0, t1 = truth["start"], truth["end"]
    pad = spacing
    hits = [s for s in segs if s[2] >= t0 - pad and s[1] <= t1 + pad]
    false = [s for s in segs if s not in hits]
    if not hits:
        return {"detected": False, "iou": 0.0, "false_alarms": len(false)}
    d0, d1 = min(s[1] for s in hits), max(s[2] for s in hits)
    inter = max(0.0, min(t1, d1) - max(t0, d0))
    union = max(t1, d1) - min(t0, d0)
    return {"detected": True, "iou": inter / union if union > 0 else 0.0,
            "false_alarms": len(false)}


def helix_from_run(run):
    """Linear model u -> (theta, z) of the theoretical helix of one run."""
    u = run.u.values
    th = np.unwrap(run.theta.values)
    a_th = np.polyfit(u, th, 1)
    a_z = np.polyfit(u, run.Theo_Z.values, 1)
    return a_th, a_z


# ---------------------------------------------------------------------------
# Prediction, case analysis and 3D figures
# ---------------------------------------------------------------------------
def predict(bundle, fb, df):
    """Twin prediction (dr, dz) for points in df. bundle: dict(model, xs, ys)."""
    X = bundle["xs"].transform(fb.transform(df))
    return bundle["ys"].inverse_transform(bundle["model"].predict(X))


def dense_helix(run, n=500):
    """Points along the theoretical helix of a run, with run metadata."""
    a_th, a_z = helix_from_run(run)
    u = np.linspace(0.0, run.u.max(), n)
    th = np.polyval(a_th, u)
    d = pd.DataFrame({"u": u, "theta": np.arctan2(np.sin(th), np.cos(th)),
                      "Theo_Z": np.polyval(a_z, u)})
    first = run.iloc[0]
    for c in ("DIR", "NUMHITS", "Tip", "METHOD", "HOLE", "Feat_X", "Feat_Y"):
        d[c] = first[c]
    d["r_theo"] = run.r_theo.median()
    return d


def analyze_case(run, bundle, fb, sigma, seed, k_sigma=6.0, **inject_kwargs):
    """Inject a defect into one healthy run, run the twin, detect, score.

    Returns a dict with a JSON-serialisable 'record' plus the arrays needed
    for plotting.
    """
    run = run.sort_values("u").reset_index(drop=True)
    spacing = float(np.median(np.diff(run.u.values))) if len(run) > 1 else 360.0
    # healthy run: false alarms only
    h_segs, _, _ = detect(run, predict(bundle, fb, run), sigma, k_sigma=k_sigma)
    # defective run
    bad, truth = inject_defect(run, seed, **inject_kwargs)
    pred = predict(bundle, fb, bad)
    segs, score, flagged = detect(bad, pred, sigma, k_sigma=k_sigma)
    s = score_case(segs, truth, spacing)
    first = run.iloc[0]
    record = {
        "run": first.RUN, "method": first.METHOD, "tip": first.Tip,
        "hole": int(first.HOLE), "dir": first.DIR, "numhits": int(first.NUMHITS),
        "n_points": int(len(run)), "spacing_deg": spacing, "seed": int(seed),
        "mode": truth["mode"], "start_deg": truth["start"], "end_deg": truth["end"],
        "detected": bool(s["detected"]), "iou": float(s["iou"]),
        "false_alarms": int(s["false_alarms"]),
        "healthy_false_alarms": int(len(h_segs)),
        "segments": [[t, float(a), float(b)] for t, a, b in segs],
    }
    return {"record": record, "run": run, "defective": bad, "pred": pred,
            "flagged": flagged, "segments": segs, "truth": truth}


def _xyz(feat_x, feat_y, r, theta, z):
    return feat_x + r * np.cos(theta), feat_y + r * np.sin(theta), z


def case_figure(cases, bundle_for, fb, titles=None, exaggerate=8.0):
    """Interactive 3D comparison of analysed cases (plotly).

    green  : theoretical helix (drawing)
    grey   : healthy thread as the digital twin expects it
    blue   : measured points
    red    : measured points that disagree with the twin (lost flank) and the
             twin's reconstruction of where thread is missing or lost
    Deviations are exaggerated radially/axially so they are visible.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(cases)
    titles = titles or [c["record"]["run"] for c in cases]
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "scene"}] * n],
                        subplot_titles=titles, horizontal_spacing=0.01)
    for j, c in enumerate(cases, start=1):
        run, bad = c["run"], c["defective"]
        fx, fy = run.Feat_X.iloc[0], run.Feat_Y.iloc[0]
        r0 = run.r_theo.median()
        dense = dense_helix(run)
        dpred = predict(bundle_for(c), fb, dense)
        show = j == 1

        x, y, z = _xyz(fx, fy, r0, dense.theta, dense.Theo_Z)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Theoretical helix",
                                   line=dict(color="green", width=4), showlegend=show,
                                   legendgroup="theo"), 1, j)
        x, y, z = _xyz(fx, fy, r0 + exaggerate * dpred[:, 0], dense.theta,
                       dense.Theo_Z + exaggerate * dpred[:, 1])
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Digital twin (expected)",
                                   line=dict(color="lightgray", width=3, dash="dash"),
                                   showlegend=show, legendgroup="twin"), 1, j)
        ok = ~c["flagged"]
        x, y, z = _xyz(fx, fy, r0 + exaggerate * bad.dr, bad.theta,
                       bad.Theo_Z + exaggerate * bad.dz)
        fig.add_trace(go.Scatter3d(x=x[ok], y=y[ok], z=z[ok], mode="markers",
                                   name="Measured", marker=dict(size=2.5, color="royalblue"),
                                   showlegend=show, legendgroup="meas"), 1, j)
        fig.add_trace(go.Scatter3d(x=x[~ok], y=y[~ok], z=z[~ok], mode="markers",
                                   name="Measured, off-twin (lost flank)",
                                   marker=dict(size=4, color="red"),
                                   showlegend=show, legendgroup="bad"), 1, j)
        # twin reconstruction over every detected segment
        first = True
        for kind, a, b in c["segments"]:
            m = (dense.u >= a) & (dense.u <= b)
            if m.sum() < 2:
                continue
            x, y, z = _xyz(fx, fy, r0 + exaggerate * dpred[m, 0], dense.theta[m],
                           dense.Theo_Z[m] + exaggerate * dpred[m, 1])
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
                                       name="Missing / lost thread (twin reconstruction)",
                                       line=dict(color="red", width=9),
                                       showlegend=show and first, legendgroup="miss"), 1, j)
            first = False
    fig.update_scenes(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)",
                      zaxis_title="Z (mm)")
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", y=-0.05),
                      title=f"Thread digital twin: deviations exaggerated x{exaggerate:g}")
    return fig


def prepare_data(csv_path, n_aug, seed, folds):
    """Stage 1: load, augment, and assign cross-validation folds.

    Folds are assigned by *source measurement*, so a run and its augmented
    copies never sit on both sides of a train/test split. Wrapped with
    joblib.Memory in the notebook, so it is computed once and then cached.
    """
    from sklearn.model_selection import GroupKFold
    df = load_points(csv_path)
    df = augment_holes(df, n_aug, seed=seed)
    fold_of = {}
    for f, (_, va) in enumerate(GroupKFold(n_splits=folds).split(df, groups=df.SRC_RUN)):
        for r in df.SRC_RUN.values[va]:
            fold_of[r] = f
    df["FOLD"] = df.SRC_RUN.map(fold_of)
    return df
