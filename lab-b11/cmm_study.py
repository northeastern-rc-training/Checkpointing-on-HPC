"""Analysis helpers for the four-part thread study.

  A. Process   - size, form and drill-vs-tap alignment of each threaded hole
  B. CNC       - true hole positions from the calibrated thread locator
  C. Defects   - categorisation of every measured point, search for local
                 defects, and simulated-defect detection limits per setup
  D. CMM       - tip, mode, direction and number of hits against the truth

Conventions: deviations are measured - nominal; offsets in micrometres (um);
angles in degrees; the hole axis is +Z (Feat_k = 1).
"""
import numpy as np
import pandas as pd

import cmm_twin as ct

TOL = ct.TOLERANCE  # 0.127 mm


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(csv_path):
    raw = pd.read_csv(csv_path).rename(columns={
        "Tip_Dim-s": "Tip", "HIT#": "HIT", "HOLE#_ISO_mm": "HOLE", "TT_SCAN_TL": "METHOD_CODE"})
    raw["METHOD"] = raw.METHOD_CODE.map({1.1: "TT", 2.1: "SCAN", 3.1: "TL"})
    return raw


def nominal_centres(raw):
    """Nominal hole centres (inner-thread feature) per hole."""
    return raw[raw.METHOD != "TL"].groupby("HOLE")[["Feat_X", "Feat_Y", "Feat_Z"]].first()


# ---------------------------------------------------------------------------
# Geometry fits
# ---------------------------------------------------------------------------
def fit_circle(x, y):
    """Least-squares circle (algebraic start, Gauss-Newton refinement).
    Returns cx, cy, r, form (std of radial residuals)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([x, y, np.ones_like(x)])
    c, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx ** 2 + cy ** 2)
    for _ in range(25):
        d = np.hypot(x - cx, y - cy)
        J = np.column_stack([-(x - cx) / d, -(y - cy) / d, -np.ones_like(x)])
        step, *_ = np.linalg.lstsq(J, -(d - r), rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-10:
            break
    res = np.hypot(x - cx, y - cy) - r
    return cx, cy, r, res.std()


def thread_locator_reference(raw):
    """True hole centres from the thread locator (one circle per hole and tip).

    The export contains 8 points at a single height per locator (normals
    horizontal), so it gives the centre at that height but not the axis.
    """
    nom = nominal_centres(raw)
    rows = []
    for (h, tip), g in raw[raw.METHOD == "TL"].groupby(["HOLE", "Tip"]):
        cx, cy, r, form = fit_circle(g.Act_X, g.Act_Y)
        rows.append({"hole": h, "tip": tip, "cx": cx, "cy": cy, "r": r,
                     "form_um": form * 1000, "z": g.Act_Z.mean(),
                     "z_span_um": (g.Act_Z.max() - g.Act_Z.min()) * 1000,
                     "k_max": g.k.abs().max()})
    per_tip = pd.DataFrame(rows).join(nom, on="hole")
    per_tip["off_x_um"] = (per_tip.cx - per_tip.Feat_X) * 1000
    per_tip["off_y_um"] = (per_tip.cy - per_tip.Feat_Y) * 1000
    ref = per_tip.groupby("hole").agg(cx=("cx", "mean"), cy=("cy", "mean"), z=("z", "mean"),
                                      off_x_um=("off_x_um", "mean"), off_y_um=("off_y_um", "mean"))
    spread = per_tip.join(ref[["cx", "cy"]], on="hole", rsuffix="_ref")
    spread = np.hypot(spread.cx - spread.cx_ref, spread.cy - spread.cy_ref) * 1000
    ref["tip_spread_um"] = spread.groupby(per_tip.hole).max()
    ref["offset_um"] = np.hypot(ref.off_x_um, ref.off_y_um)
    ref["offset_dir_deg"] = np.degrees(np.arctan2(ref.off_y_um, ref.off_x_um))
    ref["true_position_um"] = 2 * ref.offset_um          # diametral position value
    return per_tip, ref.join(nom)


def run_fits(df, robust_iter=3):
    """Per-run fit of the radial deviation:
        dr = size + ex*cos(t) + ey*sin(t) + ovality(2t) + trilobe(3t) + taper*z
    ex, ey = offset of the probed centre from nominal. Robust: points far from
    the fit are down-weighted for the final pass."""
    rows, resid = [], []
    for run, g in df.groupby("RUN", sort=True):
        g = g.sort_values("u")
        th = np.unwrap(g.theta.values)
        z = g.Theo_Z.values - g.Theo_Z.values.mean()
        cols = [np.ones_like(th), np.cos(th), np.sin(th)]
        if len(g) >= 12:
            cols += [np.cos(2 * th), np.sin(2 * th), np.cos(3 * th), np.sin(3 * th), z]
        A = np.column_stack(cols)
        y = g.dr.values
        w = np.ones(len(g), bool)
        for _ in range(robust_iter):
            c, *_ = np.linalg.lstsq(A[w], y[w], rcond=None)
            res = y - A @ c
            sig = max(1.4826 * np.median(np.abs(res[w] - np.median(res[w]))), 0.002)
            w = np.abs(res) < 4 * sig
        c = np.pad(c, (0, 8 - len(c)))
        hand = np.sign(np.polyfit(th, g.Theo_Z.values, 1)[0])
        f = g.iloc[0]
        rows.append({
            "run": run, "tip": f.Tip, "hole": int(f.HOLE), "method": f.METHOD, "dir": f.DIR,
            "numhits": int(f.NUMHITS), "n_points": len(g),
            "program_hand": "right" if hand > 0 else "left",
            "size_um": c[0] * 1000, "ex_um": c[1] * 1000, "ey_um": c[2] * 1000,
            "ovality_um": np.hypot(c[3], c[4]) * 1000, "trilobe_um": np.hypot(c[5], c[6]) * 1000,
            "taper_um_per_mm": c[7] * 1000, "noise_um": sig * 1000,
            "max_abs_z": float(np.abs(res).max() / sig), "z_mean": g.Theo_Z.mean(),
        })
        resid.append(pd.DataFrame({"index": g.index, "fit_res_um": res * 1000, "fit_z": res / sig}))
    fits = pd.DataFrame(rows)
    fits["probed_offset_um"] = np.hypot(fits.ex_um, fits.ey_um)
    return fits, pd.concat(resid).set_index("index")


def categorise_points(raw):
    """Share of the +/-0.127 mm tolerance used by each point (largest axis)."""
    out = raw.copy()
    for a in "XYZ":
        out[f"Dev_{a}"] = out[f"Act_{a}"] - out[f"Theo_{a}"]
    out["tol_used_pct"] = out[["Dev_X", "Dev_Y", "Dev_Z"]].abs().max(axis=1) / TOL * 100
    out["category"] = pd.cut(out.tol_used_pct, [-1, 50, 80, 100, 1e9],
                             labels=["OK", "WATCH", "NEAR_LIMIT", "OUT_OF_TOLERANCE"]).astype(str)
    out.loc[out.METHOD == "TL", "category"] = "REFERENCE_TL"
    return out


def prepare_study(csv_path):
    """Everything Parts A, B and D need; cached in the notebook."""
    raw = load_raw(csv_path)
    pts = ct.load_points(csv_path)                       # inner thread, cylindrical
    tl_tip, tl_ref = thread_locator_reference(raw)
    fits, resid = run_fits(pts)
    pts = pts.join(resid)
    # error of every probed centre against the thread-locator truth
    fits = fits.join(tl_ref[["off_x_um", "off_y_um"]], on="hole")
    fits["centre_err_um"] = np.hypot(fits.ex_um - fits.off_x_um, fits.ey_um - fits.off_y_um)
    cats = categorise_points(raw)
    return {"raw": raw, "points": pts, "tl_tip": tl_tip, "tl_ref": tl_ref,
            "fits": fits, "categories": cats}


# ---------------------------------------------------------------------------
# Simulated detection-limit sweep (Part C)
# ---------------------------------------------------------------------------
def sweep_run(run, bundle, fb, sigma, run_index, amps, lengths, modes, k_sigma=6.0):
    """All simulated defects for one run; returns a list of JSON records."""
    run = run.sort_values("u").reset_index(drop=True)
    spacing = float(np.median(np.diff(run.u.values)))
    h_segs, _, _ = ct.detect(run, ct.predict(bundle, fb, run), sigma, k_sigma=k_sigma)
    first = run.iloc[0]
    out = []
    j = 0
    for mode in modes:
        for length in lengths:
            for amp in (amps if mode != "missing_data" else [0.0]):
                seed = 100000 * (run_index + 1) + j
                j += 1
                bad, truth = ct.inject_defect(run, seed, mode=mode, length=length, amp=amp)
                segs, _, _ = ct.detect(bad, ct.predict(bundle, fb, bad), sigma, k_sigma=k_sigma)
                s = ct.score_case(segs, truth, spacing)
                out.append({"run": first.RUN, "method": first.METHOD, "tip": first.Tip,
                            "hole": int(first.HOLE), "dir": first.DIR, "numhits": int(first.NUMHITS),
                            "spacing_deg": spacing, "mode": mode, "length_deg": float(length),
                            "amp_um": float(amp) * 1000, "seed": int(seed),
                            "detected": bool(s["detected"]), "iou": float(s["iou"]),
                            "false_alarms": int(s["false_alarms"]),
                            "healthy_false_alarms": int(len(h_segs))})
    return out


# ---------------------------------------------------------------------------
# 3D figures (plotly)
# ---------------------------------------------------------------------------
def _helix_xyz(run, value_mm, exaggerate):
    fx, fy = run.Feat_X.iloc[0], run.Feat_Y.iloc[0]
    r = run.r_theo.values + exaggerate * value_mm
    return fx + r * np.cos(run.theta.values), fy + r * np.sin(run.theta.values), run.Theo_Z.values


def fig_holes_tolerance(points, cats, run_filter, exaggerate=6.0):
    """One 3D panel per hole: theoretical helix (green) and measured points
    coloured by the share of tolerance they use."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    holes = sorted(points.HOLE.unique())
    fig = make_subplots(rows=1, cols=len(holes), specs=[[{"type": "scene"}] * len(holes)],
                        subplot_titles=[f"Hole {h}" for h in holes], horizontal_spacing=0.0)
    for j, h in enumerate(holes, start=1):
        run = points[(points.HOLE == h) & run_filter(points)].sort_values("u")
        if run.empty:
            continue
        run = run[run.RUN == run.RUN.iloc[0]]
        x, y, z = _helix_xyz(run, 0 * run.dr.values, 0)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="green", width=4),
                                   name="Theoretical helix", showlegend=j == 1), 1, j)
        x, y, z = _helix_xyz(run, run.dr.values, exaggerate)
        used = (np.abs(run[["Act_X", "Act_Y", "Act_Z"]].values - run[["Theo_X", "Theo_Y", "Theo_Z"]].values)
                .max(axis=1) / TOL * 100)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="Measured (colour = % of tolerance used)",
            marker=dict(size=3, color=used, cmin=0, cmax=100,
                        colorscale=[[0, "green"], [0.5, "gold"], [0.8, "orange"], [1, "red"]],
                        colorbar=dict(title="% tol", len=0.6) if j == len(holes) else None,
                        showscale=j == len(holes)),
            text=[f"{v:.0f}% of tolerance" for v in used], hoverinfo="text",
            showlegend=j == 1), 1, j)
    fig.update_scenes(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0), legend=dict(orientation="h"),
                      title=f"Tolerance used along each thread (radial deviation exaggerated x{exaggerate:g})")
    return fig


def fig_plate(tl_ref, fits, exaggerate=400.0):
    """CNC view: nominal hole positions, true centres (thread locator) and the
    centres found by probing the thread, offsets exaggerated."""
    import plotly.graph_objects as go
    fig = go.Figure()
    probed = fits.groupby("hole")[["ex_um", "ey_um"]].mean()
    s = exaggerate / 1000.0
    for h, r in tl_ref.iterrows():
        nx, ny = r.Feat_X, r.Feat_Y
        tx, ty = nx + s * r.off_x_um, ny + s * r.off_y_um
        px, py = nx + s * probed.loc[h, "ex_um"], ny + s * probed.loc[h, "ey_um"]
        first = bool(h == tl_ref.index[0])
        fig.add_trace(go.Scatter3d(x=[nx], y=[ny], z=[0], mode="markers+text", text=[f"Hole {h}"],
                                   textposition="top center", marker=dict(size=6, color="gray", symbol="cross"),
                                   name="Nominal position", showlegend=first, legendgroup="n"))
        fig.add_trace(go.Scatter3d(x=[nx, tx], y=[ny, ty], z=[0, 0], mode="lines+markers",
                                   line=dict(color="green", width=8), marker=dict(size=[0, 7], color="green"),
                                   name="True centre (thread locator)", showlegend=first, legendgroup="t",
                                   hovertext=f"hole {h}: {r.offset_um:.0f} um at {r.offset_dir_deg:.0f} deg"))
        fig.add_trace(go.Scatter3d(x=[nx, px], y=[ny, py], z=[0, 0], mode="lines+markers",
                                   line=dict(color="royalblue", width=5, dash="dash"),
                                   marker=dict(size=[0, 5], color="royalblue"),
                                   name="Centre from probing the thread", showlegend=first, legendgroup="p"))
        # tolerance circle: +/-0.127 mm, exaggerated the same way
        a = np.linspace(0, 2 * np.pi, 90)
        fig.add_trace(go.Scatter3d(x=nx + s * 127 * np.cos(a), y=ny + s * 127 * np.sin(a), z=0 * a,
                                   mode="lines", line=dict(color="red", width=2),
                                   name="0.127 mm radius (exaggerated)", showlegend=first, legendgroup="tol"))
    fig.update_layout(height=600, title=f"CNC hole positions (offsets exaggerated x{exaggerate:g})",
                      scene=dict(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)",
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, t=50, b=0), legend=dict(orientation="h"))
    return fig


def fig_hole_axes(tl_ref, fits, exaggerate=200.0):
    """Per hole, the line from the probed thread centre (at thread depth) to the
    thread-locator centre (at the locator height): the apparent axis if the
    whole gap were tilt."""
    import plotly.graph_objects as go
    fig = go.Figure()
    probed = fits.groupby("hole").agg(ex=("ex_um", "mean"), ey=("ey_um", "mean"), z=("z_mean", "mean"))
    s = exaggerate / 1000.0
    for h, r in tl_ref.iterrows():
        nx, ny = r.Feat_X, r.Feat_Y
        p = probed.loc[h]
        xs = [nx + s * p.ex, nx + s * r.off_x_um]
        ys = [ny + s * p.ey, ny + s * r.off_y_um]
        zs = [p.z, r.z]
        gap = np.hypot(r.off_x_um - p.ex, r.off_y_um - p.ey)
        tilt = np.degrees(np.arctan2(gap / 1000.0, r.z - p.z))
        first = bool(h == tl_ref.index[0])
        fig.add_trace(go.Scatter3d(x=[nx, nx], y=[ny, ny], z=zs, mode="lines",
                                   line=dict(color="gray", width=3, dash="dot"),
                                   name="Nominal axis", showlegend=first, legendgroup="n"))
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers+text",
                                   line=dict(color="red", width=7), marker=dict(size=5, color=["royalblue", "green"]),
                                   text=["", f"hole {h}: gap {gap:.0f} um = {tilt:.3f} deg if tilt"],
                                   name="Apparent axis (probed thread -> locator)", showlegend=first,
                                   legendgroup="a"))
    fig.update_layout(height=650, title=f"Probed thread centre vs thread-locator centre (lateral x{exaggerate:g})",
                      scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.8),
                                 xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
                      margin=dict(l=0, r=0, t=50, b=0), legend=dict(orientation="h"))
    return fig


def fig_config_overlay(points, hole, configs, exaggerate=8.0):
    """Same hole measured with several configurations, as 3D helices."""
    import plotly.graph_objects as go
    fig = go.Figure()
    palette = ["royalblue", "darkorange", "purple", "crimson", "teal", "olive"]
    base = None
    for i, (label, sel) in enumerate(configs):
        run = points[(points.HOLE == hole) & sel(points)]
        if run.empty:
            continue
        run = run[run.RUN == sorted(run.RUN.unique())[0]].sort_values("u")
        if base is None:
            base = run
            x, y, z = _helix_xyz(run, 0 * run.dr.values, 0)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="green", width=4),
                                       name="Theoretical helix"))
        x, y, z = _helix_xyz(run, run.dr.values, exaggerate)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers",
                                   line=dict(color=palette[i % len(palette)], width=3),
                                   marker=dict(size=2, color=palette[i % len(palette)]), name=label))
    fig.update_layout(height=600, title=f"Hole {hole}: one thread, several ways to measure it "
                                        f"(deviation x{exaggerate:g})",
                      scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=50, b=0),
                      legend=dict(orientation="h"))
    return fig
