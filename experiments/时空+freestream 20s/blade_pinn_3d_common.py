from __future__ import annotations

import os
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from blade_geometry_3d import mesh_bounds, mesh_center, mesh_plane_section_polylines, points_inside_mesh, reference_length

try:
    from pyDOE import lhs as _lhs
except ImportError:
    def _lhs(n: int, samples: int) -> np.ndarray:
        cut = np.linspace(0.0, 1.0, samples + 1)
        a = cut[:samples]
        b = cut[1 : samples + 1]
        rdpoints = np.zeros((samples, n))
        for j in range(n):
            u = np.random.rand(samples)
            rdpoints[:, j] = a + (b - a) * u
            rdpoints[:, j] = rdpoints[np.random.permutation(samples), j]
        return rdpoints


def lhs(n: int, samples: int) -> np.ndarray:
    return np.asarray(_lhs(int(n), int(samples)), dtype=float)


FIELD_NAMES = ("u", "v", "w", "p")
PLANE_AXES = {
    "xy": ("x", "y"),
    "xz": ("x", "z"),
    "yz": ("y", "z"),
}
PLANE_SLICE_LABEL = {
    "xy": "z",
    "xz": "y",
    "yz": "x",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _points_in_domain(points_xyz: np.ndarray, domain: Dict[str, float]) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=float)
    return (
        (pts[:, 0] >= domain["xmin"])
        & (pts[:, 0] <= domain["xmax"])
        & (pts[:, 1] >= domain["ymin"])
        & (pts[:, 1] <= domain["ymax"])
        & (pts[:, 2] >= domain["zmin"])
        & (pts[:, 2] <= domain["zmax"])
    )



def sample_box_lhs(lb: Sequence[float], ub: Sequence[float], n_samples: int) -> np.ndarray:
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if lb.shape != ub.shape:
        raise ValueError("lb and ub must have the same shape")
    return lb + (ub - lb) * lhs(lb.size, int(n_samples))



def make_plane_points_x(x_const: float, ymin: float, ymax: float, zmin: float, zmax: float, ny: int, nz: int) -> np.ndarray:
    y = np.linspace(ymin, ymax, int(ny))
    z = np.linspace(zmin, zmax, int(nz))
    yy, zz = np.meshgrid(y, z)
    xx = np.full_like(yy, float(x_const))
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])



def make_plane_points_y(xmin: float, xmax: float, y_const: float, zmin: float, zmax: float, nx: int, nz: int) -> np.ndarray:
    x = np.linspace(xmin, xmax, int(nx))
    z = np.linspace(zmin, zmax, int(nz))
    xx, zz = np.meshgrid(x, z)
    yy = np.full_like(xx, float(y_const))
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])



def make_plane_points_z(xmin: float, xmax: float, ymin: float, ymax: float, z_const: float, nx: int, ny: int) -> np.ndarray:
    x = np.linspace(xmin, xmax, int(nx))
    y = np.linspace(ymin, ymax, int(ny))
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, float(z_const))
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])



def cart_grid_4d(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    tmin: float,
    tmax: float,
    nx: int,
    ny: int,
    nz: int,
    nt: int,
) -> np.ndarray:
    x = np.linspace(xmin, xmax, int(nx))
    y = np.linspace(ymin, ymax, int(ny))
    z = np.linspace(zmin, zmax, int(nz))
    t = np.linspace(tmin, tmax, int(nt))
    xx, yy, zz, tt = np.meshgrid(x, y, z, t, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel(), tt.ravel()])



def attach_velocity(points_xyz: np.ndarray, u0: float, v0: float, w0: float) -> np.ndarray:
    vel = np.tile(np.array([[u0, v0, w0]], dtype=float), (points_xyz.shape[0], 1))
    return np.hstack([points_xyz, vel])



def attach_velocity_steady_in_time(points_xyzt: np.ndarray, u: float, v: float, w: float) -> np.ndarray:
    vel = np.tile(np.array([[u, v, w]], dtype=float), (points_xyzt.shape[0], 1))
    return np.hstack([points_xyzt, vel])



def make_unsteady_inlet(
    inlet_xyz: np.ndarray,
    tmin: float,
    tmax: float,
    nt: int,
    u_max: float,
    phase_shift: float = -0.5 * np.pi,
) -> np.ndarray:
    t = np.linspace(tmin, tmax, int(nt))
    pts = []
    period = max(tmax - tmin, 1e-8)
    for ti in t:
        amp = 0.5 * u_max * (1.0 + np.sin(2.0 * np.pi * (ti - tmin) / period + phase_shift))
        tt = np.full((inlet_xyz.shape[0], 1), ti)
        uu = np.full((inlet_xyz.shape[0], 1), amp)
        vv = np.zeros_like(uu)
        ww = np.zeros_like(uu)
        pts.append(np.hstack([inlet_xyz, tt, uu, vv, ww]))
    return np.vstack(pts)


def make_ramp_inlet(
    inlet_xyz: np.ndarray,
    tmin: float,
    tmax: float,
    nt: int,
    u_max: float,
) -> np.ndarray:
    t = np.linspace(tmin, tmax, int(nt))
    pts = []
    period = max(tmax - tmin, 1e-8)
    for ti in t:
        ramp = np.clip((ti - tmin) / period, 0.0, 1.0)
        amp = float(u_max) * float(ramp)
        tt = np.full((inlet_xyz.shape[0], 1), ti)
        uu = np.full((inlet_xyz.shape[0], 1), amp)
        vv = np.zeros_like(uu)
        ww = np.zeros_like(uu)
        pts.append(np.hstack([inlet_xyz, tt, uu, vv, ww]))
    return np.vstack(pts)



def build_refinement_bounds_3d(
    domain: Dict[str, float],
    bmin: Sequence[float],
    bmax: Sequence[float],
    x_minus_pad: float,
    x_plus_pad: float,
    y_pad: float,
    z_pad: float,
    lref: float,
) -> Tuple[np.ndarray, np.ndarray]:
    bmin = np.asarray(bmin, dtype=float)
    bmax = np.asarray(bmax, dtype=float)
    lb = np.array(
        [
            max(domain["xmin"], bmin[0] - x_minus_pad * lref),
            max(domain["ymin"], bmin[1] - y_pad * lref),
            max(domain["zmin"], bmin[2] - z_pad * lref),
        ],
        dtype=float,
    )
    ub = np.array(
        [
            min(domain["xmax"], bmax[0] + x_plus_pad * lref),
            min(domain["ymax"], bmax[1] + y_pad * lref),
            min(domain["zmax"], bmax[2] + z_pad * lref),
        ],
        dtype=float,
    )
    return lb, ub



def build_refinement_bounds_4d(
    domain: Dict[str, float],
    bmin: Sequence[float],
    bmax: Sequence[float],
    x_minus_pad: float,
    x_plus_pad: float,
    y_pad: float,
    z_pad: float,
    lref: float,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lb3, ub3 = build_refinement_bounds_3d(domain, bmin, bmax, x_minus_pad, x_plus_pad, y_pad, z_pad, lref)
    lb = np.concatenate([lb3, np.array([tmin], dtype=float)])
    ub = np.concatenate([ub3, np.array([tmax], dtype=float)])
    return lb, ub



def plot_training_points_steady(
    collocation_xyz: np.ndarray,
    blade_xyz: np.ndarray,
    outfile: str,
    display_domain: Dict[str, float] | None = None,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    collocation_xyz = np.asarray(collocation_xyz, dtype=float)
    blade_xyz = np.asarray(blade_xyz, dtype=float)
    if display_domain is not None:
        collocation_xyz = collocation_xyz[_points_in_domain(collocation_xyz[:, :3], display_domain)]
        blade_xyz = blade_xyz[_points_in_domain(blade_xyz[:, :3], display_domain)]
    idx_c = np.random.choice(collocation_xyz.shape[0], size=min(30000, collocation_xyz.shape[0]), replace=False)
    idx_b = np.random.choice(blade_xyz.shape[0], size=min(8000, blade_xyz.shape[0]), replace=False)
    ax.scatter(collocation_xyz[idx_c, 0], collocation_xyz[idx_c, 1], collocation_xyz[idx_c, 2], s=1, alpha=0.02)
    ax.scatter(blade_xyz[idx_b, 0], blade_xyz[idx_b, 1], blade_xyz[idx_b, 2], s=2, alpha=0.10)
    ax.set_title("3D steady blade training points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if display_domain is not None:
        ax.set_xlim(display_domain["xmin"], display_domain["xmax"])
        ax.set_ylim(display_domain["ymin"], display_domain["ymax"])
        ax.set_zlim(display_domain["zmin"], display_domain["zmax"])
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)



def plot_training_points_unsteady(
    collocation_xyzt: np.ndarray,
    blade_xyzt: np.ndarray,
    outfile: str,
    display_domain: Dict[str, float] | None = None,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    collocation_xyzt = np.asarray(collocation_xyzt, dtype=float)
    blade_xyzt = np.asarray(blade_xyzt, dtype=float)
    if display_domain is not None:
        collocation_xyzt = collocation_xyzt[_points_in_domain(collocation_xyzt[:, :3], display_domain)]
        blade_xyzt = blade_xyzt[_points_in_domain(blade_xyzt[:, :3], display_domain)]
    idx_c = np.random.choice(collocation_xyzt.shape[0], size=min(30000, collocation_xyzt.shape[0]), replace=False)
    idx_b = np.random.choice(blade_xyzt.shape[0], size=min(8000, blade_xyzt.shape[0]), replace=False)
    ax.scatter(collocation_xyzt[idx_c, 0], collocation_xyzt[idx_c, 1], collocation_xyzt[idx_c, 2], s=1, alpha=0.02)
    ax.scatter(blade_xyzt[idx_b, 0], blade_xyzt[idx_b, 1], blade_xyzt[idx_b, 2], s=2, alpha=0.10)
    ax.set_title("3D unsteady blade training points (space projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if display_domain is not None:
        ax.set_xlim(display_domain["xmin"], display_domain["xmax"])
        ax.set_ylim(display_domain["ymin"], display_domain["ymax"])
        ax.set_zlim(display_domain["zmin"], display_domain["zmax"])
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)



def plot_loss_history(loss_history: Sequence[float], outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(len(loss_history)), np.asarray(loss_history, dtype=float))
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training loss history")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_loss_components_history(loss_histories: Dict[str, Sequence[float]], outfile: str) -> None:
    panels = [
        ("physics", "Physics loss"),
        ("blade", "Blade BC loss"),
        ("inlet", "Inlet BC loss"),
        ("farfield", "Farfield BC loss"),
        ("outlet", "Outlet BC loss"),
        ("ic", "Initial-condition loss"),
        ("free", "Freestream anchor loss"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, panels):
        values = np.asarray(loss_histories.get(key, []), dtype=float)
        if values.size > 0:
            ax.semilogy(np.arange(values.size), np.maximum(values, 1e-30))
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("Training loss components")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_probe_history(t_hist: np.ndarray, p_hist: np.ndarray, outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.asarray(t_hist).ravel(), np.asarray(p_hist).ravel())
    ax.set_xlabel("t")
    ax.set_ylabel("p")
    ax.set_title("Pressure history at reference probe")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_time_history(t_hist: np.ndarray, values: np.ndarray, ylabel: str, title: str, outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.asarray(t_hist).ravel(), np.asarray(values).ravel())
    ax.set_xlabel("t")
    ax.set_ylabel(str(ylabel))
    ax.set_title(str(title))
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)



def _build_plane_grid(domain: Dict[str, float], plane: str, value: float, na: int, nb: int):
    plane = plane.lower()
    if plane == "xy":
        a = np.linspace(domain["xmin"], domain["xmax"], int(na))
        b = np.linspace(domain["ymin"], domain["ymax"], int(nb))
        aa, bb = np.meshgrid(a, b)
        pts = np.column_stack([aa.ravel(), bb.ravel(), np.full(aa.size, float(value))])
    elif plane == "xz":
        a = np.linspace(domain["xmin"], domain["xmax"], int(na))
        b = np.linspace(domain["zmin"], domain["zmax"], int(nb))
        aa, bb = np.meshgrid(a, b)
        pts = np.column_stack([aa.ravel(), np.full(aa.size, float(value)), bb.ravel()])
    elif plane == "yz":
        a = np.linspace(domain["ymin"], domain["ymax"], int(na))
        b = np.linspace(domain["zmin"], domain["zmax"], int(nb))
        aa, bb = np.meshgrid(a, b)
        pts = np.column_stack([np.full(aa.size, float(value)), aa.ravel(), bb.ravel()])
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return aa, bb, pts



def _field_limits(plane_data: Dict[str, dict]) -> Dict[str, Tuple[float, float]]:
    limits = {}
    for field in FIELD_NAMES:
        values = []
        for item in plane_data.values():
            arr = np.ma.asarray(item["fields"][field])
            comp = arr.compressed()
            if comp.size:
                values.append(comp)
        if not values:
            limits[field] = (-1.0, 1.0)
            continue
        vmin = float(min(np.min(v) for v in values))
        vmax = float(max(np.max(v) for v in values))
        if np.isclose(vmin, vmax):
            pad = 1e-3 + 0.05 * max(1.0, abs(vmax))
            vmin -= pad
            vmax += pad
        limits[field] = (vmin, vmax)
    return limits



def _overlay_section(ax, section_lines: Iterable[np.ndarray]) -> None:
    for poly in section_lines:
        arr = np.asarray(poly, dtype=float)
        if arr.shape[0] < 2:
            continue
        ax.plot(arr[:, 0], arr[:, 1], "k-", linewidth=1.25)


def _predict_uniform_volume_steady(
    model,
    mesh,
    domain: Dict[str, float],
    nx: int,
    ny: int,
    nz: int,
    batch_size: int,
):
    x = np.linspace(domain["xmin"], domain["xmax"], int(nx))
    y = np.linspace(domain["ymin"], domain["ymax"], int(ny))
    z = np.linspace(domain["zmin"], domain["zmax"], int(nz))
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    inside = points_inside_mesh(pts, mesh).reshape(xx.shape)
    u, v, w, p = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], batch_size=batch_size)
    return {
        "x": x,
        "y": y,
        "z": z,
        "inside": inside,
        "u": u.reshape(xx.shape),
        "v": v.reshape(xx.shape),
        "w": w.reshape(xx.shape),
        "p": p.reshape(xx.shape),
    }


def _predict_uniform_volume_unsteady(
    model,
    mesh,
    domain: Dict[str, float],
    t_value: float,
    nx: int,
    ny: int,
    nz: int,
    batch_size: int,
):
    x = np.linspace(domain["xmin"], domain["xmax"], int(nx))
    y = np.linspace(domain["ymin"], domain["ymax"], int(ny))
    z = np.linspace(domain["zmin"], domain["zmax"], int(nz))
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    inside = points_inside_mesh(pts, mesh).reshape(xx.shape)
    tt = np.full((pts.shape[0], 1), float(t_value))
    u, v, w, p = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], tt, batch_size=batch_size)
    return {
        "x": x,
        "y": y,
        "z": z,
        "inside": inside,
        "u": u.reshape(xx.shape),
        "v": v.reshape(xx.shape),
        "w": w.reshape(xx.shape),
        "p": p.reshape(xx.shape),
    }


def _trilinear_sample_scalar(arr: np.ndarray, xg: np.ndarray, yg: np.ndarray, zg: np.ndarray, x: float, y: float, z: float) -> float:
    if x < xg[0] or x > xg[-1] or y < yg[0] or y > yg[-1] or z < zg[0] or z > zg[-1]:
        return float("nan")

    ix1 = int(np.searchsorted(xg, x, side="right"))
    iy1 = int(np.searchsorted(yg, y, side="right"))
    iz1 = int(np.searchsorted(zg, z, side="right"))
    ix1 = min(max(ix1, 1), len(xg) - 1)
    iy1 = min(max(iy1, 1), len(yg) - 1)
    iz1 = min(max(iz1, 1), len(zg) - 1)
    ix0, iy0, iz0 = ix1 - 1, iy1 - 1, iz1 - 1

    xd = 0.0 if np.isclose(xg[ix1], xg[ix0]) else (x - xg[ix0]) / (xg[ix1] - xg[ix0])
    yd = 0.0 if np.isclose(yg[iy1], yg[iy0]) else (y - yg[iy0]) / (yg[iy1] - yg[iy0])
    zd = 0.0 if np.isclose(zg[iz1], zg[iz0]) else (z - zg[iz0]) / (zg[iz1] - zg[iz0])

    c000 = arr[ix0, iy0, iz0]
    c001 = arr[ix0, iy0, iz1]
    c010 = arr[ix0, iy1, iz0]
    c011 = arr[ix0, iy1, iz1]
    c100 = arr[ix1, iy0, iz0]
    c101 = arr[ix1, iy0, iz1]
    c110 = arr[ix1, iy1, iz0]
    c111 = arr[ix1, iy1, iz1]

    c00 = c000 * (1.0 - xd) + c100 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    return float(c0 * (1.0 - zd) + c1 * zd)


def _trilinear_sample_vector(volume: dict, x: float, y: float, z: float) -> np.ndarray:
    u = _trilinear_sample_scalar(volume["u"], volume["x"], volume["y"], volume["z"], x, y, z)
    v = _trilinear_sample_scalar(volume["v"], volume["x"], volume["y"], volume["z"], x, y, z)
    w = _trilinear_sample_scalar(volume["w"], volume["x"], volume["y"], volume["z"], x, y, z)
    return np.array([u, v, w], dtype=float)


def _integrate_streamline(volume: dict, seed: np.ndarray, step_size: float, max_steps: int) -> np.ndarray | None:
    pts = [np.asarray(seed, dtype=float)]
    for _ in range(int(max_steps)):
        x, y, z = pts[-1]
        vel = _trilinear_sample_vector(volume, x, y, z)
        if not np.all(np.isfinite(vel)):
            break
        speed = np.linalg.norm(vel)
        if speed < 1e-8:
            break
        nxt = pts[-1] + step_size * (vel / speed)
        if (
            nxt[0] < volume["x"][0] or nxt[0] > volume["x"][-1]
            or nxt[1] < volume["y"][0] or nxt[1] > volume["y"][-1]
            or nxt[2] < volume["z"][0] or nxt[2] > volume["z"][-1]
        ):
            break
        pts.append(nxt)
    if len(pts) < 3:
        return None
    return np.asarray(pts, dtype=float)


def _make_seed_points_for_streamlines(mesh, domain: Dict[str, float], n_r: int = 6, n_theta: int = 18) -> np.ndarray:
    bmin, bmax = mesh_bounds(mesh)
    center = mesh_center(mesh)
    radius = 0.55 * max(bmax[1] - bmin[1], bmax[2] - bmin[2])
    x_seed = max(domain["xmin"], bmin[0] - 0.35 * reference_length(mesh, axis="x"))
    seeds = []
    for r in np.linspace(0.15 * radius, radius, int(n_r)):
        for theta in np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False):
            seeds.append([x_seed, center[1] + r * np.cos(theta), center[2] + r * np.sin(theta)])
    return np.asarray(seeds, dtype=float)


def plot_streamlines_steady(
    model,
    mesh,
    domain: Dict[str, float],
    blade_xyz: np.ndarray,
    outfile: str,
    nx: int = 28,
    ny: int = 22,
    nz: int = 22,
    batch_size: int = 65536,
) -> None:
    volume = _predict_uniform_volume_steady(model, mesh, domain, nx, ny, nz, batch_size)
    seeds = _make_seed_points_for_streamlines(mesh, domain)
    step_size = 0.06 * max(domain["xmax"] - domain["xmin"], domain["ymax"] - domain["ymin"], domain["zmax"] - domain["zmin"])
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    views = [(24, -58, "Iso view"), (18, 0, "Side view"), (90, -90, "Top view")]
    idx_b = np.random.choice(blade_xyz.shape[0], size=min(14000, blade_xyz.shape[0]), replace=False)
    blade_pts = blade_xyz[idx_b]
    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for seed in seeds:
            line = _integrate_streamline(volume, seed, step_size=step_size, max_steps=90)
            if line is not None:
                ax.plot(line[:, 0], line[:, 1], line[:, 2], color="tab:blue", linewidth=1.0, alpha=0.7)
        ax.scatter(blade_pts[:, 0], blade_pts[:, 1], blade_pts[:, 2], c="k", s=1.3, alpha=0.18, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.set_xlim(domain["xmin"], domain["xmax"])
        ax.set_ylim(domain["ymin"], domain["ymax"])
        ax.set_zlim(domain["zmin"], domain["zmax"])
    fig.suptitle("3D streamlines around blade", fontsize=14)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_streamlines_unsteady(
    model,
    mesh,
    domain: Dict[str, float],
    blade_xyz: np.ndarray,
    t_value: float,
    outfile: str,
    nx: int = 28,
    ny: int = 22,
    nz: int = 22,
    batch_size: int = 65536,
) -> None:
    volume = _predict_uniform_volume_unsteady(model, mesh, domain, t_value, nx, ny, nz, batch_size)
    seeds = _make_seed_points_for_streamlines(mesh, domain)
    step_size = 0.06 * max(domain["xmax"] - domain["xmin"], domain["ymax"] - domain["ymin"], domain["zmax"] - domain["zmin"])
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    views = [(24, -58, "Iso view"), (18, 0, "Side view"), (90, -90, "Top view")]
    idx_b = np.random.choice(blade_xyz.shape[0], size=min(14000, blade_xyz.shape[0]), replace=False)
    blade_pts = blade_xyz[idx_b]
    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for seed in seeds:
            line = _integrate_streamline(volume, seed, step_size=step_size, max_steps=90)
            if line is not None:
                ax.plot(line[:, 0], line[:, 1], line[:, 2], color="tab:blue", linewidth=1.0, alpha=0.7)
        ax.scatter(blade_pts[:, 0], blade_pts[:, 1], blade_pts[:, 2], c="k", s=1.3, alpha=0.18, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.set_xlim(domain["xmin"], domain["xmax"])
        ax.set_ylim(domain["ymin"], domain["ymax"])
        ax.set_zlim(domain["zmin"], domain["zmax"])
    fig.suptitle(f"3D streamlines at t={t_value:.4f}", fontsize=14)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def _plot_yz_plane_field(ax, yy, zz, field, title: str, section_lines, colorbar_label: str, fig, vmin: float | None = None, vmax: float | None = None) -> None:
    values = np.ma.asarray(field)
    if vmin is None or vmax is None:
        comp = values.compressed()
        if comp.size:
            vmin = float(np.min(comp))
            vmax = float(np.max(comp))
            if np.isclose(vmin, vmax):
                pad = 1e-3 + 0.05 * max(1.0, abs(vmax))
                vmin -= pad
                vmax += pad
        else:
            vmin, vmax = -1.0, 1.0
    m = ax.pcolormesh(yy, zz, values, shading="auto", cmap="turbo", vmin=vmin, vmax=vmax)
    _overlay_section(ax, section_lines)
    ax.set_aspect("equal")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(title)
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def plot_rotor_plane_steady(model, mesh, x_value: float, domain: Dict[str, float], outfile: str, ny: int = 180, nz: int = 180, batch_size: int = 65536) -> None:
    yy, zz, pts = _build_plane_grid(domain, "yz", x_value, ny, nz)
    inside = points_inside_mesh(pts, mesh).reshape(yy.shape)
    u, v, w, _ = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], batch_size=batch_size)
    speed = np.sqrt(u[:, 0] ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2).reshape(yy.shape)
    speed = np.ma.array(speed, mask=inside)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _plot_yz_plane_field(ax, yy, zz, speed, f"Rotor plane speed @ x={x_value:.4f}", mesh_plane_section_polylines(mesh, "yz", x_value), "|V|", fig)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_rotor_plane_unsteady(model, mesh, x_value: float, t_value: float, domain: Dict[str, float], outfile: str, ny: int = 180, nz: int = 180, batch_size: int = 65536) -> None:
    yy, zz, pts = _build_plane_grid(domain, "yz", x_value, ny, nz)
    inside = points_inside_mesh(pts, mesh).reshape(yy.shape)
    tt = np.full((pts.shape[0], 1), float(t_value))
    u, v, w, _ = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], tt, batch_size=batch_size)
    speed = np.sqrt(u[:, 0] ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2).reshape(yy.shape)
    speed = np.ma.array(speed, mask=inside)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _plot_yz_plane_field(ax, yy, zz, speed, f"Rotor plane speed @ t={t_value:.4f}, x={x_value:.4f}", mesh_plane_section_polylines(mesh, "yz", x_value), "|V|", fig)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_wake_planes_steady(
    model,
    mesh,
    x_positions: Sequence[float],
    domain: Dict[str, float],
    outfile: str,
    ny: int = 140,
    nz: int = 140,
    batch_size: int = 65536,
) -> None:
    fig, axes = plt.subplots(1, len(x_positions), figsize=(5 * len(x_positions), 5), constrained_layout=True)
    if len(x_positions) == 1:
        axes = [axes]
    for ax, x_value in zip(axes, x_positions):
        yy, zz, pts = _build_plane_grid(domain, "yz", x_value, ny, nz)
        inside = points_inside_mesh(pts, mesh).reshape(yy.shape)
        u, v, w, _ = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], batch_size=batch_size)
        speed = np.sqrt(u[:, 0] ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2).reshape(yy.shape)
        speed = np.ma.array(speed, mask=inside)
        _plot_yz_plane_field(ax, yy, zz, speed, f"Wake speed @ x={x_value:.4f}", mesh_plane_section_polylines(mesh, "yz", x_value), "|V|", fig)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_wake_planes_unsteady(
    model,
    mesh,
    x_positions: Sequence[float],
    t_value: float,
    domain: Dict[str, float],
    outfile: str,
    ny: int = 140,
    nz: int = 140,
    batch_size: int = 65536,
) -> None:
    fig, axes = plt.subplots(1, len(x_positions), figsize=(5 * len(x_positions), 5), constrained_layout=True)
    if len(x_positions) == 1:
        axes = [axes]
    for ax, x_value in zip(axes, x_positions):
        yy, zz, pts = _build_plane_grid(domain, "yz", x_value, ny, nz)
        inside = points_inside_mesh(pts, mesh).reshape(yy.shape)
        tt = np.full((pts.shape[0], 1), float(t_value))
        u, v, w, _ = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], tt, batch_size=batch_size)
        speed = np.sqrt(u[:, 0] ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2).reshape(yy.shape)
        speed = np.ma.array(speed, mask=inside)
        _plot_yz_plane_field(ax, yy, zz, speed, f"Wake speed @ t={t_value:.4f}, x={x_value:.4f}", mesh_plane_section_polylines(mesh, "yz", x_value), "|V|", fig)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_rotor_plane_unsteady_perturbation(
    model,
    mesh,
    x_value: float,
    t_value: float,
    freestream_u: float,
    domain: Dict[str, float],
    outfile: str,
    ny: int = 180,
    nz: int = 180,
    batch_size: int = 65536,
) -> None:
    yy, zz, pts = _build_plane_grid(domain, "yz", x_value, ny, nz)
    inside = points_inside_mesh(pts, mesh).reshape(yy.shape)
    tt = np.full((pts.shape[0], 1), float(t_value))
    u, v, w, _ = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], tt, batch_size=batch_size)
    perturb = np.sqrt((u[:, 0] - float(freestream_u)) ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2).reshape(yy.shape)
    perturb = np.ma.array(perturb, mask=inside)
    comp = perturb.compressed()
    vmax = float(np.max(comp)) if comp.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _plot_yz_plane_field(ax, yy, zz, perturb, f"Rotor plane perturbation @ t={t_value:.4f}, x={x_value:.4f}", mesh_plane_section_polylines(mesh, "yz", x_value), "|u-Uinf, v, w|", fig, vmin=0.0, vmax=vmax)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_wake_planes_unsteady_perturbation(
    model,
    mesh,
    x_positions: Sequence[float],
    t_value: float,
    freestream_u: float,
    domain: Dict[str, float],
    outfile: str,
    ny: int = 140,
    nz: int = 140,
    batch_size: int = 65536,
) -> None:
    plane_items = []
    vmax = 0.0
    for x_value in x_positions:
        yy, zz, pts = _build_plane_grid(domain, "yz", x_value, ny, nz)
        inside = points_inside_mesh(pts, mesh).reshape(yy.shape)
        tt = np.full((pts.shape[0], 1), float(t_value))
        u, v, w, _ = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], tt, batch_size=batch_size)
        perturb = np.sqrt((u[:, 0] - float(freestream_u)) ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2).reshape(yy.shape)
        perturb = np.ma.array(perturb, mask=inside)
        comp = perturb.compressed()
        if comp.size:
            vmax = max(vmax, float(np.max(comp)))
        plane_items.append((x_value, yy, zz, perturb))
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, len(x_positions), figsize=(5 * len(x_positions), 5), constrained_layout=True)
    if len(x_positions) == 1:
        axes = [axes]
    for ax, (x_value, yy, zz, perturb) in zip(axes, plane_items):
        _plot_yz_plane_field(ax, yy, zz, perturb, f"Wake perturbation @ t={t_value:.4f}, x={x_value:.4f}", mesh_plane_section_polylines(mesh, "yz", x_value), "|u-Uinf, v, w|", fig, vmin=0.0, vmax=vmax)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def _sample_volume_grid(domain: Dict[str, float], nx: int, ny: int, nz: int) -> np.ndarray:
    x = np.linspace(domain["xmin"], domain["xmax"], int(nx))
    y = np.linspace(domain["ymin"], domain["ymax"], int(ny))
    z = np.linspace(domain["zmin"], domain["zmax"], int(nz))
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _choose_scatter_subset(points: np.ndarray, values: np.ndarray, max_points: int) -> np.ndarray:
    n = points.shape[0]
    if n <= max_points:
        return np.arange(n)

    values = np.asarray(values, dtype=float).ravel()
    n_top = min(max_points // 3, n)
    n_rand = max_points - n_top

    if n_top > 0:
        idx_top = np.argsort(values)[-n_top:]
    else:
        idx_top = np.empty((0,), dtype=int)

    remain = np.setdiff1d(np.arange(n), idx_top, assume_unique=False)
    if remain.size > n_rand:
        idx_rand = np.random.choice(remain, size=n_rand, replace=False)
    else:
        idx_rand = remain

    idx = np.concatenate([idx_top, idx_rand])
    return np.unique(idx)


def _plot_whole_blade_3d_core(
    points_xyz: np.ndarray,
    field_values: np.ndarray,
    blade_xyz: np.ndarray,
    title: str,
    colorbar_label: str,
    outfile: str,
) -> None:
    idx_b = np.random.choice(blade_xyz.shape[0], size=min(12000, blade_xyz.shape[0]), replace=False)
    blade_pts = blade_xyz[idx_b]

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    views = [
        (24, -58, "Iso view"),
        (18, 0, "Side view"),
        (90, -90, "Top view"),
    ]

    vmin = float(np.min(field_values))
    vmax = float(np.max(field_values))
    if np.isclose(vmin, vmax):
        pad = 1e-3 + 0.05 * max(1.0, abs(vmax))
        vmin -= pad
        vmax += pad

    mappable = None
    for i, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        mappable = ax.scatter(
            points_xyz[:, 0],
            points_xyz[:, 1],
            points_xyz[:, 2],
            c=field_values,
            s=8,
            alpha=0.50,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.scatter(
            blade_pts[:, 0],
            blade_pts[:, 1],
            blade_pts[:, 2],
            c="k",
            s=1.5,
            alpha=0.15,
            linewidths=0,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(subtitle)
        ax.set_box_aspect(
            (
                max(np.ptp(points_xyz[:, 0]), 1e-6),
                max(np.ptp(points_xyz[:, 1]), 1e-6),
                max(np.ptp(points_xyz[:, 2]), 1e-6),
            )
        )

    fig.suptitle(title, fontsize=14)
    cbar = fig.colorbar(mappable, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label(colorbar_label)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def _predict_plane_steady(model, mesh, domain: Dict[str, float], plane: str, value: float, na: int, nb: int, batch_size: int):
    aa, bb, pts = _build_plane_grid(domain, plane, value, na, nb)
    inside = points_inside_mesh(pts, mesh).reshape(aa.shape)
    u, v, w, p = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], batch_size=batch_size)
    fields = {
        "u": np.ma.array(u.reshape(aa.shape), mask=inside),
        "v": np.ma.array(v.reshape(aa.shape), mask=inside),
        "w": np.ma.array(w.reshape(aa.shape), mask=inside),
        "p": np.ma.array(p.reshape(aa.shape), mask=inside),
    }
    return {
        "A": aa,
        "B": bb,
        "fields": fields,
        "section": mesh_plane_section_polylines(mesh, plane, value),
        "plane": plane,
        "value": float(value),
    }



def _predict_plane_unsteady(model, mesh, domain: Dict[str, float], plane: str, value: float, t_value: float, na: int, nb: int, batch_size: int):
    aa, bb, pts = _build_plane_grid(domain, plane, value, na, nb)
    inside = points_inside_mesh(pts, mesh).reshape(aa.shape)
    tt = np.full((pts.shape[0], 1), float(t_value))
    u, v, w, p = model.predict(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], tt, batch_size=batch_size)
    fields = {
        "u": np.ma.array(u.reshape(aa.shape), mask=inside),
        "v": np.ma.array(v.reshape(aa.shape), mask=inside),
        "w": np.ma.array(w.reshape(aa.shape), mask=inside),
        "p": np.ma.array(p.reshape(aa.shape), mask=inside),
    }
    return {
        "A": aa,
        "B": bb,
        "fields": fields,
        "section": mesh_plane_section_polylines(mesh, plane, value),
        "plane": plane,
        "value": float(value),
        "t": float(t_value),
    }



def plot_orthogonal_slices_steady(
    model,
    mesh,
    domain: Dict[str, float],
    x_slice: float,
    y_slice: float,
    z_slice: float,
    outfile: str,
    n_xy: Tuple[int, int] = (180, 120),
    n_xz: Tuple[int, int] = (180, 100),
    n_yz: Tuple[int, int] = (120, 100),
    batch_size: int = 65536,
) -> None:
    plane_data = {
        "xy": _predict_plane_steady(model, mesh, domain, "xy", z_slice, n_xy[0], n_xy[1], batch_size),
        "xz": _predict_plane_steady(model, mesh, domain, "xz", y_slice, n_xz[0], n_xz[1], batch_size),
        "yz": _predict_plane_steady(model, mesh, domain, "yz", x_slice, n_yz[0], n_yz[1], batch_size),
    }
    limits = _field_limits(plane_data)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        f"Steady orthogonal slices: x={x_slice:.4f}, y={y_slice:.4f}, z={z_slice:.4f}",
        fontsize=14,
    )

    for row, plane in enumerate(("xy", "xz", "yz")):
        data = plane_data[plane]
        xlabel, ylabel = PLANE_AXES[plane]
        slice_label = PLANE_SLICE_LABEL[plane]
        for col, field in enumerate(FIELD_NAMES):
            ax = axes[row, col]
            vals = data["fields"][field]
            vmin, vmax = limits[field]
            m = ax.pcolormesh(data["A"], data["B"], vals, shading="auto", cmap="rainbow", vmin=vmin, vmax=vmax)
            _overlay_section(ax, data["section"])
            ax.set_aspect("equal")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{field} on {plane.upper()} @ {slice_label}={data['value']:.4f}")
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(outfile, dpi=200)
    plt.close(fig)



def plot_orthogonal_slices_unsteady(
    model,
    mesh,
    domain: Dict[str, float],
    x_slice: float,
    y_slice: float,
    z_slice: float,
    t_value: float,
    outfile: str,
    n_xy: Tuple[int, int] = (180, 120),
    n_xz: Tuple[int, int] = (180, 100),
    n_yz: Tuple[int, int] = (120, 100),
    batch_size: int = 65536,
) -> None:
    plane_data = {
        "xy": _predict_plane_unsteady(model, mesh, domain, "xy", z_slice, t_value, n_xy[0], n_xy[1], batch_size),
        "xz": _predict_plane_unsteady(model, mesh, domain, "xz", y_slice, t_value, n_xz[0], n_xz[1], batch_size),
        "yz": _predict_plane_unsteady(model, mesh, domain, "yz", x_slice, t_value, n_yz[0], n_yz[1], batch_size),
    }
    limits = _field_limits(plane_data)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        f"Unsteady orthogonal slices at t={t_value:.4f}: x={x_slice:.4f}, y={y_slice:.4f}, z={z_slice:.4f}",
        fontsize=14,
    )

    for row, plane in enumerate(("xy", "xz", "yz")):
        data = plane_data[plane]
        xlabel, ylabel = PLANE_AXES[plane]
        slice_label = PLANE_SLICE_LABEL[plane]
        for col, field in enumerate(FIELD_NAMES):
            ax = axes[row, col]
            vals = data["fields"][field]
            vmin, vmax = limits[field]
            m = ax.pcolormesh(data["A"], data["B"], vals, shading="auto", cmap="rainbow", vmin=vmin, vmax=vmax)
            _overlay_section(ax, data["section"])
            ax.set_aspect("equal")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{field} on {plane.upper()} @ {slice_label}={data['value']:.4f}")
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_orthogonal_slices_unsteady_perturbation(
    model,
    mesh,
    domain: Dict[str, float],
    x_slice: float,
    y_slice: float,
    z_slice: float,
    t_value: float,
    freestream_u: float,
    outfile: str,
    n_xy: Tuple[int, int] = (180, 120),
    n_xz: Tuple[int, int] = (180, 100),
    n_yz: Tuple[int, int] = (120, 100),
    batch_size: int = 65536,
) -> None:
    plane_data = {
        "xy": _predict_plane_unsteady(model, mesh, domain, "xy", z_slice, t_value, n_xy[0], n_xy[1], batch_size),
        "xz": _predict_plane_unsteady(model, mesh, domain, "xz", y_slice, t_value, n_xz[0], n_xz[1], batch_size),
        "yz": _predict_plane_unsteady(model, mesh, domain, "yz", x_slice, t_value, n_yz[0], n_yz[1], batch_size),
    }

    vmax = 0.0
    perturb_data = {}
    for plane, data in plane_data.items():
        u = data["fields"]["u"]
        v = data["fields"]["v"]
        w = data["fields"]["w"]
        perturb = np.sqrt((u - float(freestream_u)) ** 2 + v ** 2 + w ** 2)
        perturb = np.ma.array(perturb, mask=np.ma.getmaskarray(u))
        perturb_data[plane] = perturb
        comp = perturb.compressed()
        if comp.size:
            vmax = max(vmax, float(np.max(comp)))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(
        f"Unsteady perturbation slices at t={t_value:.4f}: x={x_slice:.4f}, y={y_slice:.4f}, z={z_slice:.4f}, Uinf={freestream_u:.4f}",
        fontsize=14,
    )

    for ax, plane in zip(axes, ("xy", "xz", "yz")):
        data = plane_data[plane]
        xlabel, ylabel = PLANE_AXES[plane]
        slice_label = PLANE_SLICE_LABEL[plane]
        perturb = perturb_data[plane]
        m = ax.pcolormesh(data["A"], data["B"], perturb, shading="auto", cmap="rainbow", vmin=0.0, vmax=vmax)
        _overlay_section(ax, data["section"])
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"|u-Uinf, v, w| on {plane.upper()} @ {slice_label}={data['value']:.4f}")
        fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(outfile, dpi=200)
    plt.close(fig)


