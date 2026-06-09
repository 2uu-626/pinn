from __future__ import annotations


import pickle
import random
import time
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh

from blade_geometry_3d import (
    auto_domain_from_mesh,
    blade_reference_point,
    load_blade_mesh,
    mesh_bounds,
    mesh_center,
    orthogonal_slice_positions,
    reference_length,
    sample_surface_points,
)
from blade_pinn_3d_common import (
    attach_velocity_steady_in_time,
    build_refinement_bounds_4d,
    cart_grid_4d,
    ensure_dir,
    make_plane_points_x,
    make_plane_points_y,
    make_plane_points_z,
    make_ramp_inlet,
    make_unsteady_inlet,
    plot_loss_components_history,
    plot_loss_history,
    plot_orthogonal_slices_unsteady,
    plot_probe_history,
    plot_time_history,
    plot_rotor_plane_unsteady,
    plot_streamlines_unsteady,
    plot_training_points_unsteady,
    plot_wake_planes_unsteady,
    sample_box_lhs,
)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)


def configure_torch_cpu(max_threads: int | None) -> None:
    if max_threads is None:
        return
    max_threads = max(1, int(max_threads))
    torch.set_num_threads(max_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max_threads)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Rotation axis must be non-zero.")
    return v / n


def _orthonormal_frame_from_axis(axis_xyz: np.ndarray) -> np.ndarray:
    axis = _normalize(axis_xyz)
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    span_1 = _normalize(np.cross(axis, helper))
    span_2 = _normalize(np.cross(axis, span_1))
    return np.vstack([axis, span_1, span_2])


def estimate_rotor_radius(points_xyz: np.ndarray, center_xyz: np.ndarray, axis_xyz: np.ndarray) -> float:
    pts = np.asarray(points_xyz, dtype=float)
    center = np.asarray(center_xyz, dtype=float).reshape(1, 3)
    axis = _normalize(axis_xyz).reshape(1, 3)
    rel = pts - center
    axial = np.sum(rel * axis, axis=1, keepdims=True) * axis
    radial = rel - axial
    return float(np.max(np.linalg.norm(radial, axis=1)))


def sample_rotor_cylinder_lhs(
    center_xyz: np.ndarray,
    axis_xyz: np.ndarray,
    radial_radius: float,
    axial_halfwidth: float,
    tmin: float,
    tmax: float,
    n_samples: int,
) -> np.ndarray:
    n_samples = int(n_samples)
    if n_samples <= 0:
        return np.empty((0, 4), dtype=float)
    basis = _orthonormal_frame_from_axis(axis_xyz)
    unit = sample_box_lhs(np.zeros(4, dtype=float), np.ones(4, dtype=float), n_samples)
    axial = (2.0 * unit[:, 0] - 1.0) * float(axial_halfwidth)
    radius = float(radial_radius) * np.sqrt(unit[:, 1])
    theta = 2.0 * np.pi * unit[:, 2]
    tt = float(tmin) + (float(tmax) - float(tmin)) * unit[:, 3]
    xyz = (
        np.asarray(center_xyz, dtype=float).reshape(1, 3)
        + axial[:, None] * basis[0:1, :]
        + (radius * np.cos(theta))[:, None] * basis[1:2, :]
        + (radius * np.sin(theta))[:, None] * basis[2:3, :]
    )
    return np.hstack([xyz, tt.reshape(-1, 1)])



def save_slice_gif(image_paths: list[str], outfile: str, duration_s: float = 0.35) -> None:
    if len(image_paths) == 0:
        return
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"Skipping GIF export: {exc}")
        return
    frames = [imageio.imread(image_path) for image_path in image_paths]
    imageio.mimsave(outfile, frames, duration=float(duration_s), loop=0)


def inflow_speed_at_time(
    t: np.ndarray | float,
    tmin: float,
    tmax: float,
    u_max: float,
    mode: str = "ramp",
    ramp_fraction: float = 1.0,
) -> np.ndarray:
    """Return the x-inflow speed used by inlet, farfield, and IC targets.

    Using the same profile everywhere avoids the original conflict where the
    initial condition was zero while farfield was already U_MAX.
    """
    tt = np.asarray(t, dtype=float)
    mode_l = str(mode).lower()
    if mode_l in {"steady", "constant", "uniform"}:
        return np.full_like(tt, float(u_max), dtype=float)
    if mode_l not in {"ramp", "smooth_ramp", "startup"}:
        raise ValueError("INFLOW_MODE must be 'ramp' or 'steady'.")

    total_dt = max(float(tmax) - float(tmin), 1e-12)
    ramp_dt = max(float(ramp_fraction), 1e-6) * total_dt
    tau = np.clip((tt - float(tmin)) / ramp_dt, 0.0, 1.0)
    # Smoothstep ramp: zero slope at tmin and after ramp end.
    smooth = tau * tau * (3.0 - 2.0 * tau)
    return float(u_max) * smooth


def attach_inflow_profile(
    points_xyzt: np.ndarray,
    tmin: float,
    tmax: float,
    u_max: float,
    mode: str = "ramp",
    ramp_fraction: float = 1.0,
) -> np.ndarray:
    """Append target velocity columns [u, v, w] to [x, y, z, t] points."""
    pts = np.asarray(points_xyzt, dtype=float)
    u = inflow_speed_at_time(pts[:, 3:4], tmin, tmax, u_max, mode=mode, ramp_fraction=ramp_fraction)
    zeros = np.zeros_like(u)
    return np.hstack([pts, u, zeros, zeros])


def attach_initial_condition_profile(
    points_xyzt: np.ndarray,
    tmin: float,
    tmax: float,
    u_max: float,
    mode: str = "ramp",
    ramp_fraction: float = 1.0,
    pressure_value: float = 0.0,
) -> np.ndarray:
    """Append target columns [u, v, w, p] to IC points."""
    pts_with_vel = attach_inflow_profile(points_xyzt, tmin, tmax, u_max, mode=mode, ramp_fraction=ramp_fraction)
    p = np.full((pts_with_vel.shape[0], 1), float(pressure_value), dtype=float)
    return np.hstack([pts_with_vel, p])


def slice_speed_summary_unsteady(model, domain: dict[str, float], x_slice: float, y_slice: float, z_slice: float, t_value: float, n1: int = 41, n2: int = 41) -> dict[str, tuple[float, float, float]]:
    def _speed_stats(points_xyz: np.ndarray, t_scalar: float) -> tuple[float, float, float]:
        tt = np.full((points_xyz.shape[0], 1), float(t_scalar), dtype=float)
        u, v, w, _ = model.predict(points_xyz[:, 0:1], points_xyz[:, 1:2], points_xyz[:, 2:3], tt)
        speed = np.sqrt(u[:, 0] ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2)
        return float(np.min(speed)), float(np.max(speed)), float(np.mean(speed))

    yy = np.linspace(domain["ymin"], domain["ymax"], int(n1))
    zz = np.linspace(domain["zmin"], domain["zmax"], int(n2))
    Yx, Zx = np.meshgrid(yy, zz, indexing="ij")
    Xx = np.full_like(Yx, float(x_slice))
    pts_x = np.column_stack([Xx.reshape(-1), Yx.reshape(-1), Zx.reshape(-1)])

    xx = np.linspace(domain["xmin"], domain["xmax"], int(n1))
    Zy = np.linspace(domain["zmin"], domain["zmax"], int(n2))
    Xy, Zy = np.meshgrid(xx, Zy, indexing="ij")
    Yy = np.full_like(Xy, float(y_slice))
    pts_y = np.column_stack([Xy.reshape(-1), Yy.reshape(-1), Zy.reshape(-1)])

    Xz, Yz = np.meshgrid(xx, yy, indexing="ij")
    Zz = np.full_like(Xz, float(z_slice))
    pts_z = np.column_stack([Xz.reshape(-1), Yz.reshape(-1), Zz.reshape(-1)])

    return {
        "x": _speed_stats(pts_x, t_value),
        "y": _speed_stats(pts_y, t_value),
        "z": _speed_stats(pts_z, t_value),
    }


def slice_perturbation_summary_unsteady(
    model,
    domain: dict[str, float],
    x_slice: float,
    y_slice: float,
    z_slice: float,
    t_value: float,
    freestream_u: float,
    n1: int = 41,
    n2: int = 41,
) -> dict[str, tuple[float, float, float]]:
    """Report min/max/mean of |[u-U_inf, v, w]| on each slice.

    This is often easier to read than raw speed when the solution contains a
    dominant uniform inflow plus a smaller wake/rotational disturbance.
    """
    def _pert_stats(points_xyz: np.ndarray, t_scalar: float) -> tuple[float, float, float]:
        tt = np.full((points_xyz.shape[0], 1), float(t_scalar), dtype=float)
        u, v, w, _ = model.predict(points_xyz[:, 0:1], points_xyz[:, 1:2], points_xyz[:, 2:3], tt)
        perturb = np.sqrt((u[:, 0] - float(freestream_u)) ** 2 + v[:, 0] ** 2 + w[:, 0] ** 2)
        return float(np.min(perturb)), float(np.max(perturb)), float(np.mean(perturb))

    yy = np.linspace(domain["ymin"], domain["ymax"], int(n1))
    zz = np.linspace(domain["zmin"], domain["zmax"], int(n2))
    Yx, Zx = np.meshgrid(yy, zz, indexing="ij")
    Xx = np.full_like(Yx, float(x_slice))
    pts_x = np.column_stack([Xx.reshape(-1), Yx.reshape(-1), Zx.reshape(-1)])

    xx = np.linspace(domain["xmin"], domain["xmax"], int(n1))
    Zy = np.linspace(domain["zmin"], domain["zmax"], int(n2))
    Xy, Zy = np.meshgrid(xx, Zy, indexing="ij")
    Yy = np.full_like(Xy, float(y_slice))
    pts_y = np.column_stack([Xy.reshape(-1), Yy.reshape(-1), Zy.reshape(-1)])

    Xz, Yz = np.meshgrid(xx, yy, indexing="ij")
    Zz = np.full_like(Xz, float(z_slice))
    pts_z = np.column_stack([Xz.reshape(-1), Yz.reshape(-1), Zz.reshape(-1)])

    return {
        "x": _pert_stats(pts_x, t_value),
        "y": _pert_stats(pts_y, t_value),
        "z": _pert_stats(pts_z, t_value),
    }



def rotate_points_about_axis(points_xyz: np.ndarray, center_xyz: np.ndarray, axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=float)
    center = np.asarray(center_xyz, dtype=float).reshape(1, 3)
    axis = _normalize(axis_xyz).reshape(1, 3)
    rel = pts - center
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    cross = np.cross(np.repeat(axis, rel.shape[0], axis=0), rel)
    dot = np.sum(rel * axis, axis=1, keepdims=True)
    rotated = rel * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)
    return rotated + center


def rotate_points_about_axis_time(points_xyz: np.ndarray, center_xyz: np.ndarray, axis_xyz: np.ndarray, angle_rad: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=float)
    center = np.asarray(center_xyz, dtype=float).reshape(1, 3)
    axis = _normalize(axis_xyz).reshape(1, 3)
    rel = pts - center
    ang = np.asarray(angle_rad, dtype=float).reshape(-1, 1)
    if rel.shape[0] != ang.shape[0]:
        raise ValueError("angle_rad must have the same length as points_xyz")
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    axis_repeat = np.repeat(axis, rel.shape[0], axis=0)
    cross = np.cross(axis_repeat, rel)
    dot = np.sum(rel * axis_repeat, axis=1, keepdims=True)
    rotated = rel * cos_a + cross * sin_a + axis_repeat * dot * (1.0 - cos_a)
    return rotated + center


def rotate_mesh_about_axis(mesh, center_xyz: np.ndarray, axis_xyz: np.ndarray, angle_rad: float):
    rotated = mesh.copy()
    rotated.vertices = rotate_points_about_axis(np.asarray(rotated.vertices), center_xyz, axis_xyz, angle_rad)
    return rotated


def rotating_wall_velocity(points_xyz: np.ndarray, center_xyz: np.ndarray, axis_xyz: np.ndarray, omega_rad_s: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=float)
    center = np.asarray(center_xyz, dtype=float).reshape(1, 3)
    omega_vec = omega_rad_s * _normalize(axis_xyz).reshape(1, 3)
    rel = pts - center
    return np.cross(np.repeat(omega_vec, rel.shape[0], axis=0), rel)


def build_rotating_blade_dataset(
    surface_xyz: np.ndarray,
    tmin: float,
    tmax: float,
    num_t: int,
    center_xyz: np.ndarray,
    axis_xyz: np.ndarray,
    omega_rad_s: float,
    t_ref: float = 0.0,
) -> np.ndarray:
    surface_xyz = np.asarray(surface_xyz, dtype=float)
    t_values = np.linspace(float(tmin), float(tmax), int(num_t))
    blade_blocks = []
    for ti in t_values:
        angle = omega_rad_s * (ti - t_ref)
        xyz_t = rotate_points_about_axis(surface_xyz, center_xyz, axis_xyz, angle)
        vel_t = rotating_wall_velocity(xyz_t, center_xyz, axis_xyz, omega_rad_s)
        tt = np.full((xyz_t.shape[0], 1), ti, dtype=float)
        blade_blocks.append(np.hstack([xyz_t, tt, vel_t]))
    return np.vstack(blade_blocks)


def build_static_wall_dataset(surface_xyz: np.ndarray, tmin: float, tmax: float, num_t: int) -> np.ndarray:
    surface_xyz = np.asarray(surface_xyz, dtype=float)
    t_values = np.linspace(float(tmin), float(tmax), int(num_t))
    wall_blocks = []
    zeros = np.zeros((surface_xyz.shape[0], 3), dtype=float)
    for ti in t_values:
        tt = np.full((surface_xyz.shape[0], 1), ti, dtype=float)
        wall_blocks.append(np.hstack([surface_xyz, tt, zeros]))
    return np.vstack(wall_blocks)


def remove_points_inside_rotating_mesh(points_xyzt: np.ndarray, mesh, center_xyz: np.ndarray, axis_xyz: np.ndarray, omega_rad_s: float, t_ref: float = 0.0) -> np.ndarray:
    pts = np.asarray(points_xyzt, dtype=float)
    angles = -omega_rad_s * (pts[:, 3] - float(t_ref))
    xyz_back = rotate_points_about_axis_time(pts[:, :3], center_xyz, axis_xyz, angles)
    inside = np.asarray(mesh.contains(xyz_back), dtype=bool)
    return pts[~inside, :]


def remove_points_inside_static_mesh(points_xyzt: np.ndarray, mesh) -> np.ndarray:
    pts = np.asarray(points_xyzt, dtype=float)
    inside = np.asarray(mesh.contains(pts[:, :3]), dtype=bool)
    return pts[~inside, :]


def remove_points_inside_obstacles(
    points_xyzt: np.ndarray,
    rotating_mesh,
    center_xyz: np.ndarray,
    axis_xyz: np.ndarray,
    omega_rad_s: float,
    t_ref: float = 0.0,
    static_mesh=None,
) -> np.ndarray:
    pts = remove_points_inside_rotating_mesh(points_xyzt, rotating_mesh, center_xyz, axis_xyz, omega_rad_s, t_ref=t_ref)
    if static_mesh is not None:
        pts = remove_points_inside_static_mesh(pts, static_mesh)
    return pts


def combine_meshes(meshes: list) -> object | None:
    valid_meshes = [mesh.copy() for mesh in meshes if mesh is not None]
    if len(valid_meshes) == 0:
        return None
    if len(valid_meshes) == 1:
        return valid_meshes[0]
    return trimesh.util.concatenate(tuple(valid_meshes))


class PINNBladeRotatingUnsteady3D:
    def __init__(
        self,
        Collo,
        IC,
        INLET,
        FARFIELD,
        OUTLET,
        BLADE,
        uv_layers,
        lb,
        ub,
        rho=1.0,
        mu=0.005,
        collo_batch_size=4096,
        loss_weight_blade=50.0,
        loss_weight_inlet=2.0,
        loss_weight_far=2.0,
        loss_weight_outlet=1.0,
        loss_weight_ic=1.0,
        decomposition_config=None,
        device=None,
        ExistModel=0,
        uvDir="",
    ):
        self.count = 0
        self.loss_rec = []
        self.loss_components_rec = {
            "physics": [],
            "blade": [],
            "inlet": [],
            "farfield": [],
            "outlet": [],
            "ic": [],
        }
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decomposition_config = dict(decomposition_config) if decomposition_config is not None else None
        self.use_rotor_aware_decomposition = self.decomposition_config is not None

        self.lb = torch.tensor(lb, dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=self.device)

        self.rho = float(rho)
        self.mu = float(mu)
        self.collo_batch_size = max(1, int(collo_batch_size))
        self.loss_weight_blade = float(loss_weight_blade)
        self.loss_weight_inlet = float(loss_weight_inlet)
        self.loss_weight_far = float(loss_weight_far)
        self.loss_weight_outlet = float(loss_weight_outlet)
        self.loss_weight_ic = float(loss_weight_ic)

        self.x_c = torch.tensor(Collo[:, 0:1], dtype=torch.float32, device=self.device, requires_grad=True)
        self.y_c = torch.tensor(Collo[:, 1:2], dtype=torch.float32, device=self.device, requires_grad=True)
        self.z_c = torch.tensor(Collo[:, 2:3], dtype=torch.float32, device=self.device, requires_grad=True)
        self.t_c = torch.tensor(Collo[:, 3:4], dtype=torch.float32, device=self.device, requires_grad=True)

        self.x_IC = torch.tensor(IC[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_IC = torch.tensor(IC[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_IC = torch.tensor(IC[:, 2:3], dtype=torch.float32, device=self.device)
        self.t_IC = torch.tensor(IC[:, 3:4], dtype=torch.float32, device=self.device)
        if IC.shape[1] >= 7:
            self.u_IC = torch.tensor(IC[:, 4:5], dtype=torch.float32, device=self.device)
            self.v_IC = torch.tensor(IC[:, 5:6], dtype=torch.float32, device=self.device)
            self.w_IC = torch.tensor(IC[:, 6:7], dtype=torch.float32, device=self.device)
        else:
            self.u_IC = torch.zeros_like(self.x_IC)
            self.v_IC = torch.zeros_like(self.x_IC)
            self.w_IC = torch.zeros_like(self.x_IC)
        if IC.shape[1] >= 8:
            self.p_IC = torch.tensor(IC[:, 7:8], dtype=torch.float32, device=self.device)
        else:
            self.p_IC = torch.zeros_like(self.x_IC)

        self.x_INLET = torch.tensor(INLET[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_INLET = torch.tensor(INLET[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_INLET = torch.tensor(INLET[:, 2:3], dtype=torch.float32, device=self.device)
        self.t_INLET = torch.tensor(INLET[:, 3:4], dtype=torch.float32, device=self.device)
        self.u_INLET = torch.tensor(INLET[:, 4:5], dtype=torch.float32, device=self.device)
        self.v_INLET = torch.tensor(INLET[:, 5:6], dtype=torch.float32, device=self.device)
        self.w_INLET = torch.tensor(INLET[:, 6:7], dtype=torch.float32, device=self.device)

        self.x_FAR = torch.tensor(FARFIELD[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_FAR = torch.tensor(FARFIELD[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_FAR = torch.tensor(FARFIELD[:, 2:3], dtype=torch.float32, device=self.device)
        self.t_FAR = torch.tensor(FARFIELD[:, 3:4], dtype=torch.float32, device=self.device)
        self.u_FAR = torch.tensor(FARFIELD[:, 4:5], dtype=torch.float32, device=self.device)
        self.v_FAR = torch.tensor(FARFIELD[:, 5:6], dtype=torch.float32, device=self.device)
        self.w_FAR = torch.tensor(FARFIELD[:, 6:7], dtype=torch.float32, device=self.device)

        self.x_OUT = torch.tensor(OUTLET[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_OUT = torch.tensor(OUTLET[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_OUT = torch.tensor(OUTLET[:, 2:3], dtype=torch.float32, device=self.device)
        self.t_OUT = torch.tensor(OUTLET[:, 3:4], dtype=torch.float32, device=self.device)

        self.x_BLADE = torch.tensor(BLADE[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_BLADE = torch.tensor(BLADE[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_BLADE = torch.tensor(BLADE[:, 2:3], dtype=torch.float32, device=self.device)
        self.t_BLADE = torch.tensor(BLADE[:, 3:4], dtype=torch.float32, device=self.device)
        self.u_BLADE = torch.tensor(BLADE[:, 4:5], dtype=torch.float32, device=self.device)
        self.v_BLADE = torch.tensor(BLADE[:, 5:6], dtype=torch.float32, device=self.device)
        self.w_BLADE = torch.tensor(BLADE[:, 6:7], dtype=torch.float32, device=self.device)

        self.uv_layers = uv_layers
        if self.use_rotor_aware_decomposition:
            self._init_rotor_aware_decomposition(self.decomposition_config)
        else:
            self.net = self.initialize_NN(self.uv_layers).to(self.device)

        if ExistModel == 1:
            print("Loading rotating 3D unsteady NN ...")
            self.load_NN(uvDir)

        self.optimizer_Adam = optim.Adam(self.trainable_parameters(), lr=5e-4)

    def initialize_NN(self, layers):
        modules = []
        for l in range(len(layers) - 2):
            modules.append(nn.Linear(layers[l], layers[l + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        net = nn.Sequential(*modules)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        return net

    def _init_rotor_aware_decomposition(self, config: dict) -> None:
        center_xyz = np.asarray(config["center_xyz"], dtype=float).reshape(3)
        axis_xyz = _normalize(np.asarray(config["axis_xyz"], dtype=float).reshape(3))
        basis = _orthonormal_frame_from_axis(axis_xyz)
        near_radius = max(float(config["near_radius"]), 1e-6)
        axial_halfwidth = max(float(config["axial_halfwidth"]), 1e-6)
        time_window = config.get("time_window", (float(self.lb[3].item()), float(self.ub[3].item())))
        near_tmin = float(time_window[0])
        near_tmax = float(time_window[1])
        if near_tmax <= near_tmin:
            raise ValueError("Rotor-aware decomposition time_window must satisfy tmax > tmin.")

        self.rotor_center_tensor = torch.tensor(center_xyz.reshape(1, 3), dtype=torch.float32, device=self.device)
        self.rotor_basis_tensor = torch.tensor(basis, dtype=torch.float32, device=self.device)
        self.rotor_near_radius = near_radius
        self.rotor_axial_halfwidth = axial_halfwidth
        self.rotor_blend_sharpness = max(float(config.get("blend_sharpness", 8.0)), 1e-3)
        self.rotor_time_window = (near_tmin, near_tmax)
        global_tmin = float(self.lb[3].item())
        global_tmax = float(self.ub[3].item())
        self.rotor_use_time_gate = (abs(near_tmin - global_tmin) > 1e-9) or (abs(near_tmax - global_tmax) > 1e-9)

        self.rotor_local_lb = torch.tensor(
            [-axial_halfwidth, -near_radius, -near_radius, near_tmin],
            dtype=torch.float32,
            device=self.device,
        )
        self.rotor_local_ub = torch.tensor(
            [axial_halfwidth, near_radius, near_radius, near_tmax],
            dtype=torch.float32,
            device=self.device,
        )
        self.net_rotor = self.initialize_NN(self.uv_layers).to(self.device)
        self.net_far = self.initialize_NN(self.uv_layers).to(self.device)

    def model_modules(self):
        if self.use_rotor_aware_decomposition:
            return [self.net_rotor, self.net_far]
        return [self.net]

    def trainable_parameters(self):
        return [param for module in self.model_modules() for param in module.parameters()]

    def save_NN(self, fileDir):
        if self.use_rotor_aware_decomposition:
            payload = {
                "use_rotor_aware_decomposition": True,
                "net_rotor": self.net_rotor.state_dict(),
                "net_far": self.net_far.state_dict(),
            }
        else:
            payload = {
                "use_rotor_aware_decomposition": False,
                "net": self.net.state_dict(),
            }
        torch.save(payload, fileDir)
        print(f"Saved NN parameters to: {fileDir}")

    def load_NN(self, fileDir):
        payload = torch.load(fileDir, map_location=self.device)
        if isinstance(payload, dict) and "use_rotor_aware_decomposition" in payload:
            saved_is_decomp = bool(payload["use_rotor_aware_decomposition"])
            if saved_is_decomp != self.use_rotor_aware_decomposition:
                raise ValueError("Checkpoint decomposition mode does not match the current model configuration.")
            if saved_is_decomp:
                self.net_rotor.load_state_dict(payload["net_rotor"])
                self.net_far.load_state_dict(payload["net_far"])
            else:
                self.net.load_state_dict(payload["net"])
        else:
            if self.use_rotor_aware_decomposition:
                raise ValueError("Monolithic checkpoint cannot be loaded into rotor-aware decomposition mode.")
            self.net.load_state_dict(payload)
        print("Loaded NN parameters successfully...")

    @staticmethod
    def _normalize_box(X, lb, ub):
        return 2.0 * (X - lb) / (ub - lb) - 1.0

    def neural_net(self, X):
        H = self._normalize_box(X, self.lb, self.ub)
        return self.net(H)

    def _rotor_local_coordinates(self, x, y, z):
        xyz = torch.cat([x, y, z], dim=1)
        rel = xyz - self.rotor_center_tensor
        axial = torch.sum(rel * self.rotor_basis_tensor[0:1, :], dim=1, keepdim=True)
        span_1 = torch.sum(rel * self.rotor_basis_tensor[1:2, :], dim=1, keepdim=True)
        span_2 = torch.sum(rel * self.rotor_basis_tensor[2:3, :], dim=1, keepdim=True)
        return axial, span_1, span_2

    def _rotor_blend_weight(self, axial, span_1, span_2, t):
        radial = torch.sqrt(span_1 * span_1 + span_2 * span_2 + 1e-12)
        radial_margin = 1.0 - radial / self.rotor_near_radius
        axial_margin = 1.0 - torch.abs(axial) / self.rotor_axial_halfwidth
        weight = torch.sigmoid(self.rotor_blend_sharpness * radial_margin) * torch.sigmoid(
            self.rotor_blend_sharpness * axial_margin
        )
        if self.rotor_use_time_gate:
            near_tmin, near_tmax = self.rotor_time_window
            time_center = 0.5 * (near_tmin + near_tmax)
            time_halfwidth = max(0.5 * (near_tmax - near_tmin), 1e-6)
            time_margin = 1.0 - torch.abs(t - time_center) / time_halfwidth
            weight = weight * torch.sigmoid(self.rotor_blend_sharpness * time_margin)
        return torch.clamp(weight, 0.0, 1.0)

    def decomposed_neural_net(self, x, y, z, t):
        X_global = torch.cat([x, y, z, t], dim=1)
        axial, span_1, span_2 = self._rotor_local_coordinates(x, y, z)
        X_rotor = torch.cat([axial, span_1, span_2, t], dim=1)

        H_rotor = self._normalize_box(X_rotor, self.rotor_local_lb, self.rotor_local_ub)
        H_far = self._normalize_box(X_global, self.lb, self.ub)

        out_rotor = self.net_rotor(H_rotor)
        out_far = self.net_far(H_far)
        blend = self._rotor_blend_weight(axial, span_1, span_2, t)
        return blend * out_rotor + (1.0 - blend) * out_far

    def net_uvw(self, x, y, z, t):
        if self.use_rotor_aware_decomposition:
            out = self.decomposed_neural_net(x, y, z, t)
        else:
            X = torch.cat([x, y, z, t], dim=1)
            out = self.neural_net(X)
        return (
            out[:, 0:1],
            out[:, 1:2],
            out[:, 2:3],
            out[:, 3:4],
            out[:, 4:5],
            out[:, 5:6],
            out[:, 6:7],
            out[:, 7:8],
            out[:, 8:9],
            out[:, 9:10],
        )

    def net_f(self, x, y, z, t):
        rho = self.rho
        mu = self.mu
        u, v, w, p, sxx, syy, szz, sxy, sxz, syz = self.net_uvw(x, y, z, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]

        sxx_x = torch.autograd.grad(sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True)[0]
        syy_y = torch.autograd.grad(syy, y, grad_outputs=torch.ones_like(syy), create_graph=True)[0]
        szz_z = torch.autograd.grad(szz, z, grad_outputs=torch.ones_like(szz), create_graph=True)[0]
        sxy_x = torch.autograd.grad(sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True)[0]
        sxy_y = torch.autograd.grad(sxy, y, grad_outputs=torch.ones_like(sxy), create_graph=True)[0]
        sxz_x = torch.autograd.grad(sxz, x, grad_outputs=torch.ones_like(sxz), create_graph=True)[0]
        sxz_z = torch.autograd.grad(sxz, z, grad_outputs=torch.ones_like(sxz), create_graph=True)[0]
        syz_y = torch.autograd.grad(syz, y, grad_outputs=torch.ones_like(syz), create_graph=True)[0]
        syz_z = torch.autograd.grad(syz, z, grad_outputs=torch.ones_like(syz), create_graph=True)[0]

        r_c = u_x + v_y + w_z
        r_mx = rho * u_t + rho * (u * u_x + v * u_y + w * u_z) - (sxx_x + sxy_y + sxz_z)
        r_my = rho * v_t + rho * (u * v_x + v * v_y + w * v_z) - (sxy_x + syy_y + syz_z)
        r_mz = rho * w_t + rho * (u * w_x + v * w_y + w * w_z) - (sxz_x + syz_y + szz_z)

        r_sxx = -p + 2.0 * mu * u_x - sxx
        r_syy = -p + 2.0 * mu * v_y - syy
        r_szz = -p + 2.0 * mu * w_z - szz
        r_sxy = mu * (u_y + v_x) - sxy
        r_sxz = mu * (u_z + w_x) - sxz
        r_syz = mu * (v_z + w_y) - syz
        r_p = p + (sxx + syy + szz) / 3.0
        return r_c, r_mx, r_my, r_mz, r_sxx, r_syy, r_szz, r_sxy, r_sxz, r_syz, r_p

    def physics_loss_batched(self):
        n_total = self.x_c.shape[0]
        loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        for i in range(0, n_total, self.collo_batch_size):
            sl = slice(i, min(i + self.collo_batch_size, n_total))
            x = self.x_c[sl].clone().detach().requires_grad_(True)
            y = self.y_c[sl].clone().detach().requires_grad_(True)
            z = self.z_c[sl].clone().detach().requires_grad_(True)
            t = self.t_c[sl].clone().detach().requires_grad_(True)
            phys = self.net_f(x, y, z, t)
            batch_loss = sum(torch.mean(torch.square(r)) for r in phys)
            weight = float(sl.stop - sl.start) / float(n_total)
            loss_sum = loss_sum + batch_loss * weight
        return loss_sum

    @staticmethod
    def _loss_scalar(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    def _record_losses(self, loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic) -> None:
        self.loss_rec.append(self._loss_scalar(loss))
        self.loss_components_rec["physics"].append(self._loss_scalar(loss_f))
        self.loss_components_rec["blade"].append(self._loss_scalar(loss_blade))
        self.loss_components_rec["inlet"].append(self._loss_scalar(loss_inlet))
        self.loss_components_rec["farfield"].append(self._loss_scalar(loss_far))
        self.loss_components_rec["outlet"].append(self._loss_scalar(loss_outlet))
        self.loss_components_rec["ic"].append(self._loss_scalar(loss_ic))

    def total_loss(self):
        u_b, v_b, w_b, _, _, _, _, _, _, _ = self.net_uvw(self.x_BLADE, self.y_BLADE, self.z_BLADE, self.t_BLADE)
        u_i, v_i, w_i, _, _, _, _, _, _, _ = self.net_uvw(self.x_INLET, self.y_INLET, self.z_INLET, self.t_INLET)
        u_f, v_f, w_f, _, _, _, _, _, _, _ = self.net_uvw(self.x_FAR, self.y_FAR, self.z_FAR, self.t_FAR)
        _, _, _, p_o, _, _, _, _, _, _ = self.net_uvw(self.x_OUT, self.y_OUT, self.z_OUT, self.t_OUT)
        u0, v0, w0, p0, _, _, _, _, _, _ = self.net_uvw(self.x_IC, self.y_IC, self.z_IC, self.t_IC)

        loss_f = self.physics_loss_batched()

        loss_blade = (
            torch.mean(torch.square(u_b - self.u_BLADE))
            + torch.mean(torch.square(v_b - self.v_BLADE))
            + torch.mean(torch.square(w_b - self.w_BLADE))
        )
        loss_inlet = (
            torch.mean(torch.square(u_i - self.u_INLET))
            + torch.mean(torch.square(v_i - self.v_INLET))
            + torch.mean(torch.square(w_i - self.w_INLET))
        )
        loss_far = (
            torch.mean(torch.square(u_f - self.u_FAR))
            + torch.mean(torch.square(v_f - self.v_FAR))
            + torch.mean(torch.square(w_f - self.w_FAR))
        )
        loss_outlet = torch.mean(torch.square(p_o))
        loss_ic = (
            torch.mean(torch.square(u0 - self.u_IC))
            + torch.mean(torch.square(v0 - self.v_IC))
            + torch.mean(torch.square(w0 - self.w_IC))
            + torch.mean(torch.square(p0 - self.p_IC))
        )
        loss = (
            loss_f
            + self.loss_weight_blade * loss_blade
            + self.loss_weight_inlet * loss_inlet
            + self.loss_weight_far * loss_far
            + self.loss_weight_outlet * loss_outlet
            + self.loss_weight_ic * loss_ic
        )
        return loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic

    def train(self, iters, learning_rate):
        self.optimizer_Adam.param_groups[0]["lr"] = learning_rate
        for it in range(iters):
            self.optimizer_Adam.zero_grad()
            loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic = self.total_loss()
            loss.backward()
            self.optimizer_Adam.step()
            self._record_losses(loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic)
            if it % 10 == 0:
                print(
                    f"It: {it}, total={loss.item():.3e}, phys={loss_f.item():.3e}, "
                    f"blade={loss_blade.item():.3e}, inlet={loss_inlet.item():.3e}, "
                    f"far={loss_far.item():.3e}, outlet={loss_outlet.item():.3e}, ic={loss_ic.item():.3e}"
                )

    def callback(self, loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic):
        self.count += 1
        self._record_losses(loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic)
        print(f"{self.count} th LBFGS iterations, Loss: {self._loss_scalar(loss):.6e}")

    def train_lbfgs(self, max_iter=50000, history_size=20):
        optimizer = optim.LBFGS(
            self.trainable_parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=max_iter,
            tolerance_grad=1e-10,
            tolerance_change=1e-10,
            history_size=history_size,
        )

        def closure():
            optimizer.zero_grad()
            loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic = self.total_loss()
            loss.backward()
            self.callback(loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet, loss_ic)
            return loss

        optimizer.step(closure)

    def predict(self, x_star, y_star, z_star, t_star, batch_size=65536):
        x_star = np.asarray(x_star, dtype=np.float32).reshape(-1, 1)
        y_star = np.asarray(y_star, dtype=np.float32).reshape(-1, 1)
        z_star = np.asarray(z_star, dtype=np.float32).reshape(-1, 1)
        t_star = np.asarray(t_star, dtype=np.float32).reshape(-1, 1)
        n_total = x_star.shape[0]
        u_list, v_list, w_list, p_list = [], [], [], []
        for module in self.model_modules():
            module.eval()
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                sl = slice(i, min(i + batch_size, n_total))
                x = torch.tensor(x_star[sl], dtype=torch.float32, device=self.device)
                y = torch.tensor(y_star[sl], dtype=torch.float32, device=self.device)
                z = torch.tensor(z_star[sl], dtype=torch.float32, device=self.device)
                t = torch.tensor(t_star[sl], dtype=torch.float32, device=self.device)
                u, v, w, p, _, _, _, _, _, _ = self.net_uvw(x, y, z, t)
                u_list.append(u.detach().cpu().numpy())
                v_list.append(v.detach().cpu().numpy())
                w_list.append(w.detach().cpu().numpy())
                p_list.append(p.detach().cpu().numpy())
        for module in self.model_modules():
            module.train()
        return np.vstack(u_list), np.vstack(v_list), np.vstack(w_list), np.vstack(p_list)


if __name__ == "__main__":
    # -----------------------------
    # Rotating-blade demo configuration
    # -----------------------------
    CPU_SAFE_MODE = True
    MAX_CPU_THREADS = 12
    BLADE_MESH_FILE = "blade.STL"
    STATIC_MESH_FILE = "pillar.STL"  # Optional stationary obstacle STL, e.g. a pillar/tower in the same source coordinates.
    BLADE_SCALE = 1.0
    BLADE_TARGET_LREF = 1.0
    BLADE_TRANSLATION = (0.35, 5.934, 1.75)
    BLADE_ROTATION_DEG = (0.0, -90.0, 0.0)
    STATIC_ROTATION_DEG = (90.0, 0.0, 0.0)
    STATIC_TRANSLATION = None  # Set an explicit (x, y, z) to override the auto tower placement below.
    STATIC_ALIGN_MODE = "tower_top_to_rotor"  # "tower_top_to_rotor" or "manual"
    STATIC_TOP_AXIS = "z"  # Which axis points from tower base to tower top after STATIC_ROTATION_DEG.
    STATIC_X_OFFSET_D = 0.25  # Positive means the tower sits downstream of the rotor center, so the blade is in front.
    REQUIRE_WATERTIGHT = True
    EXTRACT_BOUNDARY_FROM_VOLUME = False

    # This demo uses a true moving boundary:
    # 1) blade wall points are rotated in time
    # 2) the wall target velocity is omega x r
    # 3) interior collocation points inside the rotating blade are removed in time
    ROTOR_AXIS = (1.0, 0.0, 0.0)  # Wind-turbine-like rotation around the x-axis.
    ROTOR_CENTER_MODE = "blade_root"  # "blade_root", "mesh_center", or "manual"
    ROTOR_CENTER = None  # Used only when ROTOR_CENTER_MODE == "manual".
    TSR_TARGET = 6.0  # Tip-speed ratio used to estimate rotor speed from wind speed.
    RPM_LIMIT = 60.0  # Optional upper bound for the estimated rotor RPM.
    USE_AUTO_DOMAIN = False
    AUTO_DOMAIN = {
        "upstream_lengths": 2.5,
        "downstream_lengths": 6.0,
        "y_padding_lengths": 2.5,
        "z_padding_lengths": 2.5,
    }
    PLOT_USE_FOCUSED_DOMAIN = True
    PLOT_AUTO_DOMAIN = {
        "upstream_lengths": 0.8,
        "downstream_lengths": 3.0,
        "y_padding_lengths": 1.2,
        "z_padding_lengths": 1.2,
    }
    # Manual-domain extents in multiples of rotor diameter D, centered on rotor_center.
    MANUAL_DOMAIN_D = {
        "upstream": 10.0,
        "downstream": 15.0,
        "y_half_width": 5.0,
        "z_half_width": 5.0,
    }
    MANUAL_DOMAIN = None

    T_MIN = 0.0
    # Use an explicit 20 s window for export and diagnostics.
    # "rotor_periods" makes T_MAX equal to N_ROTOR_PERIODS revolutions after
    # omega is known. Use "manual" to keep T_MAX_MANUAL.
    TIME_WINDOW_MODE = "manual"
    N_ROTOR_PERIODS = 1.0
    T_MAX_MANUAL = 20.0
    RAMP_DURATION_S = 5.0

    # Keep inlet, farfield, and initial condition mutually consistent.
    # Current setup: 0-5 s ramps from 0 to U_MAX, then 5-20 s stays at U_MAX.
    # "ramp": startup from 0 to U_MAX; "steady": U_MAX everywhere from t=0.
    INFLOW_MODE = "ramp"
    RAMP_FRACTION = RAMP_DURATION_S / T_MAX_MANUAL

    RHO = 1.225
    MU = 1.8e-5
    U_MAX = 8.0

    # Loss weights. A stronger moving-wall term reduces over-smoothing near the blade.
    LOSS_WEIGHT_BLADE = 20.0
    LOSS_WEIGHT_INLET = 2.0
    LOSS_WEIGHT_FAR = 3.0
    LOSS_WEIGHT_OUTLET = 1.0
    LOSS_WEIGHT_IC = 2.0
    USE_ROTOR_AWARE_DECOMPOSITION = True
    ROTOR_AWARE_NEAR_RADIUS_FACTOR = 1.35
    ROTOR_AWARE_AXIAL_HALFWIDTH_D = 1.0
    ROTOR_AWARE_NEAR_TIME_FRACTION = (0.0, 1.0)
    ROTOR_AWARE_BLEND_SHARPNESS = 10.0

    # Lower-VRAM preset: narrower subnetworks and smaller physics batches.
    UV_LAYERS = [4] + 8 * [88] + [10]
    ADAM_ITERS = 12000
    LBFGS_MAX_ITER = 2000
    LBFGS_HISTORY_SIZE = 25
    LEARNING_RATE = 1e-4

    N_SURFACE = 900
    N_COLLO_BULK = 6500
    N_COLLO_NEAR = 5000
    N_COLLO_WAKE = 5000
    N_COLLO_ROTOR = 1800

    INLET_NY, INLET_NZ, INLET_NT = 11, 9, 21
    OUTLET_NY, OUTLET_NZ, OUTLET_NT = 11, 9, 21
    FAR_NX, FAR_NY, FAR_NZ, FAR_NT = 11,11, 9,11
    WALL_NT = 8
    IC_NX, IC_NY, IC_NZ = 9, 7, 7
    COLLO_BATCH_SIZE = 2048
    PREDICT_BATCH_SIZE = 24576

    SLICE_X = -0.07
    SLICE_Y = 7.9
    SLICE_Z = 4.5
    # None means the script will draw one full configured time window with uniform spacing.
    SLICE_TIMES = None
    SLICE_DT = 0.25

    OUTPUT_DIR = "rotating_demo_output"
    CHECKPOINT = f"{OUTPUT_DIR}/uvNN_blade3d_rotating_demo.pt"
    LOSS_FILE = f"{OUTPUT_DIR}/loss_history_blade3d_rotating_demo.pickle"
    LOSS_FIG = f"{OUTPUT_DIR}/loss_history_blade3d_rotating_demo.png"
    LOSS_COMPONENTS_FILE = f"{OUTPUT_DIR}/loss_components_blade3d_rotating_demo.pickle"
    LOSS_COMPONENTS_FIG = f"{OUTPUT_DIR}/loss_components_blade3d_rotating_demo.png"
    TRAINING_POINTS_FIG = f"{OUTPUT_DIR}/rotating_blade3d_training_points.png"
    PROBE_HISTORY_FIG = f"{OUTPUT_DIR}/rotating_blade3d_pressure_history.png"
    PROBE_SPEED_HISTORY_FIG = f"{OUTPUT_DIR}/rotating_blade3d_probe_speed_history.png"
    STREAMLINE_FIG = f"{OUTPUT_DIR}/rotating_blade3d_streamlines.png"
    ROTOR_PLANE_FIG = f"{OUTPUT_DIR}/rotating_blade3d_rotor_plane_speed.png"
    WAKE_PLANES_FIG = f"{OUTPUT_DIR}/rotating_blade3d_wake_planes_speed.png"

    SLICE_GIF = f"{OUTPUT_DIR}/rotating_blade3d_slice_animation.gif"


    ensure_dir(OUTPUT_DIR)

    configure_torch_cpu(MAX_CPU_THREADS)
    if CPU_SAFE_MODE:
        print(f"CPU safe mode enabled. Torch threads limited to {MAX_CPU_THREADS}.")

    blade_mesh_raw = load_blade_mesh(
        BLADE_MESH_FILE,
        scale=BLADE_SCALE,
        translation=(0.0, 0.0, 0.0),
        rotation_deg=BLADE_ROTATION_DEG,
        require_watertight=REQUIRE_WATERTIGHT,
        extract_boundary_from_volume=EXTRACT_BOUNDARY_FROM_VOLUME,
    )
    raw_lref = reference_length(blade_mesh_raw, axis="x")
    effective_scale = BLADE_SCALE if BLADE_TARGET_LREF is None else BLADE_SCALE * float(BLADE_TARGET_LREF) / raw_lref
    blade_mesh = load_blade_mesh(
        BLADE_MESH_FILE,
        scale=effective_scale,
        translation=BLADE_TRANSLATION,
        rotation_deg=BLADE_ROTATION_DEG,
        require_watertight=REQUIRE_WATERTIGHT,
        extract_boundary_from_volume=EXTRACT_BOUNDARY_FROM_VOLUME,
    )
    blade_surface = sample_surface_points(blade_mesh, N_SURFACE)
    base_probe = blade_reference_point(blade_mesh)
    bmin, bmax = mesh_bounds(blade_mesh)
    center = mesh_center(blade_mesh)
    lref = reference_length(blade_mesh, axis="x")

    if ROTOR_CENTER_MODE == "blade_root":
        rotor_center = base_probe.copy()
    elif ROTOR_CENTER_MODE == "mesh_center":
        rotor_center = center.copy()
    elif ROTOR_CENTER_MODE == "manual":
        if ROTOR_CENTER is None:
            raise ValueError("ROTOR_CENTER must be provided when ROTOR_CENTER_MODE is 'manual'.")
        rotor_center = np.asarray(ROTOR_CENTER, dtype=float)
    else:
        raise ValueError("ROTOR_CENTER_MODE must be 'blade_root', 'mesh_center', or 'manual'.")
    rotor_axis = _normalize(np.asarray(ROTOR_AXIS, dtype=float))
    rotor_radius = estimate_rotor_radius(np.asarray(blade_mesh.vertices), rotor_center, rotor_axis)
    if rotor_radius < 1e-8:
        raise ValueError("Estimated rotor radius is too small. Please check the blade mesh scale and rotor center.")
    rotor_diameter = 2.0 * rotor_radius
    omega_rad_s = float(TSR_TARGET) * float(U_MAX) / rotor_radius
    rotor_rpm = omega_rad_s * 60.0 / (2.0 * np.pi)
    if RPM_LIMIT is not None:
        rotor_rpm = min(rotor_rpm, float(RPM_LIMIT))
        omega_rad_s = 2.0 * np.pi * rotor_rpm / 60.0
    tip_speed = omega_rad_s * rotor_radius
    rotor_period = 2.0 * np.pi / max(abs(omega_rad_s), 1e-12)
    if str(TIME_WINDOW_MODE).lower() in {"rotor_period", "rotor_periods", "period", "periods", "revolutions"}:
        T_MAX = T_MIN + float(N_ROTOR_PERIODS) * rotor_period
    elif str(TIME_WINDOW_MODE).lower() == "manual":
        T_MAX = float(T_MAX_MANUAL)
    else:
        raise ValueError("TIME_WINDOW_MODE must be 'rotor_periods' or 'manual'.")

    rotor_decomposition_config = None
    if USE_ROTOR_AWARE_DECOMPOSITION:
        near_t0_frac, near_t1_frac = ROTOR_AWARE_NEAR_TIME_FRACTION
        near_t0_frac = float(near_t0_frac)
        near_t1_frac = float(near_t1_frac)
        if not (0.0 <= near_t0_frac < near_t1_frac <= 1.0):
            raise ValueError("ROTOR_AWARE_NEAR_TIME_FRACTION must satisfy 0 <= start < end <= 1.")
        rotor_decomposition_config = {
            "center_xyz": rotor_center,
            "axis_xyz": rotor_axis,
            "near_radius": float(ROTOR_AWARE_NEAR_RADIUS_FACTOR) * rotor_radius,
            "axial_halfwidth": float(ROTOR_AWARE_AXIAL_HALFWIDTH_D) * rotor_diameter,
            "time_window": (
                T_MIN + near_t0_frac * (T_MAX - T_MIN),
                T_MIN + near_t1_frac * (T_MAX - T_MIN),
            ),
            "blend_sharpness": float(ROTOR_AWARE_BLEND_SHARPNESS),
        }

    print(
        f"Rotor motion: mode={ROTOR_CENTER_MODE}, axis={rotor_axis}, center={rotor_center}, radius={rotor_radius:.6g} m, "
        f"TSR={float(TSR_TARGET):.6g}, RPM={rotor_rpm:.6g}, omega={omega_rad_s:.6g} rad/s, "
        f"tip_speed={tip_speed:.6g} m/s, period={rotor_period:.6g} s"
    )
    print(f"Calculated rotor diameter D = {rotor_diameter:.6g} m")
    if rotor_decomposition_config is not None:
        print(
            "Rotor-aware decomposition: near_radius={:.6g} m, axial_halfwidth={:.6g} m, time_window=({:.6g}, {:.6g}) s, "
            "blend_sharpness={:.3g}".format(
                rotor_decomposition_config["near_radius"],
                rotor_decomposition_config["axial_halfwidth"],
                rotor_decomposition_config["time_window"][0],
                rotor_decomposition_config["time_window"][1],
                rotor_decomposition_config["blend_sharpness"],
            )
        )
    static_mesh = None
    static_surface = np.empty((0, 3), dtype=float)
    if STATIC_MESH_FILE is not None:
        static_mesh_base = load_blade_mesh(
            STATIC_MESH_FILE,
            scale=effective_scale,
            translation=(0.0, 0.0, 0.0),
            rotation_deg=STATIC_ROTATION_DEG,
            require_watertight=REQUIRE_WATERTIGHT,
            extract_boundary_from_volume=EXTRACT_BOUNDARY_FROM_VOLUME,
        )
        if STATIC_ALIGN_MODE == "tower_top_to_rotor":
            sbmin0, sbmax0 = mesh_bounds(static_mesh_base)
            top_axis = str(STATIC_TOP_AXIS).lower()
            if top_axis == "x":
                top_center = np.array(
                    [sbmax0[0], 0.5 * (sbmin0[1] + sbmax0[1]), 0.5 * (sbmin0[2] + sbmax0[2])],
                    dtype=float,
                )
            elif top_axis == "y":
                top_center = np.array(
                    [0.5 * (sbmin0[0] + sbmax0[0]), sbmax0[1], 0.5 * (sbmin0[2] + sbmax0[2])],
                    dtype=float,
                )
            elif top_axis == "z":
                top_center = np.array(
                    [0.5 * (sbmin0[0] + sbmax0[0]), 0.5 * (sbmin0[1] + sbmax0[1]), sbmax0[2]],
                    dtype=float,
                )
            else:
                raise ValueError("STATIC_TOP_AXIS must be 'x', 'y', or 'z'.")
            target_top = np.asarray(rotor_center, dtype=float) + np.array([STATIC_X_OFFSET_D * rotor_diameter, 0.0, 0.0], dtype=float)
            static_translation = tuple((target_top - top_center).tolist()) if STATIC_TRANSLATION is None else tuple(STATIC_TRANSLATION)
        elif STATIC_ALIGN_MODE == "manual":
            static_translation = BLADE_TRANSLATION if STATIC_TRANSLATION is None else tuple(STATIC_TRANSLATION)
        else:
            raise ValueError("STATIC_ALIGN_MODE must be 'tower_top_to_rotor' or 'manual'.")
        static_mesh = load_blade_mesh(
            STATIC_MESH_FILE,
            scale=effective_scale,
            translation=static_translation,
            rotation_deg=STATIC_ROTATION_DEG,
            require_watertight=REQUIRE_WATERTIGHT,
            extract_boundary_from_volume=EXTRACT_BOUNDARY_FROM_VOLUME,
        )
        static_surface = sample_surface_points(static_mesh, max(512, N_SURFACE // 2))
        static_bmin, static_bmax = mesh_bounds(static_mesh)
        print(
            f"Static obstacle loaded: {STATIC_MESH_FILE}, rotation={STATIC_ROTATION_DEG}, translation={static_translation}, "
            f"bounds=({static_bmin} -> {static_bmax})"
        )
    if not USE_AUTO_DOMAIN:
        cx, cy, cz = (float(v) for v in rotor_center)
        manual_upstream = float(MANUAL_DOMAIN_D["upstream"]) * rotor_diameter
        manual_downstream = float(MANUAL_DOMAIN_D["downstream"]) * rotor_diameter
        manual_y_half = float(MANUAL_DOMAIN_D["y_half_width"]) * rotor_diameter
        manual_z_half = float(MANUAL_DOMAIN_D["z_half_width"]) * rotor_diameter
        MANUAL_DOMAIN = {
            "xmin": cx - manual_upstream,
            "xmax": cx + manual_downstream,
            "ymin": cy - manual_y_half,
            "ymax": cy + manual_y_half,
            "zmin": cz - manual_z_half,
            "zmax": cz + manual_z_half,
        }
    if USE_AUTO_DOMAIN:
        domain = auto_domain_from_mesh(blade_mesh, **AUTO_DOMAIN)
    else:
        domain = MANUAL_DOMAIN.copy()
    plot_mesh = combine_meshes([blade_mesh, static_mesh])
    plot_domain = auto_domain_from_mesh(plot_mesh, **PLOT_AUTO_DOMAIN) if PLOT_USE_FOCUSED_DOMAIN else domain.copy()

    xmin, xmax = domain["xmin"], domain["xmax"]
    ymin, ymax = domain["ymin"], domain["ymax"]
    zmin, zmax = domain["zmin"], domain["zmax"]
    lb = np.array([xmin, ymin, zmin, T_MIN], dtype=float)
    ub = np.array([xmax, ymax, zmax, T_MAX], dtype=float)

    inlet_xyz = make_plane_points_x(xmin, ymin, ymax, zmin, zmax, ny=INLET_NY, nz=INLET_NZ)
    inlet_xyz = remove_points_inside_obstacles(
        np.hstack([inlet_xyz, np.full((inlet_xyz.shape[0], 1), T_MIN)]),
        blade_mesh,
        rotor_center,
        rotor_axis,
        omega_rad_s,
        t_ref=T_MIN,
        static_mesh=static_mesh,
    )[:, :3]
    inlet_t = np.linspace(T_MIN, T_MAX, INLET_NT)
    INLET = np.vstack([
        attach_inflow_profile(
            np.hstack([inlet_xyz, np.full((inlet_xyz.shape[0], 1), ti)]),
            T_MIN,
            T_MAX,
            U_MAX,
            mode=INFLOW_MODE,
            ramp_fraction=RAMP_FRACTION,
        )
        for ti in inlet_t
    ])

    outlet_xyz = make_plane_points_x(xmax, ymin, ymax, zmin, zmax, ny=OUTLET_NY, nz=OUTLET_NZ)
    outlet_t = np.linspace(T_MIN, T_MAX, OUTLET_NT)
    OUTLET = np.vstack([np.hstack([outlet_xyz, np.full((outlet_xyz.shape[0], 1), ti)]) for ti in outlet_t])
    OUTLET = remove_points_inside_obstacles(OUTLET, blade_mesh, rotor_center, rotor_axis, omega_rad_s, t_ref=T_MIN, static_mesh=static_mesh)

    far_ymin = make_plane_points_y(xmin, xmax, ymin, zmin, zmax, nx=FAR_NX, nz=FAR_NZ)
    far_ymax = make_plane_points_y(xmin, xmax, ymax, zmin, zmax, nx=FAR_NX, nz=FAR_NZ)
    far_zmin = make_plane_points_z(xmin, xmax, ymin, ymax, zmin, nx=FAR_NX, ny=FAR_NY)
    far_zmax = make_plane_points_z(xmin, xmax, ymin, ymax, zmax, nx=FAR_NX, ny=FAR_NY)
    farfield_xyz = np.vstack([far_ymin, far_ymax, far_zmin, far_zmax])
    far_t = np.linspace(T_MIN, T_MAX, FAR_NT)
    FARFIELD = np.vstack([
        attach_inflow_profile(
            np.hstack([farfield_xyz, np.full((farfield_xyz.shape[0], 1), ti)]),
            T_MIN,
            T_MAX,
            U_MAX,
            mode=INFLOW_MODE,
            ramp_fraction=RAMP_FRACTION,
        )
        for ti in far_t
    ])
    FARFIELD_XYZT = remove_points_inside_obstacles(
        FARFIELD[:, :4], blade_mesh, rotor_center, rotor_axis, omega_rad_s, t_ref=T_MIN, static_mesh=static_mesh
    )
    FARFIELD = attach_inflow_profile(
        FARFIELD_XYZT,
        T_MIN,
        T_MAX,
        U_MAX,
        mode=INFLOW_MODE,
        ramp_fraction=RAMP_FRACTION,
    )

    BLADE = build_rotating_blade_dataset(
        blade_surface,
        tmin=T_MIN,
        tmax=T_MAX,
        num_t=WALL_NT,
        center_xyz=rotor_center,
        axis_xyz=rotor_axis,
        omega_rad_s=omega_rad_s,
        t_ref=T_MIN,
    )
    STATIC_WALL = build_static_wall_dataset(static_surface, tmin=T_MIN, tmax=T_MAX, num_t=WALL_NT) if static_mesh is not None else np.empty((0, 7), dtype=float)
    WALL = np.vstack([BLADE, STATIC_WALL]) if STATIC_WALL.size > 0 else BLADE

    IC = cart_grid_4d(xmin, xmax, ymin, ymax, zmin, zmax, T_MIN, T_MIN, nx=IC_NX, ny=IC_NY, nz=IC_NZ, nt=1)
    IC = remove_points_inside_obstacles(IC, blade_mesh, rotor_center, rotor_axis, omega_rad_s, t_ref=T_MIN, static_mesh=static_mesh)
    IC = attach_initial_condition_profile(
        IC,
        T_MIN,
        T_MAX,
        U_MAX,
        mode=INFLOW_MODE,
        ramp_fraction=RAMP_FRACTION,
        pressure_value=0.0,
    )

    bulk_xyzt = sample_box_lhs(lb, ub, N_COLLO_BULK)
    near_lb, near_ub = build_refinement_bounds_4d(domain, bmin, bmax, 0.75, 1.25, 0.75, 0.75, lref, T_MIN, T_MAX)
    near_xyzt = sample_box_lhs(near_lb, near_ub, N_COLLO_NEAR)
    wake_lb = np.array([max(xmin, bmax[0] - 0.10 * lref), max(ymin, center[1] - 1.00 * lref), max(zmin, center[2] - 1.00 * lref), T_MIN], dtype=float)
    wake_ub = np.array([min(xmax, bmax[0] + 6.00 * lref), min(ymax, center[1] + 1.00 * lref), min(zmax, center[2] + 1.00 * lref), T_MAX], dtype=float)
    wake_xyzt = sample_box_lhs(wake_lb, wake_ub, N_COLLO_WAKE)
    rotor_xyzt = np.empty((0, 4), dtype=float)
    if rotor_decomposition_config is not None and N_COLLO_ROTOR > 0:
        rotor_xyzt = sample_rotor_cylinder_lhs(
            rotor_center,
            rotor_axis,
            radial_radius=rotor_decomposition_config["near_radius"],
            axial_halfwidth=rotor_decomposition_config["axial_halfwidth"],
            tmin=rotor_decomposition_config["time_window"][0],
            tmax=rotor_decomposition_config["time_window"][1],
            n_samples=N_COLLO_ROTOR,
        )
        rotor_xyzt = remove_points_inside_obstacles(
            rotor_xyzt,
            blade_mesh,
            rotor_center,
            rotor_axis,
            omega_rad_s,
            t_ref=T_MIN,
            static_mesh=static_mesh,
        )

    XYZT_c = np.vstack([bulk_xyzt, near_xyzt, wake_xyzt, rotor_xyzt])
    XYZT_c = remove_points_inside_obstacles(XYZT_c, blade_mesh, rotor_center, rotor_axis, omega_rad_s, t_ref=T_MIN, static_mesh=static_mesh)
    XYZT_c = np.vstack([XYZT_c, WALL[:, :4], OUTLET, INLET[:, :4], FARFIELD[:, :4], IC[:, :4]])

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Inflow mode: {INFLOW_MODE}, U_MAX={U_MAX:.6g} m/s, ramp_fraction={RAMP_FRACTION:.3g}")
    print(f"Time window (s): [{T_MIN:.6g}, {T_MAX:.6g}] ({(T_MAX - T_MIN) / rotor_period:.4g} rotor periods)")
    print(f"Fluid properties: rho={RHO:.6g} kg/m^3, mu={MU:.6g} Pa*s")
    print(
        "Loss weights: blade={:.3g}, inlet={:.3g}, far={:.3g}, outlet={:.3g}, ic={:.3g}".format(
            LOSS_WEIGHT_BLADE, LOSS_WEIGHT_INLET, LOSS_WEIGHT_FAR, LOSS_WEIGHT_OUTLET, LOSS_WEIGHT_IC
        )
    )
    print(
        f"Rotor-aware mode: {'enabled' if rotor_decomposition_config is not None else 'disabled'}, "
        f"extra rotor collocation={rotor_xyzt.shape[0]}"
    )
    print("Domain:", domain)
    print("Plot domain:", plot_domain)
    print("Collocation cloud shape:", XYZT_c.shape)
    plot_training_points_unsteady(XYZT_c, WALL, TRAINING_POINTS_FIG, display_domain=plot_domain)

    model = PINNBladeRotatingUnsteady3D(
        XYZT_c,
        IC,
        INLET,
        FARFIELD,
        OUTLET,
        WALL,
        UV_LAYERS,
        lb,
        ub,
        rho=RHO,
        mu=MU,
        collo_batch_size=COLLO_BATCH_SIZE,
        loss_weight_blade=LOSS_WEIGHT_BLADE,
        loss_weight_inlet=LOSS_WEIGHT_INLET,
        loss_weight_far=LOSS_WEIGHT_FAR,
        loss_weight_outlet=LOSS_WEIGHT_OUTLET,
        loss_weight_ic=LOSS_WEIGHT_IC,
        decomposition_config=rotor_decomposition_config,
    )

    start_time = time.time()
    model.train(iters=ADAM_ITERS, learning_rate=LEARNING_RATE)
    if LBFGS_MAX_ITER > 0:
        model.train_lbfgs(max_iter=LBFGS_MAX_ITER, history_size=LBFGS_HISTORY_SIZE)
    print(f"--- {time.time() - start_time:.2f} seconds ---")

    model.save_NN(CHECKPOINT)
    with open(LOSS_FILE, "wb") as f:
        pickle.dump(model.loss_rec, f)
    with open(LOSS_COMPONENTS_FILE, "wb") as f:
        pickle.dump(model.loss_components_rec, f)
    plot_loss_history(model.loss_rec, LOSS_FIG)
    plot_loss_components_history(model.loss_components_rec, LOSS_COMPONENTS_FIG)

    t_hist = np.linspace(T_MIN, T_MAX, 200)[:, None]
    probe_xyz = rotate_points_about_axis_time(np.tile(base_probe.reshape(1, 3), (t_hist.shape[0], 1)), rotor_center, rotor_axis, omega_rad_s * (t_hist[:, 0] - T_MIN))
    u_hist, v_hist, w_hist, p_hist = model.predict(
        probe_xyz[:, 0:1], probe_xyz[:, 1:2], probe_xyz[:, 2:3], t_hist, batch_size=PREDICT_BATCH_SIZE
    )
    plot_probe_history(t_hist, p_hist, PROBE_HISTORY_FIG)
    speed_hist = np.sqrt(u_hist[:, 0] ** 2 + v_hist[:, 0] ** 2 + w_hist[:, 0] ** 2)
    plot_time_history(
        t_hist,
        speed_hist,
        ylabel="Speed (m/s)",
        title="Probe speed history",
        outfile=PROBE_SPEED_HISTORY_FIG,
    )

    x0, y0, z0 = orthogonal_slice_positions(blade_mesh)
    x_slice = float(x0 if SLICE_X is None else SLICE_X)
    y_slice = float(y0 if SLICE_Y is None else SLICE_Y)
    z_slice = float(z0 if SLICE_Z is None else SLICE_Z)
    print(f"Slice positions: SLICE_X={x_slice:.6g}, SLICE_Y={y_slice:.6g}, SLICE_Z={z_slice:.6g}")
    wake_x_positions = [min(domain["xmax"], bmax[0] + factor * lref) for factor in (0.25, 0.75, 1.50, 2.50)]
    if SLICE_TIMES is None:
        slice_times = list(np.arange(T_MIN, T_MAX + 0.5 * SLICE_DT, SLICE_DT))
    else:
        slice_times = list(SLICE_TIMES)
    for stale_slice in Path(OUTPUT_DIR).glob("rotating_blade3d_orthogonal_slices_t_*.png"):
        stale_slice.unlink()
    slice_files = []

    for ti in slice_times:
        angle = omega_rad_s * (float(ti) - T_MIN)
        mesh_t = rotate_mesh_about_axis(blade_mesh, rotor_center, rotor_axis, angle)
        plot_mesh_t = combine_meshes([mesh_t, static_mesh])
        outfile = f"{OUTPUT_DIR}/rotating_blade3d_orthogonal_slices_t_{ti:.4f}.png"
        plot_orthogonal_slices_unsteady(
            model,
            plot_mesh_t,
            plot_domain,
            x_slice=x_slice,
            y_slice=y_slice,
            z_slice=z_slice,
            t_value=float(ti),
            outfile=outfile,
            batch_size=PREDICT_BATCH_SIZE,
        )
        slice_files.append(outfile)
        speed_stats = slice_speed_summary_unsteady(model, plot_domain, x_slice, y_slice, z_slice, float(ti), n1=33, n2=33)
        u_inf_t = float(inflow_speed_at_time(float(ti), T_MIN, T_MAX, U_MAX, mode=INFLOW_MODE, ramp_fraction=RAMP_FRACTION))
        pert_stats = slice_perturbation_summary_unsteady(
            model, plot_domain, x_slice, y_slice, z_slice, float(ti), freestream_u=u_inf_t, n1=33, n2=33
        )
        print(
            "Frame diagnostics t={:.4f}s -> speed X[min,max,mean]=({:.4f}, {:.4f}, {:.4f}), Y=({:.4f}, {:.4f}, {:.4f}), Z=({:.4f}, {:.4f}, {:.4f})".format(
                float(ti),
                speed_stats["x"][0], speed_stats["x"][1], speed_stats["x"][2],
                speed_stats["y"][0], speed_stats["y"][1], speed_stats["y"][2],
                speed_stats["z"][0], speed_stats["z"][1], speed_stats["z"][2],
            )
        )
        print(
            "                     perturbation |u-Uinf,v,w| with Uinf={:.4f} -> X=({:.4f}, {:.4f}, {:.4f}), Y=({:.4f}, {:.4f}, {:.4f}), Z=({:.4f}, {:.4f}, {:.4f})".format(
                u_inf_t,
                pert_stats["x"][0], pert_stats["x"][1], pert_stats["x"][2],
                pert_stats["y"][0], pert_stats["y"][1], pert_stats["y"][2],
                pert_stats["z"][0], pert_stats["z"][1], pert_stats["z"][2],
            )
        )

        print(f"Saved orthogonal slice figure to: {outfile}")
    save_slice_gif(slice_files, SLICE_GIF, duration_s=0.45)
    print(f"Saved slice animation to: {SLICE_GIF}")

    preview_t = float(slice_times[-1])
    preview_angle = omega_rad_s * (preview_t - T_MIN)
    preview_mesh = rotate_mesh_about_axis(blade_mesh, rotor_center, rotor_axis, preview_angle)
    preview_surface = rotate_points_about_axis(blade_surface, rotor_center, rotor_axis, preview_angle)
    preview_plot_mesh = combine_meshes([preview_mesh, static_mesh])
    preview_plot_surface = np.vstack([preview_surface, static_surface]) if static_mesh is not None else preview_surface
    plot_streamlines_unsteady(
        model,
        preview_plot_mesh,
        plot_domain,
        preview_plot_surface,
        t_value=preview_t,
        outfile=STREAMLINE_FIG,
        batch_size=PREDICT_BATCH_SIZE,
    )
    plot_rotor_plane_unsteady(
        model,
        preview_plot_mesh,
        x_value=x_slice,
        t_value=preview_t,
        domain=plot_domain,
        outfile=ROTOR_PLANE_FIG,
        batch_size=PREDICT_BATCH_SIZE,
    )
    plot_wake_planes_unsteady(
        model,
        preview_plot_mesh,
        x_positions=wake_x_positions,
        t_value=preview_t,
        domain=plot_domain,
        outfile=WAKE_PLANES_FIG,
        batch_size=PREDICT_BATCH_SIZE,
    )

    final_loss = model.total_loss()
    print(
        "Final losses -> total: {:.3e}, phys: {:.3e}, blade: {:.3e}, inlet: {:.3e}, far: {:.3e}, outlet: {:.3e}, ic: {:.3e}".format(
            final_loss[0].item(),
            final_loss[1].item(),
            final_loss[2].item(),
            final_loss[3].item(),
            final_loss[4].item(),
            final_loss[5].item(),
            final_loss[6].item(),
        )
    )
    print(f"Saved checkpoint to: {CHECKPOINT}")
    print(f"Saved loss history to: {LOSS_FILE}")
    print(f"Saved loss figure to: {LOSS_FIG}")
    print(f"Saved component loss history to: {LOSS_COMPONENTS_FILE}")
    print(f"Saved component loss figure to: {LOSS_COMPONENTS_FIG}")
    print(f"Saved training-point figure to: {TRAINING_POINTS_FIG}")
    print(f"Saved pressure history to: {PROBE_HISTORY_FIG}")
    print(f"Saved probe speed history to: {PROBE_SPEED_HISTORY_FIG}")
    print(f"Saved streamline figure to: {STREAMLINE_FIG}")
    print(f"Saved rotor-plane figure to: {ROTOR_PLANE_FIG}")
    print(f"Saved wake-plane figure to: {WAKE_PLANES_FIG}")
