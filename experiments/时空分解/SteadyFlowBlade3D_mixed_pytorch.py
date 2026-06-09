from __future__ import annotations

import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from blade_geometry_3d import (
    auto_domain_from_mesh,
    load_blade_mesh,
    mesh_bounds,
    mesh_center,
    orthogonal_slice_positions,
    reference_length,
    remove_points_inside_mesh,
    sample_surface_points,
)
from blade_pinn_3d_common import (
    attach_velocity,
    build_refinement_bounds_3d,
    ensure_dir,
    make_plane_points_x,
    make_plane_points_y,
    make_plane_points_z,
    plot_loss_history,
    plot_orthogonal_slices_steady,
    plot_training_points_steady,
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


class PINNBladeSteady3D:
    def __init__(
        self,
        Collo,
        INLET,
        FARFIELD,
        OUTLET,
        BLADE,
        uv_layers,
        lb,
        ub,
        rho=1.0,
        mu=0.02,
        device=None,
        ExistModel=0,
        uvDir="",
    ):
        self.count = 0
        self.loss_rec = []
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lb = torch.tensor(lb, dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=self.device)

        self.rho = float(rho)
        self.mu = float(mu)

        self.x_c = torch.tensor(Collo[:, 0:1], dtype=torch.float32, device=self.device, requires_grad=True)
        self.y_c = torch.tensor(Collo[:, 1:2], dtype=torch.float32, device=self.device, requires_grad=True)
        self.z_c = torch.tensor(Collo[:, 2:3], dtype=torch.float32, device=self.device, requires_grad=True)

        self.x_INLET = torch.tensor(INLET[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_INLET = torch.tensor(INLET[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_INLET = torch.tensor(INLET[:, 2:3], dtype=torch.float32, device=self.device)
        self.u_INLET = torch.tensor(INLET[:, 3:4], dtype=torch.float32, device=self.device)
        self.v_INLET = torch.tensor(INLET[:, 4:5], dtype=torch.float32, device=self.device)
        self.w_INLET = torch.tensor(INLET[:, 5:6], dtype=torch.float32, device=self.device)

        self.x_FAR = torch.tensor(FARFIELD[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_FAR = torch.tensor(FARFIELD[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_FAR = torch.tensor(FARFIELD[:, 2:3], dtype=torch.float32, device=self.device)
        self.u_FAR = torch.tensor(FARFIELD[:, 3:4], dtype=torch.float32, device=self.device)
        self.v_FAR = torch.tensor(FARFIELD[:, 4:5], dtype=torch.float32, device=self.device)
        self.w_FAR = torch.tensor(FARFIELD[:, 5:6], dtype=torch.float32, device=self.device)

        self.x_OUT = torch.tensor(OUTLET[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_OUT = torch.tensor(OUTLET[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_OUT = torch.tensor(OUTLET[:, 2:3], dtype=torch.float32, device=self.device)

        self.x_BLADE = torch.tensor(BLADE[:, 0:1], dtype=torch.float32, device=self.device)
        self.y_BLADE = torch.tensor(BLADE[:, 1:2], dtype=torch.float32, device=self.device)
        self.z_BLADE = torch.tensor(BLADE[:, 2:3], dtype=torch.float32, device=self.device)

        self.uv_layers = uv_layers
        self.net = self.initialize_NN(self.uv_layers).to(self.device)

        if ExistModel == 1:
            print("Loading 3D steady NN ...")
            self.load_NN(uvDir)

        self.optimizer_Adam = optim.Adam(self.net.parameters(), lr=5e-4)

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

    def save_NN(self, fileDir):
        torch.save(self.net.state_dict(), fileDir)
        print(f"Saved NN parameters to: {fileDir}")

    def load_NN(self, fileDir):
        self.net.load_state_dict(torch.load(fileDir, map_location=self.device))
        print("Loaded NN parameters successfully...")

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def net_uvw(self, x, y, z):
        X = torch.cat([x, y, z], dim=1)
        out = self.neural_net(X)
        u = out[:, 0:1]
        v = out[:, 1:2]
        w = out[:, 2:3]
        p = out[:, 3:4]
        sxx = out[:, 4:5]
        syy = out[:, 5:6]
        szz = out[:, 6:7]
        sxy = out[:, 7:8]
        sxz = out[:, 8:9]
        syz = out[:, 9:10]
        return u, v, w, p, sxx, syy, szz, sxy, sxz, syz

    def net_f(self, x, y, z):
        rho = self.rho
        mu = self.mu
        u, v, w, p, sxx, syy, szz, sxy, sxz, syz = self.net_uvw(x, y, z)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]

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

        r_mx = rho * (u * u_x + v * u_y + w * u_z) - (sxx_x + sxy_y + sxz_z)
        r_my = rho * (u * v_x + v * v_y + w * v_z) - (sxy_x + syy_y + syz_z)
        r_mz = rho * (u * w_x + v * w_y + w * w_z) - (sxz_x + syz_y + szz_z)

        r_sxx = -p + 2.0 * mu * u_x - sxx
        r_syy = -p + 2.0 * mu * v_y - syy
        r_szz = -p + 2.0 * mu * w_z - szz
        r_sxy = mu * (u_y + v_x) - sxy
        r_sxz = mu * (u_z + w_x) - sxz
        r_syz = mu * (v_z + w_y) - syz
        r_p = p + (sxx + syy + szz) / 3.0

        return r_c, r_mx, r_my, r_mz, r_sxx, r_syy, r_szz, r_sxy, r_sxz, r_syz, r_p

    def total_loss(self):
        u_b, v_b, w_b, _, _, _, _, _, _, _ = self.net_uvw(self.x_BLADE, self.y_BLADE, self.z_BLADE)
        u_i, v_i, w_i, _, _, _, _, _, _, _ = self.net_uvw(self.x_INLET, self.y_INLET, self.z_INLET)
        u_f, v_f, w_f, _, _, _, _, _, _, _ = self.net_uvw(self.x_FAR, self.y_FAR, self.z_FAR)
        _, _, _, p_o, _, _, _, _, _, _ = self.net_uvw(self.x_OUT, self.y_OUT, self.z_OUT)

        phys = self.net_f(self.x_c, self.y_c, self.z_c)
        loss_f = sum(torch.mean(torch.square(r)) for r in phys)

        loss_blade = (
            torch.mean(torch.square(u_b))
            + torch.mean(torch.square(v_b))
            + torch.mean(torch.square(w_b))
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

        loss = loss_f + 5.0 * loss_blade + 2.0 * loss_inlet + 2.0 * loss_far + 1.0 * loss_outlet
        return loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet

    def train(self, iters, learning_rate):
        self.optimizer_Adam.param_groups[0]["lr"] = learning_rate

        for it in range(iters):
            self.optimizer_Adam.zero_grad()
            loss, loss_f, loss_blade, loss_inlet, loss_far, loss_outlet = self.total_loss()
            loss.backward()
            self.optimizer_Adam.step()

            self.loss_rec.append(loss.item())
            if it % 10 == 0:
                print(
                    f"It: {it}, total={loss.item():.3e}, phys={loss_f.item():.3e}, "
                    f"blade={loss_blade.item():.3e}, inlet={loss_inlet.item():.3e}, "
                    f"far={loss_far.item():.3e}, outlet={loss_outlet.item():.3e}"
                )

    def callback(self, loss):
        self.count += 1
        self.loss_rec.append(loss)
        print(f"{self.count} th LBFGS iterations, Loss: {loss}")

    def train_lbfgs(self, max_iter=50000):
        optimizer = optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=max_iter,
            tolerance_grad=1e-10,
            tolerance_change=1e-10,
            history_size=50,
        )

        def closure():
            optimizer.zero_grad()
            loss, *_ = self.total_loss()
            loss.backward()
            self.callback(loss.item())
            return loss

        optimizer.step(closure)

    def predict(self, x_star, y_star, z_star, batch_size=65536):
        x_star = np.asarray(x_star, dtype=np.float32).reshape(-1, 1)
        y_star = np.asarray(y_star, dtype=np.float32).reshape(-1, 1)
        z_star = np.asarray(z_star, dtype=np.float32).reshape(-1, 1)
        n_total = x_star.shape[0]

        u_list, v_list, w_list, p_list = [], [], [], []
        self.net.eval()
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                sl = slice(i, min(i + batch_size, n_total))
                x = torch.tensor(x_star[sl], dtype=torch.float32, device=self.device)
                y = torch.tensor(y_star[sl], dtype=torch.float32, device=self.device)
                z = torch.tensor(z_star[sl], dtype=torch.float32, device=self.device)
                u, v, w, p, _, _, _, _, _, _ = self.net_uvw(x, y, z)
                u_list.append(u.detach().cpu().numpy())
                v_list.append(v.detach().cpu().numpy())
                w_list.append(w.detach().cpu().numpy())
                p_list.append(p.detach().cpu().numpy())
        self.net.train()
        return (
            np.vstack(u_list),
            np.vstack(v_list),
            np.vstack(w_list),
            np.vstack(p_list),
        )


if __name__ == "__main__":
    # -----------------------------
    # User configuration
    # -----------------------------
    CPU_SAFE_MODE = True
    MAX_CPU_THREADS = 16  # Good balance for a 32-core CPU without making the desktop unusable.

    BLADE_MESH_FILE = "blade.STL"  # Preferred: watertight STL/OBJ/PLY. If this file is missing and exactly one STL exists in the folder, it will be auto-used.
    BLADE_SCALE = 10.0  # Optional pre-scale for STL units before auto-normalization.
    BLADE_TARGET_LREF = 1.0  # Set to None to disable auto-normalization and keep the raw mesh scale.
    BLADE_TRANSLATION = (0.35, 0.30, 1)                    
    BLADE_ROTATION_DEG = (0.0, -90.0, 0.0)  # Rotate the blade so the mean flow is along +x.
    REQUIRE_WATERTIGHT = True
    EXTRACT_BOUNDARY_FROM_VOLUME = False  # Only set True for a blade-only solid mesh file, not a full fluid-domain mesh.

    USE_AUTO_DOMAIN = True
    # 计算域模板：
    # 单个叶片细节�    # AUTO_DOMAIN = {"upstream_lengths": 2.0, "downstream_lengths": 5.0, "y_padding_lengths": 2.0, "z_padding_lengths": 2.0}
    # PLOT_AUTO_DOMAIN = {"upstream_lengths": 0.4, "downstream_lengths": 0.8, "y_padding_lengths": 0.5, "z_padding_lengths": 0.5}
    # 整个三叶风轮�?    # AUTO_DOMAIN = {"upstream_lengths": 3.0, "downstream_lengths": 6.0, "y_padding_lengths": 3.0, "z_padding_lengths": 3.0}
    # PLOT_AUTO_DOMAIN = {"upstream_lengths": 0.8, "downstream_lengths": 1.2, "y_padding_lengths": 1.0, "z_padding_lengths": 1.0}
    # 长尾迹显示：
    # AUTO_DOMAIN = {"upstream_lengths": 3.0, "downstream_lengths": 10.0, "y_padding_lengths": 3.0, "z_padding_lengths": 3.0}
    # PLOT_AUTO_DOMAIN = {"upstream_lengths": 1.0, "downstream_lengths": 3.0, "y_padding_lengths": 1.2, "z_padding_lengths": 1.2}
    AUTO_DOMAIN = {
        "upstream_lengths": 2.0,
        "downstream_lengths": 5.0,
        "y_padding_lengths": 2.0,
        "z_padding_lengths": 2.0,
    }
    PLOT_USE_FOCUSED_DOMAIN = True
    PLOT_AUTO_DOMAIN = {
        "upstream_lengths": 0.4,
        "downstream_lengths": 0.8,
        "y_padding_lengths": 0.5,
        "z_padding_lengths": 0.5,
    }
    MANUAL_DOMAIN = {
        "xmin": 0.0,
        "xmax": 1.2,
        "ymin": 0.0,
        "ymax": 0.4,
        "zmin": 0.0,
        "zmax": 0.3,
    }

    # Physical units: geometry in meters, velocity in m/s, density in kg/m^3, dynamic viscosity in Pa*s.
    # If the blade mesh is not modeled in meters, rescale it before using real wind speeds here.
    RHO = 1.225
    MU = 1.8e-5
    U_INF = 8.0
    V_INF = 0.0
    W_INF = 0.0

    UV_LAYERS = [3] + 8 * [96] + [10]
    ADAM_ITERS = 10000
    LBFGS_MAX_ITER = 100000
    LEARNING_RATE = 5e-4

    N_SURFACE = 5000
    N_COLLO_BULK = 50000
    N_COLLO_NEAR = 18000
    N_COLLO_WAKE = 18000

    INLET_NY, INLET_NZ = 33, 25
    FAR_NX, FAR_NY, FAR_NZ = 49, 33, 25

    SLICE_X = None  # If None, the script uses the blade bounding-box center.
    SLICE_Y = None
    SLICE_Z = None

    OUTPUT_DIR = "steady_output"
    CHECKPOINT = f"{OUTPUT_DIR}/uvNN_blade3d_steady_pytorch.pt"
    LOSS_FILE = f"{OUTPUT_DIR}/loss_history_blade3d_steady.pickle"
    LOSS_FIG = f"{OUTPUT_DIR}/loss_history_blade3d_steady.png"
    TRAINING_POINTS_FIG = f"{OUTPUT_DIR}/steady_blade3d_training_points.png"
    ORTHOGONAL_SLICE_FIG = f"{OUTPUT_DIR}/steady_blade3d_orthogonal_slices.png"
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
    raw_bmin, raw_bmax = mesh_bounds(blade_mesh_raw)
    raw_lref = reference_length(blade_mesh_raw, axis="x")

    effective_scale = BLADE_SCALE
    if BLADE_TARGET_LREF is not None:
        effective_scale = BLADE_SCALE * float(BLADE_TARGET_LREF) / raw_lref

    blade_mesh = load_blade_mesh(
        BLADE_MESH_FILE,
        scale=effective_scale,
        translation=BLADE_TRANSLATION,
        rotation_deg=BLADE_ROTATION_DEG,
        require_watertight=REQUIRE_WATERTIGHT,
        extract_boundary_from_volume=EXTRACT_BOUNDARY_FROM_VOLUME,
    )
    blade_surface = sample_surface_points(blade_mesh, N_SURFACE)
    bmin, bmax = mesh_bounds(blade_mesh)
    center = mesh_center(blade_mesh)
    lref = reference_length(blade_mesh, axis="x")

    print("Blade raw bounds:", raw_bmin, raw_bmax)
    print(f"Blade raw reference length (x-span): {raw_lref:.6g}")
    if BLADE_TARGET_LREF is None:
        print(f"Blade normalization disabled. Using user scale={BLADE_SCALE:.6g}")
    else:
        print(
            f"Blade target reference length: {float(BLADE_TARGET_LREF):.6g}, "
            f"effective scale={effective_scale:.6g}"
        )
    print("Blade final bounds:", bmin, bmax)
    print(f"Blade final reference length (x-span): {lref:.6g}")

    if USE_AUTO_DOMAIN:
        domain = auto_domain_from_mesh(blade_mesh, **AUTO_DOMAIN)
    else:
        domain = MANUAL_DOMAIN.copy()

    if PLOT_USE_FOCUSED_DOMAIN:
        plot_domain = auto_domain_from_mesh(blade_mesh, **PLOT_AUTO_DOMAIN)
    else:
        plot_domain = domain.copy()

    xmin, xmax = domain["xmin"], domain["xmax"]
    ymin, ymax = domain["ymin"], domain["ymax"]
    zmin, zmax = domain["zmin"], domain["zmax"]
    lb = np.array([xmin, ymin, zmin], dtype=float)
    ub = np.array([xmax, ymax, zmax], dtype=float)

    inlet_xyz = make_plane_points_x(xmin, ymin, ymax, zmin, zmax, ny=INLET_NY, nz=INLET_NZ)
    inlet_xyz = remove_points_inside_mesh(inlet_xyz, blade_mesh)
    INLET = attach_velocity(inlet_xyz, U_INF, V_INF, W_INF)

    outlet_xyz = make_plane_points_x(xmax, ymin, ymax, zmin, zmax, ny=INLET_NY, nz=INLET_NZ)
    outlet_xyz = remove_points_inside_mesh(outlet_xyz, blade_mesh)
    OUTLET = outlet_xyz

    far_ymin = make_plane_points_y(xmin, xmax, ymin, zmin, zmax, nx=FAR_NX, nz=FAR_NZ)
    far_ymax = make_plane_points_y(xmin, xmax, ymax, zmin, zmax, nx=FAR_NX, nz=FAR_NZ)
    far_zmin = make_plane_points_z(xmin, xmax, ymin, ymax, zmin, nx=FAR_NX, ny=FAR_NY)
    far_zmax = make_plane_points_z(xmin, xmax, ymin, ymax, zmax, nx=FAR_NX, ny=FAR_NY)
    farfield_xyz = np.vstack([far_ymin, far_ymax, far_zmin, far_zmax])
    farfield_xyz = remove_points_inside_mesh(farfield_xyz, blade_mesh)
    FARFIELD = attach_velocity(farfield_xyz, U_INF, V_INF, W_INF)

    BLADE = blade_surface

    bulk_xyz = sample_box_lhs(lb, ub, N_COLLO_BULK)
    near_lb, near_ub = build_refinement_bounds_3d(
        domain,
        bmin,
        bmax,
        x_minus_pad=0.75,
        x_plus_pad=1.25,
        y_pad=0.75,
        z_pad=0.75,
        lref=lref,
    )
    near_xyz = sample_box_lhs(near_lb, near_ub, N_COLLO_NEAR)

    wake_lb = np.array(
        [
            max(xmin, bmax[0] - 0.10 * lref),
            max(ymin, center[1] - 1.00 * lref),
            max(zmin, center[2] - 1.00 * lref),
        ],
        dtype=float,
    )
    wake_ub = np.array(
        [
            min(xmax, bmax[0] + 3.50 * lref),
            min(ymax, center[1] + 1.00 * lref),
            min(zmax, center[2] + 1.00 * lref),
        ],
        dtype=float,
    )
    wake_xyz = sample_box_lhs(wake_lb, wake_ub, N_COLLO_WAKE)

    XYZ_c = np.vstack([bulk_xyz, near_xyz, wake_xyz])
    XYZ_c = remove_points_inside_mesh(XYZ_c, blade_mesh)
    XYZ_c = np.vstack([XYZ_c, BLADE, OUTLET, INLET[:, 0:3], FARFIELD[:, 0:3]])

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Inlet velocity (m/s): U={U_INF:.6g}, V={V_INF:.6g}, W={W_INF:.6g}")
    print(f"Fluid properties: rho={RHO:.6g} kg/m^3, mu={MU:.6g} Pa*s")
    print("Domain:", domain)
    print("Plot domain:", plot_domain)
    print("Collocation cloud shape:", XYZ_c.shape)
    plot_training_points_steady(XYZ_c, BLADE, TRAINING_POINTS_FIG, display_domain=plot_domain)

    model = PINNBladeSteady3D(
        XYZ_c,
        INLET,
        FARFIELD,
        OUTLET,
        BLADE,
        UV_LAYERS,
        lb,
        ub,
        rho=RHO,
        mu=MU,
    )

    start_time = time.time()
    model.train(iters=ADAM_ITERS, learning_rate=LEARNING_RATE)
    model.train_lbfgs(max_iter=LBFGS_MAX_ITER)
    elapsed = time.time() - start_time
    print(f"--- {elapsed:.2f} seconds ---")

    model.save_NN(CHECKPOINT)
    with open(LOSS_FILE, "wb") as f:
        pickle.dump(model.loss_rec, f)
    plot_loss_history(model.loss_rec, LOSS_FIG)

    x0, y0, z0 = orthogonal_slice_positions(blade_mesh)
    x_slice = float(x0 if SLICE_X is None else SLICE_X)
    y_slice = float(y0 if SLICE_Y is None else SLICE_Y)
    z_slice = float(z0 if SLICE_Z is None else SLICE_Z)
    print(f"Slice positions: SLICE_X={x_slice:.6g}, SLICE_Y={y_slice:.6g}, SLICE_Z={z_slice:.6g}")
    plot_orthogonal_slices_steady(
        model,
        blade_mesh,
        plot_domain,
        x_slice=x_slice,
        y_slice=y_slice,
        z_slice=z_slice,
        outfile=ORTHOGONAL_SLICE_FIG,
    )
    final_loss = model.total_loss()
    print(
        "Final losses -> total: {:.3e}, phys: {:.3e}, blade: {:.3e}, inlet: {:.3e}, far: {:.3e}, outlet: {:.3e}".format(
            final_loss[0].item(),
            final_loss[1].item(),
            final_loss[2].item(),
            final_loss[3].item(),
            final_loss[4].item(),
            final_loss[5].item(),
        )
    )
    print(f"Saved checkpoint to: {CHECKPOINT}")
    print(f"Saved loss history to: {LOSS_FILE}")
    print(f"Saved loss figure to: {LOSS_FIG}")
    print(f"Saved training-point figure to: {TRAINING_POINTS_FIG}")
    print(f"Saved orthogonal slice figure to: {ORTHOGONAL_SLICE_FIG}")
