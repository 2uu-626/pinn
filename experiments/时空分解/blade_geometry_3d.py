from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    import trimesh
except ImportError as e:
    raise ImportError(
        "This helper requires trimesh. Install it with: pip install trimesh rtree"
    ) from e


SURFACE_EXTENSIONS = {
    ".stl",
    ".obj",
    ".ply",
    ".off",
    ".glb",
    ".gltf",
}
MESHIO_EXTENSIONS = {
    ".msh",
    ".cdb",
    ".inp",
    ".nas",
    ".bdf",
    ".vtk",
    ".vtu",
}


def _resolve_mesh_filepath(filepath: str) -> str:
    filepath = str(filepath)
    if os.path.exists(filepath):
        return filepath

    requested = Path(filepath)
    search_dir = requested.parent if str(requested.parent) not in ("", ".") else Path(".")
    if not search_dir.exists():
        raise FileNotFoundError(f"Blade mesh file not found: {filepath}")

    stl_matches = sorted(search_dir.glob("*.stl")) + sorted(search_dir.glob("*.STL"))
    stl_matches = [p for i, p in enumerate(stl_matches) if p not in stl_matches[:i]]

    if len(stl_matches) == 1:
        resolved = str(stl_matches[0])
        print(f"Blade mesh file not found: {filepath}. Auto-using STL: {resolved}")
        return resolved

    if len(stl_matches) > 1:
        choices = ", ".join(str(p.name) for p in stl_matches)
        raise FileNotFoundError(
            f"Blade mesh file not found: {filepath}. Multiple STL files were found: {choices}. "
            "Please set BLADE_MESH_FILE explicitly."
        )

    raise FileNotFoundError(f"Blade mesh file not found: {filepath}")


def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _triangulate_quads(quads: np.ndarray) -> np.ndarray:
    quads = np.asarray(quads, dtype=int)
    if quads.size == 0:
        return np.empty((0, 3), dtype=int)
    t1 = quads[:, [0, 1, 2]]
    t2 = quads[:, [0, 2, 3]]
    return np.vstack([t1, t2])


def _add_faces_to_counter(face_counter: dict, faces: np.ndarray) -> None:
    for face in np.asarray(faces, dtype=int):
        key = tuple(sorted(int(v) for v in face))
        if key in face_counter:
            face_counter[key]["count"] += 1
        else:
            face_counter[key] = {"face": np.asarray(face, dtype=int), "count": 1}


VOLUME_FACE_PATTERNS = {
    "tetra": [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ],
    "hexahedron": [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ],
    "wedge": [
        (0, 1, 2),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ],
    "pyramid": [
        (0, 1, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ],
}


def _surface_mesh_from_meshio(filepath: str, extract_boundary_from_volume: bool = False) -> "trimesh.Trimesh":
    try:
        import meshio
    except ImportError as e:
        raise ImportError(
            "Reading .msh/.cdb/.inp/.nas files requires meshio. Install it with: pip install meshio"
        ) from e

    mesh = meshio.read(filepath)
    points = np.asarray(mesh.points[:, :3], dtype=float)

    triangles: List[np.ndarray] = []
    surface_found = False
    face_counter = {}

    for block in mesh.cells:
        cell_type = str(block.type)
        data = np.asarray(block.data, dtype=int)

        if cell_type == "triangle":
            triangles.append(data)
            surface_found = True
        elif cell_type == "quad":
            triangles.append(_triangulate_quads(data))
            surface_found = True
        elif extract_boundary_from_volume and cell_type in VOLUME_FACE_PATTERNS:
            patterns = VOLUME_FACE_PATTERNS[cell_type]
            for pattern in patterns:
                _add_faces_to_counter(face_counter, data[:, pattern])

    if surface_found:
        faces = np.vstack([tri for tri in triangles if tri.size > 0])
        return trimesh.Trimesh(vertices=points, faces=faces, process=False)

    if not extract_boundary_from_volume:
        raise ValueError(
            "No surface triangles/quads found in the mesh file. Export the blade surface only, or set "
            "extract_boundary_from_volume=True for a blade-only solid mesh. Do not pass a full fluid-domain volume mesh."
        )

    boundary_triangles: List[np.ndarray] = []
    for item in face_counter.values():
        if item["count"] != 1:
            continue
        face = item["face"]
        if len(face) == 3:
            boundary_triangles.append(face[None, :])
        elif len(face) == 4:
            boundary_triangles.append(_triangulate_quads(face[None, :]))

    if not boundary_triangles:
        raise ValueError(
            "Failed to extract any boundary faces from the volume mesh. Export the blade wall surface as STL/OBJ/PLY instead."
        )

    faces = np.vstack(boundary_triangles)
    return trimesh.Trimesh(vertices=points, faces=faces, process=False)


def _coerce_to_trimesh(mesh_like) -> "trimesh.Trimesh":
    if isinstance(mesh_like, trimesh.Scene):
        geometries = [g for g in mesh_like.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise TypeError("Loaded scene does not contain any trimesh.Trimesh geometry")
        mesh_like = trimesh.util.concatenate(tuple(geometries))

    if not isinstance(mesh_like, trimesh.Trimesh):
        raise TypeError("Loaded geometry is not a trimesh.Trimesh")
    return mesh_like


def _repair_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    mesh = mesh.copy()
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass
    return mesh


def load_blade_mesh(
    filepath: str,
    scale: float = 1.0,
    translation: Iterable[float] = (0.0, 0.0, 0.0),
    rotation_deg: Iterable[float] = (0.0, 0.0, 0.0),
    repair: bool = True,
    require_watertight: bool = True,
    extract_boundary_from_volume: bool = False,
) -> "trimesh.Trimesh":
    """Load a blade surface mesh.

    Preferred inputs are watertight surface meshes such as STL/OBJ/PLY.
    If the input is an Ansys or generic mesh file (.msh/.cdb/.inp/.nas/.vtk/.vtu),
    install meshio and export the blade wall surface only. Boundary extraction from
    a blade-only solid mesh can be enabled with extract_boundary_from_volume=True.
    """
    filepath = _resolve_mesh_filepath(filepath)

    ext = Path(filepath).suffix.lower()
    if ext in SURFACE_EXTENSIONS:
        loaded = trimesh.load_mesh(filepath, process=False)
        mesh = _coerce_to_trimesh(loaded)
    elif ext in MESHIO_EXTENSIONS:
        mesh = _surface_mesh_from_meshio(filepath, extract_boundary_from_volume=extract_boundary_from_volume)
    else:
        loaded = trimesh.load_mesh(filepath, process=False)
        mesh = _coerce_to_trimesh(loaded)

    if repair:
        mesh = _repair_mesh(mesh)
    else:
        mesh = mesh.copy()

    mesh.apply_scale(float(scale))

    rot = _rotation_matrix_xyz(*rotation_deg)
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = np.asarray(list(translation), dtype=float)
    mesh.apply_transform(tf)

    if require_watertight and not mesh.is_watertight:
        raise ValueError(
            "The loaded blade surface is not watertight. Export a closed blade surface (recommended: STL) or repair the mesh first."
        )

    return mesh


def sample_surface_points(mesh: "trimesh.Trimesh", n_points: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface_even(mesh, int(n_points))
    return np.asarray(pts, dtype=float)



def points_inside_mesh(points: np.ndarray, mesh: "trimesh.Trimesh") -> np.ndarray:
    points = np.asarray(points, dtype=float)
    try:
        return np.asarray(mesh.contains(points[:, :3]), dtype=bool)
    except ModuleNotFoundError as e:
        raise ImportError(
            "Point-in-mesh queries require rtree. Install it with: pip install rtree"
        ) from e



def remove_points_inside_mesh(points: np.ndarray, mesh: "trimesh.Trimesh") -> np.ndarray:
    points = np.asarray(points, dtype=float)
    inside = points_inside_mesh(points, mesh)
    return points[~inside, :]



def extrude_surface_in_time(surface_xyz: np.ndarray, tmin: float, tmax: float, num_t: int) -> np.ndarray:
    surface_xyz = np.asarray(surface_xyz, dtype=float)
    t = np.linspace(float(tmin), float(tmax), int(num_t))
    tiled_xyz = np.tile(surface_xyz, (len(t), 1))
    tiled_t = np.repeat(t, len(surface_xyz))[:, None]
    return np.hstack([tiled_xyz, tiled_t])



def mesh_bounds(mesh: "trimesh.Trimesh") -> Tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    return bounds[0].copy(), bounds[1].copy()



def mesh_center(mesh: "trimesh.Trimesh") -> np.ndarray:
    bmin, bmax = mesh_bounds(mesh)
    return 0.5 * (bmin + bmax)



def blade_reference_point(mesh: "trimesh.Trimesh") -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=float)
    idx = np.argmin(verts[:, 0])
    return verts[idx].copy()



def reference_length(mesh: "trimesh.Trimesh", axis: str = "x") -> float:
    bmin, bmax = mesh_bounds(mesh)
    extents = bmax - bmin
    axis = axis.lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError(f"Unknown axis: {axis}")
    return float(max(extents[axis_map[axis]], 1e-8))



def auto_domain_from_mesh(
    mesh: "trimesh.Trimesh",
    upstream_lengths: float = 2.0,
    downstream_lengths: float = 5.0,
    y_padding_lengths: float = 2.0,
    z_padding_lengths: float = 2.0,
    reference_axis: str = "x",
) -> dict:
    bmin, bmax = mesh_bounds(mesh)
    center = 0.5 * (bmin + bmax)
    lref = reference_length(mesh, axis=reference_axis)

    half_y = max(0.5 * (bmax[1] - bmin[1]) + y_padding_lengths * lref, 0.75 * lref)
    half_z = max(0.5 * (bmax[2] - bmin[2]) + z_padding_lengths * lref, 0.75 * lref)

    domain = {
        "xmin": float(bmin[0] - upstream_lengths * lref),
        "xmax": float(bmax[0] + downstream_lengths * lref),
        "ymin": float(center[1] - half_y),
        "ymax": float(center[1] + half_y),
        "zmin": float(center[2] - half_z),
        "zmax": float(center[2] + half_z),
    }
    return domain



def orthogonal_slice_positions(mesh: "trimesh.Trimesh") -> Tuple[float, float, float]:
    center = mesh_center(mesh)
    return float(center[0]), float(center[1]), float(center[2])



def mesh_plane_section_polylines(mesh: "trimesh.Trimesh", plane: str, value: float) -> List[np.ndarray]:
    plane = plane.lower()
    if plane == "xy":
        origin = [0.0, 0.0, float(value)]
        normal = [0.0, 0.0, 1.0]
        keep = (0, 1)
    elif plane == "xz":
        origin = [0.0, float(value), 0.0]
        normal = [0.0, 1.0, 0.0]
        keep = (0, 2)
    elif plane == "yz":
        origin = [float(value), 0.0, 0.0]
        normal = [1.0, 0.0, 0.0]
        keep = (1, 2)
    else:
        raise ValueError(f"Unknown plane: {plane}")

    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return []

    polylines: List[np.ndarray] = []
    for poly in getattr(section, "discrete", []):
        arr = np.asarray(poly, dtype=float)
        if arr.size == 0:
            continue
        polylines.append(arr[:, keep])
    return polylines
