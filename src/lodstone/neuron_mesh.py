"""Pinned biological neuron surfaces with bounded, auditable smoothing."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np

from .biological_mesh import (
    _fetch,
    sha256,
    topology,
    write_surface_levels,
)


def smooth_surface(
    vertices,
    faces,
    *,
    iterations=20,
    max_displacement_um=0.256,
    max_relative_volume_change=0.05,
):
    """Taubin pairs in physical XYZ; clamp displacement relative to raw geometry.

    Connectivity is unchanged. The maximum corresponding-vertex movement also
    bounds the continuous surface distance because triangles keep their indices.
    Back off per connected component until signed-volume change is within the
    declared tolerance and no face is degenerate or reverses relative to raw.
    These checks do not establish absence of self-intersection.
    """
    if iterations < 0 or not isinstance(iterations, int):
        raise ValueError("iterations must be a nonnegative integer")
    if not np.isfinite(max_displacement_um) or max_displacement_um < 0:
        raise ValueError("maximum displacement must be finite and nonnegative")
    if not 0 <= max_relative_volume_change < 1:
        raise ValueError("relative volume tolerance must be in [0, 1)")
    sparse = import_module("scipy.sparse")
    raw = np.asarray(vertices, dtype=float)
    v = raw.copy()
    edges = np.unique(
        np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )
    rows, cols = np.concatenate([edges, edges[:, ::-1]]).T
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(v), len(v))
    )
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1, 1)
    for _ in range(iterations):
        for factor in (0.5, -0.53):
            mean = np.divide(adjacency @ v, degree, out=v.copy(), where=degree > 0)
            v += factor * (mean - v)
        delta = v - raw
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        v = raw + delta * np.minimum(
            1.0, max_displacement_um / np.maximum(length, 1e-30)
        )
    component_count, components = import_module(
        "scipy.sparse.csgraph"
    ).connected_components(adjacency)
    backoffs = 0
    max_component_change = 0.0
    for component in range(component_count):
        vertex_mask = components == component
        component_faces = faces[components[faces[:, 0]] == component]
        # Center for stable signed-volume arithmetic in physical world space.
        center = raw[vertex_mask].mean(axis=0)

        def normals_and_volume(points, center=center, component_faces=component_faces):
            t = (points - center)[component_faces]
            normals = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            volume = float(np.einsum("ij,ij->i", t[:, 0], normals).sum() / 6)
            return normals, volume

        raw_normals, raw_volume = normals_and_volume(raw)
        for attempt in range(21):
            normals, volume = normals_and_volume(v)
            change = abs(volume / raw_volume - 1) if abs(raw_volume) > 1e-20 else 0.0
            good = np.all(np.einsum("ij,ij->i", raw_normals, normals) > 0)
            if good and change <= max_relative_volume_change + 1e-12:
                max_component_change = max(max_component_change, change)
                break
            v[vertex_mask] = raw[vertex_mask] + 0.5 * (
                v[vertex_mask] - raw[vertex_mask]
            )
            backoffs += 1
        else:
            v[vertex_mask] = raw[vertex_mask]
    movement = np.linalg.norm(v - raw, axis=1)

    def signed_volume(points):
        t = (points - points.mean(axis=0))[faces]
        return float(
            np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
        )

    raw_volume, smooth_volume = signed_volume(raw), signed_volume(v)
    return v, {
        "max_vertex_displacement_um": float(movement.max(initial=0)),
        "rms_vertex_displacement_um": float(np.sqrt(np.mean(movement**2))),
        "raw_signed_volume_um3": raw_volume,
        "smooth_signed_volume_um3": smooth_volume,
        "relative_volume_change": (smooth_volume / raw_volume - 1)
        if raw_volume
        else None,
        "connectivity_unchanged": True,
        "component_volume_backoffs": backoffs,
        "max_component_relative_volume_change": max_component_change,
    }


SOURCE_REVISION = "4fe56c87fb8c2b0c9f89ceb3a4beb6dbf8f30839"
SOURCE_BASE = (
    f"https://raw.githubusercontent.com/navis-org/navis/{SOURCE_REVISION}/navis/data"
)
SOURCE_DIGESTS = {
    "1734350788.ply": "569b413ba6ea84521e05eeae430d3372c5c071c19f843c7d8a63a193fdd6ba7f",
    "1734350908.ply": "d0899c24f37dd535792ae38e50b357120ad0fd3d63985f55491f2779fa56245c",
    "722817260.ply": "017e6c67d94b6b9ee2521353c2bec493d2db492df68eb648510f09dcc413519a",
    "754534424.ply": "d5cb4d58027e534bde30c7f13f007cf070e06ec1e646edceb5b30af4bbec78d8",
    "754538881.ply": "74f03ce718e6f255218c3857cbc39fc5e513f9f6db5c3836e2a6d81751d78864",
    "meta.json": "f3f6f7eaee96195fe9c3cf5e771e2eb90ac0880a7338290d072470237d456744",
    "readme.md": "265df23b0fc0f56b4bf4116b942fb6ba8c57e203387134e2fb9694380eb55073",
    "load_data.py": "df3c76d1c95d57be3c60a33cfb27eeceeabd3f362928bfef563ed0756fe203cc",
}
ATTRIBUTION = {
    "study": "Janelia hemibrain / NAVis DA1 olfactory projection-neuron examples",
    "citation": "Scheffer et al. (2020), eLife 9:e57443",
    "doi": "https://doi.org/10.7554/eLife.57443",
    "source": SOURCE_BASE,
    "source_revision": SOURCE_REVISION,
    "provenance": f"{SOURCE_BASE}/readme.md",
    "units_source": f"{SOURCE_BASE}/load_data.py",
    "source_units": "8 nanometers per coordinate unit",
    "coordinate_scale_to_um": 0.008,
    "repository_license": "GPL-3.0 (NAVis repository; downloaded data retains source terms)",
    "interpretation": "Entire five bundled example meshes; not a whole-brain reconstruction or proof of complete biological arbors.",
}


def read_example_ply(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Read only the exact pinned triangular little-endian PLY schema.

    A deliberately narrow fixture parser, not a general PLY importer. No mesh
    processing, welding, face removal or coordinate changes occur here.
    """
    header_bytes, body = payload.split(b"end_header\n", 1)
    header = header_bytes.decode("ascii").splitlines()
    vertex_line = next(line for line in header if line.startswith("element vertex "))
    face_line = next(line for line in header if line.startswith("element face "))
    nv, nf = int(vertex_line.split()[-1]), int(face_line.split()[-1])
    expected = [
        "ply",
        "format binary_little_endian 1.0",
        "comment https://github.com/mikedh/trimesh",
        vertex_line,
        "property float x",
        "property float y",
        "property float z",
        face_line,
        "property list uchar int vertex_indices",
    ]
    if header != expected or nv <= 0 or nf <= 0 or len(body) != nv * 12 + nf * 13:
        raise ValueError("example PLY schema changed")
    vertices = (
        np.frombuffer(body, dtype="<f4", count=nv * 3).reshape(nv, 3).astype(float)
    )
    records = np.frombuffer(
        body, dtype=[("count", "u1"), ("indices", "<i4", (3,))], offset=nv * 12
    )
    faces = records["indices"].astype(np.int64)
    if (
        not np.all(records["count"] == 3)
        or faces.min() < 0
        or faces.max() >= nv
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("invalid triangular PLY geometry")
    return vertices, faces


def load_neuron_meshes(cache: Path) -> tuple[list, dict]:
    """Download pinned files and verify their bytes on every launch."""
    native = cache / "native"
    native.mkdir(parents=True, exist_ok=True)
    evidence = []
    meshes = []
    for filename, digest in SOURCE_DIGESTS.items():
        path = native / filename
        url = f"{SOURCE_BASE}/{'ply/' if filename.endswith('.ply') else ''}{filename}"
        if not path.exists():
            print(f"Fetching NAVis hemibrain example {filename}", flush=True)
            path.write_bytes(_fetch(url))
        payload = path.read_bytes()
        if sha256(payload) != digest:
            raise ValueError(f"source checksum mismatch: {filename}")
        evidence.append(
            {"file": filename, "url": url, "sha256": digest, "bytes": len(payload)}
        )
        if filename.endswith(".ply"):
            vertices, faces = read_example_ply(payload)
            meshes.append((int(path.stem), vertices * 0.008, faces))
    manifest = {
        **ATTRIBUTION,
        "files": evidence,
        "source_payload_bytes": sum(f["bytes"] for f in evidence),
    }
    (cache / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    return meshes, manifest


def prepare_neuron_mesh(cache: Path, *, smooth_iterations: int = 20) -> dict:
    """Prepare all five neuron meshes with no additional spatial clipping."""
    meshes, manifest = load_neuron_meshes(cache)
    output = cache / f"surfaces-v4-smooth{smooth_iterations}"
    report_path = output / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for name in ("reference", "raw_reference", "coarse_reference"):
            if (
                sha256(Path(report[f"{name}_path"]).read_bytes())
                != report[f"{name}_sha256"]
            ):
                raise ValueError(f"cached {name} checksum mismatch")
        for relative, digest in report["mesh_files_sha256"].items():
            if sha256((Path(report["mesh_path"]) / relative).read_bytes()) != digest:
                raise ValueError(f"cached mesh checksum mismatch: {relative}")
        return report
    output.mkdir(parents=True, exist_ok=True)
    from .mesh_simplification import constrained_coarse_mesh

    smoothing = {
        "iterations": smooth_iterations,
        "max_displacement_um": 0.15,
        "max_relative_volume_change": 0.05,
    }
    reference, raw_reference, coarse_reference, objects = {}, {}, {}, []
    levels: list[list] = [[], []]
    for object_id, v, f in meshes:
        raw_reference[f"vertices_{object_id}"], raw_reference[f"faces_{object_id}"] = (
            v,
            f,
        )
        sv, change = smooth_surface(v, f, **smoothing)
        fine = topology(sv, f)
        print(
            f"Preparing constrained coarse neuron {object_id} ({len(objects) + 1}/{len(meshes)})",
            flush=True,
        )
        cv, cf, decimation = constrained_coarse_mesh(
            sv, f, max_error=0.5, target_reduction=0.6
        )
        reduction = decimation["actual_reduction"]
        reference[f"vertices_{object_id}"], reference[f"faces_{object_id}"] = sv, f
        (
            coarse_reference[f"vertices_{object_id}"],
            coarse_reference[f"faces_{object_id}"],
        ) = cv, cf
        levels[0].append((sv, f, object_id))
        levels[1].append((cv, cf, object_id))
        objects.append(
            {
                "label_id": object_id,
                "source_name": "DA1_lPN_R",
                "fine": fine,
                "coarse": topology(cv, cf),
                "target_reduction": reduction,
                "smoothing_change": change,
                "fine_float32_quantization_um": float(
                    np.linalg.norm(sv - sv.astype(np.float32), axis=1).max()
                ),
                "coarse_float32_quantization_um": float(
                    np.linalg.norm(cv - cv.astype(np.float32), axis=1).max()
                ),
                "surface_distance_upper_bound_um": decimation[
                    "surface_distance_upper_bound_um"
                ],
                "coarse_construction": decimation,
            }
        )
    np.savez_compressed(output / "reference.npz", **reference)
    np.savez_compressed(output / "raw-reference.npz", **raw_reference)
    np.savez_compressed(output / "coarse-reference.npz", **coarse_reference)
    has_safe_coarse = any(o["target_reduction"] > 0 for o in objects)
    if not has_safe_coarse:
        levels = levels[:1]
    root = write_surface_levels(output, levels)
    vertices = np.concatenate([v for _, v, _ in meshes])
    report = {
        **ATTRIBUTION,
        "sample": "hemibrain",
        "display_title": "Hemibrain · DA1 projection-neuron surfaces",
        "credit": "Scheffer et al. 2020 · Janelia FlyEM · NAVis example meshes",
        "source_cache": str(cache),
        "source_manifest": manifest,
        "mesh_path": str(root),
        "reference_path": str(output / "reference.npz"),
        "raw_reference_path": str(output / "raw-reference.npz"),
        "coarse_reference_path": str(output / "coarse-reference.npz"),
        "coarse_reference_sha256": sha256(
            (output / "coarse-reference.npz").read_bytes()
        ),
        "reference_sha256": sha256((output / "reference.npz").read_bytes()),
        "raw_reference_sha256": sha256((output / "raw-reference.npz").read_bytes()),
        "objects": objects,
        "smoothing": smoothing,
        "lod_policy": "Constrained regular-edge collapses; source singularity neighborhoods retained exactly, topology checks and source-face-map error certificate",
        "has_safe_coarse": has_safe_coarse,
        "bounds_xyz_um": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "coverage": {
            "source_labels": len(meshes),
            "meshed_labels": len(objects),
            "omitted_labels": [],
            "clipped_labels": [],
            "scope": "All five bundled neuron meshes, not the whole brain",
        },
        "level_errors_um": [
            max(o["fine_float32_quantization_um"] for o in objects),
            max(
                o["surface_distance_upper_bound_um"]
                + o["coarse_float32_quantization_um"]
                for o in objects
            ),
        ][: len(levels)],
        "mesh_files_sha256": {
            str(p.relative_to(root)): sha256(p.read_bytes())
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
        "notes": "All supplied faces retained at fine LOD with original face indices. Coordinates converted from 8 nm units to micrometers. Smoothing is bounded preprocessing before chunking, with per-component volume and local face-orientation guards. This does not establish absence of self-intersection or prove biological thin-process accuracy. Coarse LOD simplifies regular regions while fixing source singularity neighborhoods. A certified source-face map bounds continuous surface displacement by 0.5 micrometers before float32 quantization. Source meshes can already contain nonmanifold edges, disconnected pieces or truncated arbors; their topology diagnostics are retained, not repaired silently. LOD errors are relative to the retained smoothed reference.",
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"Prepared all {len(meshes)} hemibrain neuron meshes; {report_path}", flush=True
    )
    return report
