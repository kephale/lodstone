"""Reproducible, bounded IDR0062 image/label crop and derived surface fixture."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

BASE_URL = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
ATTRIBUTION = {
    "study": "IDR0062 / Nessys, image 6001240",
    "authors": "Blin G, Sadurska D, Portero Migueles R, Chen N, Watson JA, Lowell S",
    "citation": "Blin et al. (2019), PLOS Biology 17(8):e3000388",
    "doi": "https://doi.org/10.1371/journal.pbio.3000388",
    "catalog": "https://idr.github.io/ome-ngff-samples/",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "source": BASE_URL,
    "interpretation": "Meshes derived from supplied instance labels, not new biological ground truth.",
}
CROP = ((48, 176), (32, 224), (32, 224))  # ZYX, level-0 voxel centers
INPUT_DIGESTS = {
    "image": "2099c0fd60aa22bc7423e99ceb6c6bed2854ffd68eecf0b4977bdf3e553c18db",
    "labels": "9571bd5abcf65ba1044532559ebe8132787e05558254a93757d46fa5a2ce264a",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=40) as response:
                return response.read()
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _scale(attributes: dict[str, Any]) -> np.ndarray:
    multiscale = attributes["multiscales"][0]
    if [axis["name"] for axis in multiscale["axes"]] != ["c", "z", "y", "x"]:
        raise ValueError("expected CZYX source axes")
    if any(axis.get("unit") != "micrometer" for axis in multiscale["axes"][1:]):
        raise ValueError("expected micrometer source coordinates")
    transformations = multiscale["datasets"][0]["coordinateTransformations"]
    if len(transformations) != 1 or transformations[0]["type"] != "scale":
        raise ValueError("fixture requires the declared scale-only source transform")
    return np.asarray(transformations[0]["scale"], dtype=float)[1:]


def download_crop(cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fetch only intersecting native chunks; a missing network chunk is a fault."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "input-manifest.json"
    inputs = cache / "inputs.npz"
    if inputs.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if sha256(inputs.read_bytes()) != manifest["inputs_sha256"]:
            raise ValueError("cached biological inputs failed checksum verification")
        with np.load(inputs) as data:
            _validate_inputs(data["image"], data["labels"])
            return data["image"], data["labels"], data["spacing_zyx"], manifest
    paths = [".zattrs", "labels/0/.zattrs", "0/.zarray", "labels/0/0/.zarray"]
    raw_meta = {path: _fetch(f"{BASE_URL}/{path}") for path in paths}
    image_attrs, label_attrs = (json.loads(raw_meta[path]) for path in paths[:2])
    spacing = _scale(image_attrs)
    np.testing.assert_allclose(spacing, _scale(label_attrs), rtol=0, atol=0)
    chunks = []
    for source_path, name, channels in [
        ("0", "image", (0, 1)),
        ("labels/0/0", "labels", (0,)),
    ]:
        directory = cache / "native" / name
        directory.mkdir(parents=True, exist_ok=True)
        encoded = raw_meta[f"{source_path}/.zarray"]
        metadata = json.loads(encoded)
        if metadata["shape"][1:] != [236, 275, 271]:
            raise ValueError(
                "source dimensions changed; inspect provenance before continuing"
            )
        (directory / ".zarray").write_bytes(encoded)
        grid = [
            range(start // size, (stop - 1) // size + 1)
            for (start, stop), size in zip(CROP, metadata["chunks"][1:], strict=True)
        ]
        separator = metadata.get("dimension_separator", ".")
        for channel, *indices in product(channels, *grid):
            key = separator.join(str(i) for i in (channel, *indices))
            chunks.append((f"{BASE_URL}/{source_path}/{key}", directory / key))

    def download(item):
        url, destination = item
        payload = destination.read_bytes() if destination.exists() else _fetch(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {"url": url, "sha256": sha256(payload), "bytes": len(payload)}

    print(
        f"Fetching {len(chunks)} native image/label chunks from IDR; cached on subsequent launches.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        evidence = list(executor.map(download, chunks))
    zarr = import_module("zarr")
    slices = tuple(slice(start, stop) for start, stop in CROP)
    image = zarr.open_array(str(cache / "native/image"), mode="r")[
        (slice(0, 2), *slices)
    ]
    labels = zarr.open_array(str(cache / "native/labels"), mode="r")[(0, *slices)]
    _validate_inputs(image, labels)
    np.savez_compressed(inputs, image=image, labels=labels, spacing_zyx=spacing)
    manifest = {
        **ATTRIBUTION,
        "crop_zyx": CROP,
        "spacing_zyx_um": spacing.tolist(),
        "origin_xyz_um": (np.asarray([a for a, _ in CROP]) * spacing)[::-1].tolist(),
        "input_metadata": {
            path: {"sha256": sha256(data), "value": json.loads(data)}
            for path, data in raw_meta.items()
        },
        "chunks": evidence,
        "download_payload_bytes": sum(c["bytes"] for c in evidence),
        "inputs_sha256": sha256(inputs.read_bytes()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return image, labels, spacing, manifest


def _validate_inputs(image: np.ndarray, labels: np.ndarray) -> None:
    for name, data in [("image", image), ("labels", labels)]:
        if sha256(data.tobytes()) != INPUT_DIGESTS[name]:
            raise ValueError(
                f"IDR {name} crop checksum changed; inspect the source before proceeding"
            )


def label_surface(
    mask: np.ndarray, spacing: np.ndarray, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a binary ZYX label to XYZ coordinates at voxel centers."""
    measure = import_module("skimage.measure")
    voxel_vertices, faces, _, _ = measure.marching_cubes(
        np.pad(mask, 1),
        level=0.5,
        allow_degenerate=False,
    )
    vertices = ((voxel_vertices.astype(float) - 1) * spacing + origin)[:, ::-1].copy()
    return vertices, faces[:, [0, 2, 1]].astype(np.int64)


def topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    """Referenced vertices, components, Euler characteristic and bad edges."""
    sparse = import_module("scipy.sparse")
    csgraph = import_module("scipy.sparse.csgraph")
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    graph = sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    components = csgraph.connected_components(graph, directed=False)[0]
    return {
        "vertices": len(vertices),
        "faces": len(faces),
        "components": int(components),
        "euler": int(len(vertices) - len(unique) + len(faces)),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
    }


def surface_error_bound(fine_v, fine_f, coarse_v, coarse_f) -> dict[str, float]:
    """Conservative continuous-surface bound using vertex covers, not an RMS claim.

    Every triangle point is within its longest edge of a vertex. Add the
    largest nearest-vertex distance to the other surface, in both directions.
    This is deliberately looser than the sampled vertex distance alone.
    """
    tree = import_module("scipy.spatial").cKDTree
    a = float(tree(coarse_v).query(fine_v)[0].max())
    b = float(tree(fine_v).query(coarse_v)[0].max())

    def edge_length(vertices, faces):
        triangles = vertices[faces]
        return max(
            float(np.linalg.norm(triangles[:, i] - triangles[:, j], axis=1).max())
            for i, j in [(0, 1), (1, 2), (2, 0)]
        )

    return {
        "sampled_vertex_distance_max_um": max(a, b),
        "surface_distance_upper_bound_um": max(
            a + edge_length(fine_v, fine_f), b + edge_length(coarse_v, coarse_f)
        ),
    }


def prepare_biological_mesh(cache: Path, *, max_objects: int = 6) -> dict:
    """Build two surface levels and retain the original fine geometry."""
    output = cache / f"surfaces-v3-{max_objects}"
    report_path = output / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        download_crop(cache)  # verify the paired inputs on cached launches too
        if (
            sha256(Path(report["reference_path"]).read_bytes())
            != report["reference_sha256"]
        ):
            raise ValueError("cached reference geometry failed checksum verification")
        for relative, digest in report["mesh_files_sha256"].items():
            if sha256((Path(report["mesh_path"]) / relative).read_bytes()) != digest:
                raise ValueError(
                    f"cached mesh file failed checksum verification: {relative}"
                )
        return report
    _image, labels, spacing, manifest = download_crop(cache)
    simplifier = import_module("fast_simplification")
    building = import_module("zarr_vectors.building")
    ndimage = import_module("scipy.ndimage")
    boundary = np.unique(
        np.concatenate(
            [
                labels[0].ravel(),
                labels[-1].ravel(),
                labels[:, 0].ravel(),
                labels[:, -1].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )
    values, counts = np.unique(labels, return_counts=True)
    eligible = sorted(
        (
            (int(n), int(i))
            for i, n in zip(values, counts, strict=True)
            if i > 0 and i not in boundary and n >= 100
        ),
        reverse=True,
    )
    chosen = [i for _, i in eligible[:max_objects]]
    if not chosen:
        raise ValueError("crop contains no complete labeled objects")
    output.mkdir(parents=True, exist_ok=True)
    offset_zyx = np.asarray([a for a, _ in CROP]) * spacing
    all_levels: list[list[tuple]] = [[], []]
    objects = []
    reference = {}
    for object_id in chosen:
        mask = labels == object_id
        v, f = label_surface(mask, spacing, offset_zyx)
        original_topology = topology(v, f)
        if (
            original_topology["boundary_edges"]
            or original_topology["nonmanifold_edges"]
        ):
            raise ValueError(
                f"reference label {object_id} does not yield a closed manifold"
            )
        sample_coords = ((v[:, ::-1] - offset_zyx) / spacing).T
        residual = float(
            np.abs(
                ndimage.map_coordinates(mask.astype(float), sample_coords, order=1)
                - 0.5
            ).max()
        )
        # Lewiner's ambiguous-cell interior vertices need not lie exactly on
        # the trilinear .5 contour. That residual is not an alignment oracle.
        # Compare the XYZ inverse against the known ZYX source bounds. The
        # affine itself is independently exercised on an asymmetric fixture.
        recovered = sample_coords.T * spacing + offset_zyx
        alignment_error = float(np.abs(recovered - v[:, ::-1]).max() / spacing.min())
        if alignment_error > 1e-10:
            raise ValueError("mesh/image coordinate round trip failed")
        for reduction in (0.7, 0.4, 0.15):
            # Keep the retained source geometry independent of simplifier buffers.
            cv, cf = simplifier.simplify(v.copy(), f.copy(), target_reduction=reduction)
            coarse_topology = topology(cv, cf)
            if all(
                coarse_topology[k] == original_topology[k]
                for k in ("components", "euler", "boundary_edges", "nonmanifold_edges")
            ):
                break
        else:
            raise ValueError(
                f"no topology-preserving coarse candidate for label {object_id}"
            )
        error = surface_error_bound(v, f, cv, cf)
        fine_quantization = float(
            np.linalg.norm(v - v.astype(np.float32), axis=1).max()
        )
        coarse_quantization = float(
            np.linalg.norm(cv - cv.astype(np.float32), axis=1).max()
        )
        reference[f"vertices_{object_id}"] = v
        reference[f"faces_{object_id}"] = f
        all_levels[0].append((v, f, object_id))
        all_levels[1].append((cv, cf, object_id))
        objects.append(
            {
                "label_id": object_id,
                "fine": original_topology,
                "coarse": coarse_topology,
                "target_reduction": reduction,
                "label_isosurface_residual": residual,
                "coordinate_round_trip_error_voxels": alignment_error,
                "fine_float32_quantization_um": fine_quantization,
                "coarse_float32_quantization_um": coarse_quantization,
                **error,
            }
        )
    np.savez_compressed(output / "reference.npz", **reference)
    root = output / "mesh.zarrvectors"
    for index, blocks in enumerate(all_levels):
        vertices, faces, ids = [], [], []
        count = 0
        for v, f, object_id in blocks:
            vertices.append(v)
            faces.append(f + count)
            ids.append(np.full(len(v), object_id, dtype=np.int64))
            count += len(v)
        destination = root if index == 0 else output / "coarse.zarrvectors"
        building.write_mesh(
            destination,
            np.concatenate(vertices).astype(np.float32),
            np.concatenate(faces),
            object_ids=np.concatenate(ids),
            chunk_shape=(8.0, 8.0, 8.0),
        )
        if index:
            shutil.copytree(destination / "0", root / str(index), dirs_exist_ok=True)
            node = root / str(index) / "zarr.json"
            metadata = json.loads(node.read_text())
            metadata["attributes"]["zarr_vectors_level"]["level"] = index
            node.write_text(json.dumps(metadata))
    node = root / "zarr.json"
    metadata = json.loads(node.read_text())
    multiscales = metadata["attributes"]["multiscales"][0]
    second = dict(multiscales["datasets"][0])
    second["path"] = "1"
    multiscales["datasets"] = [multiscales["datasets"][0], second]
    node.write_text(json.dumps(metadata))
    # Measured relative to the retained fine reference, not the unknown tissue surface.
    report = {
        **ATTRIBUTION,
        "mesh_path": str(root),
        "inputs_path": str(cache / "inputs.npz"),
        "reference_path": str(output / "reference.npz"),
        "objects": objects,
        "crop_zyx": CROP,
        "spacing_zyx_um": spacing.tolist(),
        "origin_xyz_um": offset_zyx[::-1].tolist(),
        "level_errors_um": [
            max(o["fine_float32_quantization_um"] for o in objects),
            max(
                o["surface_distance_upper_bound_um"]
                + o["coarse_float32_quantization_um"]
                for o in objects
            ),
        ],
        "inputs_sha256": manifest["inputs_sha256"],
        "reference_sha256": sha256((output / "reference.npz").read_bytes()),
        "mesh_files_sha256": {
            str(path.relative_to(root)): sha256(path.read_bytes())
            for path in sorted(root.rglob("*"))
            if path.is_file()
        },
        "notes": "Complete objects only; labels touching crop edges excluded. Derived surfaces retain source label IDs. Coarse candidates checked for components, Euler characteristic and edge incidence. This does not independently validate segmentation biology.",
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Prepared {len(objects)} labeled nuclei; report: {report_path}", flush=True)
    return report


def verify_reference(report: dict) -> dict:
    """Compare the multiset of oriented streamed triangles to the retained mesh."""
    building = import_module("zarr_vectors.building")

    def keys(vertices, faces):
        triangles = np.asarray(vertices, dtype=np.float32)[faces]
        return [
            min(np.roll(t, shift, axis=0).tobytes() for shift in range(3))
            for t in triangles
        ]

    expected = []
    with np.load(report["inputs_path"]) as inputs:
        labels = inputs["labels"]
    spacing = np.asarray(report["spacing_zyx_um"])
    origin = np.asarray(report["origin_xyz_um"])[::-1]
    with np.load(report["reference_path"]) as original:
        for item in report["objects"]:
            object_id = item["label_id"]
            regenerated_v, regenerated_f = label_surface(
                labels == object_id, spacing, origin
            )
            np.testing.assert_array_equal(
                original[f"vertices_{object_id}"], regenerated_v
            )
            np.testing.assert_array_equal(original[f"faces_{object_id}"], regenerated_f)
            expected.extend(
                keys(original[f"vertices_{object_id}"], original[f"faces_{object_id}"])
            )
    actual = []
    with building.MeshReader(report["mesh_path"], level=0) as reader:
        owners = len(reader.chunks)
        for chunk in reader.chunks:
            result = reader.read([chunk.key], max_bytes=64 << 20)
            actual.extend(keys(result["vertices"], result["faces"]))
    expected, actual = sorted(expected), sorted(actual)
    if actual != expected:
        raise ValueError("streamed fine geometry differs from the retained reference")
    return {
        "reference_faces": len(expected),
        "streamed_faces": len(actual),
        "owner_cells": owners,
        "oriented_triangle_sha256": sha256(b"".join(actual)),
        "exact_after_declared_float32_quantization": True,
        "retained_reference_matches_label_extraction": True,
    }
