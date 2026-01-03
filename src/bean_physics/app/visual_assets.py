"""Visual-only mesh loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def loader_available() -> bool:
    try:
        import trimesh  # noqa: F401
    except Exception:
        return False
    return True


def load_mesh_data(path: str | Path) -> dict[str, Any]:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("trimesh is required to load mesh assets") from exc

    mesh = trimesh.load(str(path))
    if mesh is None:
        raise RuntimeError("unable to load mesh")

    is_scene = isinstance(mesh, trimesh.Scene)
    if is_scene:
        dumped = mesh.dump(concatenate=False)
        geometries = [(f"geom_{i}", geom) for i, geom in enumerate(dumped)]
    else:
        geometries = [(getattr(mesh, "name", "mesh"), mesh)]

    fmt = Path(path).suffix.lower().lstrip(".")
    chunks, report = _build_chunks(geometries, str(path), fmt, is_scene)
    _print_report(report)

    return {"chunks": chunks, "report": report}


def _build_chunks(
    geometries: list[tuple[str, Any]],
    path: str,
    fmt: str,
    is_scene: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "path": path,
        "format": fmt,
        "is_scene": is_scene,
        "num_geometries": len(geometries),
        "geometries": [],
    }
    for name, mesh in geometries:
        chunk, entry = _build_chunk(mesh, name)
        chunks.append(chunk)
        report["geometries"].append(entry)
    return chunks, report


def _build_chunk(mesh: Any, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    visual = getattr(mesh, "visual", None)
    visual_type = type(visual).__name__ if visual is not None else "None"
    vertex_colors_raw = getattr(visual, "vertex_colors", None) if visual is not None else None
    uv = getattr(visual, "uv", None) if visual is not None else None
    material = getattr(visual, "material", None) if visual is not None else None
    image, image_source = _extract_material_image(material)
    base_color = _extract_base_color(material)
    has_base_color = base_color is not None
    material_type = type(material).__name__ if material is not None else "None"

    warnings: list[str] = []
    chosen = "fallback neutral (no colors/UV/texture)"
    colors = None
    to_color_used = False
    to_color_error: str | None = None

    if vertex_colors_raw is not None:
        colors = _vertex_colors(vertex_colors_raw)
        vertices, faces, colors = _align_vertex_colors(vertices, faces, colors)
        chosen = "using vertex colors"
    elif visual is not None and hasattr(visual, "to_color"):
        try:
            converted = visual.to_color()
            converted_colors = getattr(converted, "vertex_colors", None)
            if converted_colors is not None:
                colors = _vertex_colors(converted_colors)
                vertices, faces, colors = _align_vertex_colors(vertices, faces, colors)
                chosen = "using visual.to_color()"
                to_color_used = True
        except Exception as exc:  # pragma: no cover - diagnostic path
            to_color_error = str(exc)
    elif uv is not None and image is not None:
        prepared = _prepare_uv(vertices, faces, uv)
        if prepared is None:
            warnings.append("UV count mismatch; cannot bake texture.")
        else:
            vertices, faces, uv = prepared
            colors = _bake_texture_colors(uv, image)
            if colors is None:
                warnings.append("Texture bake failed; missing image or UVs.")
            else:
                chosen = "baking texture via UVs"
    elif has_base_color:
        colors = _solid_color(vertices.shape[0], base_color)
        chosen = "using material base color"

    vertices, _ = _center_vertices(vertices)
    stats = _color_stats(colors) if colors is not None else None
    if colors is None:
        warnings.append("No vertex colors or texture/UVs found; rendering untextured.")
        if uv is not None and image is None:
            warnings.append(
                "UVs present but no texture image; check material or install Pillow to decode textures."
            )

    chunk = {
        "name": name,
        "vertices": vertices,
        "faces": faces,
        "vertex_colors": colors,
        "chosen_path": chosen,
    }
    entry = {
        "name": name,
        "n_vertices": int(vertices.shape[0]),
        "n_faces": int(faces.shape[0]),
        "visual_type": visual_type,
        "has_vertex_colors": vertex_colors_raw is not None,
        "has_uv": uv is not None,
        "material_type": material_type,
        "has_material_image": image is not None,
        "image_source": image_source,
        "image_shape": getattr(image, "shape", None),
        "has_base_color": has_base_color,
        "base_color": base_color,
        "chosen_path": chosen,
        "to_color_used": to_color_used,
        "to_color_error": to_color_error,
        "baked_colors_stats": stats,
        "warnings": warnings,
    }
    return chunk, entry


def _vertex_colors(vertex_colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(vertex_colors, dtype=np.float32)
    if colors.shape[1] == 4:
        colors = colors[:, :3]
    colors = colors / 255.0
    alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
    return np.hstack([colors, alpha])


def _bake_texture_colors(uv: np.ndarray, image: Any, v_flip: bool = True) -> np.ndarray | None:
    try:
        img = np.asarray(image)
    except Exception:
        return None
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.float32) / 255.0
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2:
        return None
    colors = _sample_texture(img, uv, v_flip=v_flip)
    if img.shape[2] == 4:
        alpha = colors[:, 3:4]
        rgb = colors[:, :3]
    else:
        rgb = colors[:, :3]
        alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
    return np.hstack([rgb, alpha])


def _sample_texture(image: np.ndarray, uv: np.ndarray, v_flip: bool = True) -> np.ndarray:
    h, w, _ = image.shape
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(1.0 - uv[:, 1], 1.0) if v_flip else np.mod(uv[:, 1], 1.0)
    x = u * (w - 1)
    y = v * (h - 1)
    x0 = np.floor(x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]
    c0 = c00 * (1 - wx) + c10 * wx
    c1 = c01 * (1 - wx) + c11 * wx
    return c0 * (1 - wy) + c1 * wy


def _prepare_uv(
    vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim == 3 and uv.shape[0] == faces.shape[0] and uv.shape[1:] == (3, 2):
        vertices, faces = _unroll_vertices(vertices, faces)
        uv = uv.reshape(-1, 2)
        return vertices, faces, uv
    if uv.ndim != 2 or uv.shape[1] != 2:
        return None
    if uv.shape[0] == vertices.shape[0]:
        return vertices, faces, uv
    if uv.shape[0] == faces.shape[0] * 3:
        vertices, faces = _unroll_vertices(vertices, faces)
        uv = uv.reshape(-1, 2)
        return vertices, faces, uv
    return None


def _align_vertex_colors(
    vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if colors.shape[0] == vertices.shape[0]:
        return vertices, faces, colors
    if colors.shape[0] == faces.shape[0] * 3:
        vertices, faces = _unroll_vertices(vertices, faces)
        colors = colors.reshape(-1, colors.shape[-1])
        return vertices, faces, colors
    return vertices, faces, colors


def _center_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if vertices.size == 0:
        return vertices, np.zeros(3, dtype=np.float32)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    return vertices - center, center


def _unroll_vertices(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = vertices[faces].reshape(-1, 3)
    faces = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 3)
    return vertices, faces


def _extract_material_image(material: Any) -> tuple[Any | None, str | None]:
    if material is None:
        return None, None
    image = getattr(material, "image", None)
    if image is not None:
        return image, "material.image"
    texture = getattr(material, "baseColorTexture", None)
    if texture is not None:
        tex_image = getattr(texture, "image", None)
        if tex_image is not None:
            return tex_image, "material.baseColorTexture.image"
    return None, None


def _extract_base_color(material: Any) -> list[float] | None:
    if material is None:
        return None
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = getattr(material, "diffuse", None)
    if color is None:
        return None
    try:
        vals = [float(v) for v in color]
    except Exception:
        return None
    if len(vals) >= 3:
        return vals[:3]
    return None


def _solid_color(count: int, rgb: list[float]) -> np.ndarray:
    color = np.asarray(rgb, dtype=np.float32)
    if color.shape[0] != 3:
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rgbs = np.tile(color[None, :], (count, 1))
    alpha = np.ones((count, 1), dtype=np.float32)
    return np.hstack([rgbs, alpha])


def _color_stats(colors: np.ndarray) -> dict[str, list[float]]:
    mins = np.min(colors, axis=0).tolist()
    maxs = np.max(colors, axis=0).tolist()
    means = np.mean(colors, axis=0).tolist()
    return {"min": mins, "max": maxs, "mean": means}


def _print_report(report: dict[str, Any]) -> None:
    lines = [
        "Visual asset report:",
        f"  path: {report.get('path')}",
        f"  format: {report.get('format')}",
        f"  scene: {report.get('is_scene')} (geometries={report.get('num_geometries')})",
    ]
    for entry in report.get("geometries", []):
        lines.append(f"  geometry: {entry.get('name')}")
        lines.append(
            "    verts/faces: "
            f"{entry.get('n_vertices')}/{entry.get('n_faces')}"
        )
        lines.append(f"    visual: {entry.get('visual_type')} ({entry.get('material_type')})")
        lines.append(
            "    has_vertex_colors/uv/image: "
            f"{entry.get('has_vertex_colors')}/"
            f"{entry.get('has_uv')}/"
            f"{entry.get('has_material_image')}"
        )
        if entry.get("to_color_used"):
            lines.append("    to_color: used")
        if entry.get("to_color_error"):
            lines.append(f"    to_color_error: {entry.get('to_color_error')}")
        if entry.get("image_source"):
            lines.append(f"    image_source: {entry.get('image_source')}")
        if entry.get("has_base_color"):
            lines.append(f"    base_color: {entry.get('base_color')}")
        if entry.get("image_shape") is not None:
            lines.append(f"    image_shape: {entry.get('image_shape')}")
        lines.append(f"    chosen: {entry.get('chosen_path')}")
        stats = entry.get("baked_colors_stats")
        if stats is not None:
            lines.append(f"    colors min/max: {stats['min']} / {stats['max']}")
        warnings = entry.get("warnings", [])
        for warning in warnings:
            lines.append(f"    warning: {warning}")
    print("\n".join(lines))
