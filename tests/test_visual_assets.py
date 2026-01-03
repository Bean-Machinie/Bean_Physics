import numpy as np

from bean_physics.app import visual_assets


class _DummyMaterial:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image


class _DummyVisual:
    def __init__(
        self,
        kind: str,
        vertex_colors: np.ndarray | None = None,
        uv: np.ndarray | None = None,
        image: np.ndarray | None = None,
    ) -> None:
        self.kind = kind
        self.vertex_colors = vertex_colors
        self.uv = uv
        self.material = _DummyMaterial(image) if image is not None else None


class _DummyMesh:
    def __init__(
        self,
        visual: _DummyVisual,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        name: str = "mesh",
    ) -> None:
        self.visual = visual
        self.vertices = (
            vertices
            if vertices is not None
            else np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        )
        self.faces = (
            faces
            if faces is not None
            else np.array([[0, 0, 0]], dtype=np.int32)
        )
        self.name = name


def test_center_vertices_bbox() -> None:
    vertices = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    centered, center = visual_assets._center_vertices(vertices)
    assert np.allclose(center, np.array([2.0, 3.0, 4.0], dtype=np.float32))
    mins = centered.min(axis=0)
    maxs = centered.max(axis=0)
    assert np.allclose((mins + maxs) * 0.5, np.zeros(3, dtype=np.float32))


def test_color_priority_vertex_over_texture() -> None:
    vertex_colors = np.array([[255, 128, 0, 255], [0, 128, 255, 255]], dtype=np.uint8)
    uv = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    visual = _DummyVisual(
        kind="texture",
        vertex_colors=vertex_colors,
        uv=uv,
        image=image,
    )
    mesh = _DummyMesh(
        visual,
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        faces=np.array([[0, 1, 1]], dtype=np.int32),
    )
    chunk, entry = visual_assets._build_chunk(mesh, "dummy")
    expected = visual_assets._vertex_colors(vertex_colors)
    assert entry["chosen_path"] == "using vertex colors"
    assert chunk["vertex_colors"] is not None
    assert np.allclose(chunk["vertex_colors"], expected)


def test_sample_texture_bilinear() -> None:
    image = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    uv = np.array([[0.5, 0.5]], dtype=np.float32)
    colors = visual_assets._sample_texture(image, uv)
    expected = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    assert np.allclose(colors, expected)


def test_prepare_uv_unrolls_faces() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    prepared = visual_assets._prepare_uv(vertices, faces, uv)
    assert prepared is not None
    out_vertices, out_faces, out_uv = prepared
    assert out_vertices.shape[0] == faces.shape[0] * 3
    assert out_faces.shape[0] == faces.shape[0]
    assert out_uv.shape[0] == faces.shape[0] * 3


def test_build_chunks_multi_geometry() -> None:
    visual = _DummyVisual(kind="vertex", vertex_colors=np.array([[255, 0, 0, 255]], dtype=np.uint8))
    mesh_a = _DummyMesh(visual, name="a")
    mesh_b = _DummyMesh(visual, name="b")
    chunks, report = visual_assets._build_chunks([("a", mesh_a), ("b", mesh_b)], "path", "obj", True)
    assert len(chunks) == 2
    assert report["num_geometries"] == 2
