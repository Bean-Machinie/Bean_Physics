"""Textured mesh visual for UV-mapped meshes."""

from __future__ import annotations

from typing import Any

import numpy as np
from vispy import gloo
from vispy.visuals import Visual
from vispy.scene.visuals import create_visual_node

_VERT_SHADER = """
attribute vec3 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = $transform(vec4(a_position, 1.0));
}
"""

_FRAG_SHADER = """
uniform sampler2D u_texture;
uniform vec4 u_tint;
varying vec2 v_texcoord;

void main() {
    vec4 tex_color = texture2D(u_texture, v_texcoord);
    gl_FragColor = tex_color * u_tint;
}
"""


class TexturedMeshVisual(Visual):
    def __init__(
        self,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        texcoords: np.ndarray | None = None,
        texture: np.ndarray | None = None,
        tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        Visual.__init__(self, vcode=_VERT_SHADER, fcode=_FRAG_SHADER, **kwargs)
        self.set_gl_state("opaque", depth_test=True, blend=False)
        self._vertices = gloo.VertexBuffer(np.zeros((0, 3), dtype=np.float32))
        self._texcoords = gloo.VertexBuffer(np.zeros((0, 2), dtype=np.float32))
        self._texture = gloo.Texture2D(np.zeros((2, 2, 4), dtype=np.uint8))
        self._tint = np.array(tint, dtype=np.float32)
        self._data_changed = True
        self._index_buffer = gloo.IndexBuffer(
            np.zeros((0, 3), dtype=np.uint32)
        )
        self._draw_mode = "triangles"
        if vertices is not None and faces is not None and texcoords is not None:
            self.set_data(vertices, faces, texcoords, texture, tint=tint)
        self.freeze()

    def set_data(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        texcoords: np.ndarray,
        texture: np.ndarray | None,
        tint: tuple[float, float, float, float] | None = None,
        has_alpha: bool = False,
    ) -> None:
        verts = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        uvs = np.asarray(texcoords, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if uvs.ndim != 2 or uvs.shape[1] != 2:
            raise ValueError("texcoords must have shape (N, 2)")
        if uvs.shape[0] != verts.shape[0]:
            raise ValueError("texcoords must match vertices count")
        if texture is None:
            raise ValueError("texture is required for TexturedMeshVisual")
        tex = np.asarray(texture, dtype=np.uint8)
        if tex.ndim != 3 or tex.shape[2] != 4:
            raise ValueError("texture must have shape (H, W, 4)")
        self._vertices.set_data(verts)
        self._texcoords.set_data(uvs)
        self._index_buffer = gloo.IndexBuffer(faces)
        self._texture = gloo.Texture2D(tex, interpolation="linear", wrapping="repeat")
        if tint is not None:
            self._tint = np.array(tint, dtype=np.float32)
        if has_alpha:
            self.set_gl_state("translucent", depth_test=True, blend=True)
        else:
            self.set_gl_state("opaque", depth_test=True, blend=False)
        self._data_changed = True
        self.update()

    def _prepare_draw(self, view: Any) -> None:
        if self._data_changed:
            self.shared_program["a_position"] = self._vertices
            self.shared_program["a_texcoord"] = self._texcoords
            self.shared_program["u_texture"] = self._texture
            self.shared_program["u_tint"] = self._tint
            self._data_changed = False

    @staticmethod
    def _prepare_transforms(view: Any) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _compute_bounds(self, axis: int, view: Any) -> tuple[float, float] | None:
        data = self._vertices.data
        if data is None or data.size == 0:
            return None
        return (float(np.min(data[:, axis])), float(np.max(data[:, axis])))


TexturedMesh = create_visual_node(TexturedMeshVisual)
