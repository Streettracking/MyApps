"""FlyWire shell and skeleton scene for the 3D window.

``artifacts/fly_brain_view.npz`` is built by ``tools/fetch_flywire_view.py``
from the public 783 skeleton cache and the navis-flybrains ``FLYWIRE.ply``
neuropil mesh. No CAVE token. When the file is missing, the window keeps the
soma point cloud.

Display axes, baked into the file: smaller FlyWire x is anatomical left
(negative x, «глаз Л» / recognized_L), smaller y is dorsal, larger z is
anterior. Optic tags: 0 central, 1 left lobe, 2 right lobe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

KIND_KC_G = 1
KIND_KC_AB = 2
KIND_KC_APB = 3
KIND_KC_OTHER = 4
KIND_MBON = 5
KIND_PAM = 6
KIND_PPL1 = 7

KIND_RGB = {
    KIND_KC_G: (72, 230, 148),
    KIND_KC_AB: (255, 146, 48),
    KIND_KC_APB: (198, 112, 255),
    KIND_KC_OTHER: (96, 176, 255),
    KIND_MBON: (110, 228, 236),
    KIND_PAM: (255, 214, 96),
    KIND_PPL1: (255, 86, 128),
}
_PAM_HOT = (176, 255, 190)
_PPL_HOT = (255, 148, 158)
_INK = (214, 216, 222)
_HINT = (168, 170, 176)

_SCENE = None
_GL = None
_GL_FAILED = False


def view_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "artifacts" / "fly_brain_view.npz"
    return Path(__file__).resolve().parents[1] / "artifacts" / "fly_brain_view.npz"


VIEW_NPZ = view_path()


def _moderngl():
    try:
        import moderngl
    except ImportError:
        return None
    return moderngl


class FlyScene:
    def __init__(self, path: Path):
        z = np.load(str(path))
        try:
            self.vertices = np.asarray(z["vertices"], dtype=np.float32)
            self.faces = np.asarray(z["faces"], dtype=np.int32)
            self.optic = np.asarray(z["optic"], dtype=np.uint8)
            self.vertices_lod = np.asarray(z["vertices_lod"], dtype=np.float32) if "vertices_lod" in z.files else self.vertices
            self.faces_lod = np.asarray(z["faces_lod"], dtype=np.int32) if "faces_lod" in z.files else self.faces
            self.optic_lod = np.asarray(z["optic_lod"], dtype=np.uint8) if "optic_lod" in z.files else self.optic
            self.sk_points = np.asarray(z["sk_points"], dtype=np.float32)
            self.sk_edges = np.asarray(z["sk_edges"], dtype=np.int32)
            self.neuron_ptr = np.asarray(z["neuron_ptr"], dtype=np.int32)
            self.neuron_kind = np.asarray(z["neuron_kind"], dtype=np.uint8)
            self.neuron_side = np.asarray(z["neuron_side"], dtype=np.uint8)
            self.neuron_kc = np.asarray(z["neuron_kc"], dtype=np.int32)
            self.root_id = np.asarray(z["root_id"], dtype=np.int64)
            self.fit_dist = float(np.asarray(z["fit_dist"]).reshape(-1)[0]) if "fit_dist" in z.files else 4.6
        finally:
            z.close()
        counts = np.diff(self.neuron_ptr).astype(np.int32)
        if int(counts.sum()) != len(self.sk_edges):
            raise RuntimeError("neuron_ptr не сходится с рёбрами скелета")
        self.edge_neuron = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        self.edge_kind = self.neuron_kind[self.edge_neuron]
        self.path = path
        self.mtime = path.stat().st_mtime_ns
        self.token = (str(path), self.mtime, int(len(self.sk_edges)))
        self.face_optic = _face_tag(self.optic, self.faces)
        self.face_optic_lod = _face_tag(self.optic_lod, self.faces_lod)
        self.wire_lod = _unique_edges(self.faces_lod)
        self.wire = _unique_edges(self.faces)

    @property
    def n_neurons(self) -> int:
        return int(len(self.neuron_kind))


def _face_tag(optic: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tags = optic[faces]
    left = (tags == 1).sum(axis=1)
    right = (tags == 2).sum(axis=1)
    out = np.zeros(len(faces), dtype=np.uint8)
    out[left >= 2] = 1
    out[right >= 2] = 2
    return out


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int32)


def load_scene() -> FlyScene | None:
    global _SCENE
    path = view_path()
    if not path.is_file() or path.stat().st_size < 1000:
        return None
    mtime = path.stat().st_mtime_ns
    if _SCENE is not None and _SCENE.path == path and _SCENE.mtime == mtime:
        return _SCENE
    try:
        _SCENE = FlyScene(path)
    except Exception as exc:
        print("мозг: не читается %s (%s)" % (path.name, exc), file=sys.stderr)
        return None
    return _SCENE


def available() -> bool:
    return load_scene() is not None


def lobe_anchors(scene: FlyScene):
    """Optic lobes. Tag 1 is anatomical left, «глаз Л» / recognized_L."""
    found = []
    for tag, side, label in ((1, 1, "глаз Л"), (2, 0, "глаз П")):
        mask = scene.optic == tag
        if not np.any(mask):
            continue
        pos = scene.vertices[mask].mean(axis=0).astype(np.float64)
        found.append({"tag": tag, "side": side, "label": label, "pos": pos})
    return found


def forward_arrow(scene: FlyScene):
    verts = scene.vertices
    top = float(np.percentile(verts[:, 1], 99))
    front = float(np.percentile(verts[:, 2], 93))
    start = np.array([0.0, top + 0.18, front - 0.20], dtype=np.float64)
    tip = np.array([0.0, top + 0.18, front + 0.32], dtype=np.float64)
    return start, tip


def _ids(packet, key) -> np.ndarray:
    raw = packet.get(key) or []
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        return np.zeros(0, dtype=np.int32)
    try:
        return np.asarray(raw, dtype=np.int32)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=np.int32)


def activity(scene: FlyScene, packet: dict):
    """KC skeletons named by the packet, and DAN flashes per hemisphere."""
    lit = np.zeros(scene.n_neurons, dtype=bool)
    ids = np.concatenate([_ids(packet, "kc_l"), _ids(packet, "kc_r")])
    if ids.size:
        lit |= (scene.neuron_kc >= 0) & np.isin(scene.neuron_kc, ids)
    flash = np.zeros(scene.n_neurons, dtype=np.uint8)
    for side, key in ((1, "flash_l"), (0, "flash_r")):
        kind = str(packet.get(key) or "")
        mask = scene.neuron_side == side
        if kind == "pam":
            flash[mask & (scene.neuron_kind == KIND_PAM)] = 1
        elif kind == "ppl1":
            flash[mask & (scene.neuron_kind == KIND_PPL1)] = 2
    return lit, flash


def _bright(rgb):
    return tuple(int(channel * 0.40 + 255 * 0.60) for channel in rgb)


_PROJECT_GLSL = """
vec4 project(vec3 p) {
    float cyaw = cos(yaw);
    float syaw = sin(yaw);
    float cp = cos(pitch);
    float sp = sin(pitch);
    float x1 = p.x * cyaw - p.z * syaw - pan_x;
    float z1 = p.x * syaw + p.z * cyaw;
    float y2 = p.y * cp - z1 * sp - pan_y;
    float z2 = p.y * sp + z1 * cp + dist;
    if (z2 < 0.35) {
        return vec4(0.0, 0.0, 0.0, -1.0);
    }
    float clip_x = (2.0 * cx / width - 1.0) * z2 + x1 * fov * 2.0 / width;
    float clip_y = (1.0 - 2.0 * cy / height) * z2 + y2 * fov * 2.0 / height;
    float ndc_z = clamp((z2 - 0.25) / (z2 + 6.0), 0.0, 1.0);
    return vec4(clip_x, clip_y, ndc_z * z2, z2);
}
"""

_MESH_VERT = """
#version 330
uniform float yaw, pitch, dist, pan_x, pan_y, fov, cx, cy, width, height;
in vec3 in_pos;
in float in_optic;
flat out float v_optic;
%s
void main() {
    v_optic = in_optic;
    gl_Position = project(in_pos);
}
""" % _PROJECT_GLSL

_MESH_FRAG = """
#version 330
uniform float rec_l, rec_r;
flat in float v_optic;
out vec4 fragColor;
void main() {
    vec3 col = vec3(0.58, 0.59, 0.62);
    float a = 0.15;
    if (v_optic > 1.5) {
        col = vec3(0.48, 0.54, 0.62);
        a = 0.20;
        if (rec_r > 0.5) {
            col = vec3(0.68, 0.84, 0.96);
            a = 0.38;
        }
    } else if (v_optic > 0.5) {
        col = vec3(0.48, 0.54, 0.62);
        a = 0.20;
        if (rec_l > 0.5) {
            col = vec3(0.68, 0.84, 0.96);
            a = 0.38;
        }
    }
    fragColor = vec4(col, a);
}
"""

_LINE_VERT = """
#version 330
uniform float yaw, pitch, dist, pan_x, pan_y, fov, cx, cy, width, height;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
%s
void main() {
    v_color = in_color;
    gl_Position = project(in_pos);
}
""" % _PROJECT_GLSL

_LINE_FRAG = """
#version 330
uniform float alpha;
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, alpha);
}
"""


def _pack_lines(points: np.ndarray, edges: np.ndarray, color: np.ndarray) -> np.ndarray:
    """color is (E, 3) float 0..1 or a single RGB tuple."""
    pts = points[edges]
    if isinstance(color, tuple):
        col = np.empty((len(edges), 2, 3), dtype=np.float32)
        col[:] = np.asarray(color, dtype=np.float32)
    else:
        col = np.repeat(color.reshape(-1, 1, 3), 2, axis=1)
    return np.concatenate([pts, col.astype(np.float32)], axis=2).reshape(-1, 6).astype(np.float32)


class _GLRenderer:
    def __init__(self, scene: FlyScene):
        moderngl = _moderngl()
        if moderngl is None:
            raise RuntimeError("moderngl не установлен")
        self.moderngl = moderngl
        self.ctx = moderngl.create_standalone_context(require=330)
        self.mesh_prog = self.ctx.program(vertex_shader=_MESH_VERT, fragment_shader=_MESH_FRAG)
        self.line_prog = self.ctx.program(vertex_shader=_LINE_VERT, fragment_shader=_LINE_FRAG)
        optic = scene.optic.astype(np.float32).reshape(-1, 1)
        mesh = np.concatenate([scene.vertices, optic], axis=1).astype(np.float32)
        self.mesh_vbo = self.ctx.buffer(np.ascontiguousarray(mesh).tobytes())
        self.mesh_ibo = self.ctx.buffer(np.ascontiguousarray(scene.faces.astype(np.uint32)).tobytes())
        self.mesh_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(self.mesh_vbo, "3f 1f", "in_pos", "in_optic")],
            self.mesh_ibo,
        )
        rgb = np.zeros((len(scene.sk_edges), 3), dtype=np.float32)
        for kind, color in KIND_RGB.items():
            rgb[scene.edge_kind == kind] = np.asarray(color, dtype=np.float32) / 255.0
        packed = _pack_lines(scene.sk_points, scene.sk_edges, rgb)
        self.line_vbo = self.ctx.buffer(np.ascontiguousarray(packed).tobytes())
        self.line_vao = self.ctx.vertex_array(self.line_prog, [(self.line_vbo, "3f 3f", "in_pos", "in_color")])
        self.n_line_verts = int(packed.shape[0])
        wire_col = (0.72, 0.75, 0.80)
        wire = _pack_lines(scene.vertices, scene.wire, wire_col)
        self.wire_vbo = self.ctx.buffer(np.ascontiguousarray(wire).tobytes())
        self.wire_vao = self.ctx.vertex_array(self.line_prog, [(self.wire_vbo, "3f 3f", "in_pos", "in_color")])
        self.n_wire_verts = int(wire.shape[0])
        self.token = scene.token
        self.size = (0, 0)
        self.fbo = None
        self.color = None
        self._hot_vbo = None
        self._hot_vao = None
        self._hot_cap = 0

    def resize(self, w: int, h: int) -> None:
        if self.size == (w, h) and self.fbo is not None:
            return
        if self.fbo is not None:
            self.fbo.release()
            self.color.release()
        self.color = self.ctx.texture((w, h), 4)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color])
        self.size = (w, h)

    def _uniforms(self, prog, yaw, pitch, dist, pan_x, pan_y, w, h) -> None:
        fov = float(min(w, h) * 1.05)
        prog["yaw"].value = float(yaw)
        prog["pitch"].value = float(pitch)
        prog["dist"].value = float(dist)
        prog["pan_x"].value = float(pan_x)
        prog["pan_y"].value = float(pan_y)
        prog["fov"].value = fov
        prog["cx"].value = float(w) * 0.50
        prog["cy"].value = float(h) * 0.54
        prog["width"].value = float(w)
        prog["height"].value = float(h)

    def _hot(self, packed: np.ndarray) -> None:
        if packed.shape[0] == 0:
            return
        raw = np.ascontiguousarray(packed, dtype=np.float32)
        need = int(raw.nbytes)
        if self._hot_vbo is None or need > self._hot_cap:
            if self._hot_vao is not None:
                self._hot_vao.release()
                self._hot_vbo.release()
            self._hot_cap = max(need, 256 * 6 * 4)
            self._hot_vbo = self.ctx.buffer(reserve=self._hot_cap)
            self._hot_vao = self.ctx.vertex_array(
                self.line_prog, [(self._hot_vbo, "3f 3f", "in_pos", "in_color")]
            )
        self._hot_vbo.write(raw.tobytes())
        self._hot_vao.render(self.moderngl.LINES, vertices=int(raw.shape[0]))

    def draw(self, scene: FlyScene, packet: dict, w: int, h: int, yaw, pitch, dist, pan_x, pan_y, frames: bool) -> np.ndarray:
        self.resize(w, h)
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(self.moderngl.DEPTH_TEST)
        self.ctx.disable(self.moderngl.CULL_FACE)
        self.ctx.enable(self.moderngl.BLEND)
        self.ctx.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._uniforms(self.mesh_prog, yaw, pitch, dist, pan_x, pan_y, w, h)
        self.mesh_prog["rec_l"].value = 1.0 if packet.get("rec_l") else 0.0
        self.mesh_prog["rec_r"].value = 1.0 if packet.get("rec_r") else 0.0
        self._sort_mesh(scene, yaw, pitch, dist, pan_x, pan_y)
        self.mesh_vao.render(self.moderngl.TRIANGLES)
        self._uniforms(self.line_prog, yaw, pitch, dist, pan_x, pan_y, w, h)
        if frames and self.n_wire_verts:
            self.line_prog["alpha"].value = 0.45
            self.wire_vao.render(self.moderngl.LINES, vertices=self.n_wire_verts)
        if self.n_line_verts:
            self.line_prog["alpha"].value = 0.55
            self.line_vao.render(self.moderngl.LINES, vertices=self.n_line_verts)
        lit, flash = activity(scene, packet)
        self.line_prog["alpha"].value = 1.0
        self._draw_hot(scene, lit, flash)
        data = self.fbo.read(components=3, alignment=1)
        image = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.flipud(image).copy()

    def _sort_mesh(self, scene: FlyScene, yaw, pitch, dist, pan_x, pan_y) -> None:
        """Far triangles first, so the translucent shell does not stack on itself."""
        x = scene.vertices[:, 0].astype(np.float64)
        y = scene.vertices[:, 1].astype(np.float64)
        z = scene.vertices[:, 2].astype(np.float64)
        cyaw, syaw = np.cos(float(yaw)), np.sin(float(yaw))
        cp, sp = np.cos(float(pitch)), np.sin(float(pitch))
        z1 = x * syaw + z * cyaw
        z2 = y * sp + z1 * cp + float(dist)
        depth = z2[scene.faces].mean(axis=1)
        ordered = scene.faces[np.argsort(-depth)].astype(np.uint32)
        self.mesh_ibo.write(np.ascontiguousarray(ordered).tobytes())

    def _draw_hot(self, scene: FlyScene, lit: np.ndarray, flash: np.ndarray) -> None:
        if lit.any():
            sel = np.flatnonzero(lit[scene.edge_neuron])
            if sel.size:
                rgb = np.zeros((sel.size, 3), dtype=np.float32)
                kinds = scene.edge_kind[sel]
                for kind, color in KIND_RGB.items():
                    rgb[kinds == kind] = np.asarray(_bright(color), dtype=np.float32) / 255.0
                self._hot(_pack_lines(scene.sk_points, scene.sk_edges[sel], rgb))
        for code, color in ((1, _PAM_HOT), (2, _PPL_HOT)):
            sel = np.flatnonzero(flash[scene.edge_neuron] == code)
            if sel.size == 0:
                continue
            tint = tuple(channel / 255.0 for channel in color)
            self._hot(_pack_lines(scene.sk_points, scene.sk_edges[sel], tint))


def _gl_renderer(scene: FlyScene):
    global _GL, _GL_FAILED
    if _GL_FAILED or os.environ.get("MB_VIEW_SOFTWARE") == "1":
        return None
    if _moderngl() is None:
        return None
    try:
        if _GL is None or _GL.token != scene.token:
            _GL = _GLRenderer(scene)
        return _GL
    except Exception as exc:
        _GL_FAILED = True
        _GL = None
        print("мозг: moderngl недоступен, рисую линиями pygame (%s)" % exc, file=sys.stderr)
        return None


def _blit_rgb(surface, image: np.ndarray) -> None:
    import pygame

    view = np.ascontiguousarray(np.transpose(image, (1, 0, 2)))
    pygame.surfarray.blit_array(surface, view)


def _camera(w: int, h: int):
    return w * 0.50, h * 0.54, float(min(w, h)) * 1.05


def _paint_software(surface, scene: FlyScene, packet: dict, yaw, pitch, dist, pan_x, pan_y, frames: bool, fast: bool) -> None:
    import pygame

    from .mb_view3d import _project

    w, h = surface.get_size()
    surface.fill((0, 0, 0))
    cx, cy, fov = _camera(w, h)
    rec_l = bool(packet.get("rec_l"))
    rec_r = bool(packet.get("rec_r"))
    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    _draw_shell(layer, scene, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y, rec_l, rec_r)
    surface.blit(layer, (0, 0))
    sx, sy, dep = _project(scene.sk_points, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    if frames and len(scene.wire_lod):
        _draw_indexed(surface, sx, sy, dep, scene.wire_lod, np.arange(len(scene.wire_lod)), (176, 182, 190), w, h, scene.vertices_lod, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    step = 1
    if fast and len(scene.sk_edges) > 22000:
        step = int(np.ceil(len(scene.sk_edges) / 22000.0))
    for kind, color in KIND_RGB.items():
        idx = np.flatnonzero(scene.edge_kind == kind)
        if step > 1:
            idx = idx[::step]
        _draw_indexed(surface, sx, sy, dep, scene.sk_edges, idx, tuple(int(c * 0.62) for c in color), w, h)
    lit, flash = activity(scene, packet)
    if lit.any():
        for kind, color in KIND_RGB.items():
            sel = np.flatnonzero(lit[scene.edge_neuron] & (scene.edge_kind == kind))
            _draw_indexed(surface, sx, sy, dep, scene.sk_edges, sel, _bright(color), w, h)
    for code, color in ((1, _PAM_HOT), (2, _PPL_HOT)):
        sel = np.flatnonzero(flash[scene.edge_neuron] == code)
        _draw_indexed(surface, sx, sy, dep, scene.sk_edges, sel, color, w, h)


def _draw_shell(layer, scene, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y, rec_l, rec_r) -> None:
    import pygame

    from .mb_view3d import _project

    verts = scene.vertices_lod
    faces = scene.faces_lod
    if len(faces) == 0:
        return
    sx, sy, dep = _project(verts, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    depth = dep[faces].mean(axis=1)
    order = np.argsort(-depth)
    tags = scene.face_optic_lod
    for i in order:
        tri = faces[i]
        if dep[tri[0]] < 0.35 or dep[tri[1]] < 0.35 or dep[tri[2]] < 0.35:
            continue
        tag = int(tags[i])
        if tag == 1 and rec_l:
            color = (150, 196, 224, 120)
        elif tag == 2 and rec_r:
            color = (150, 196, 224, 120)
        elif tag:
            color = (118, 128, 142, 78)
        else:
            color = (148, 150, 156, 46)
        pts = [(int(sx[tri[k]]), int(sy[tri[k]])) for k in range(3)]
        pygame.draw.polygon(layer, color, pts)


def _draw_indexed(surface, sx, sy, dep, edges, indices, color, w, h, verts=None, yaw=0, pitch=0, dist=0, cx=0, cy=0, fov=0, pan_x=0, pan_y=0) -> None:
    """Draw edges. ``verts`` set means the indices point at that cloud, not the skeleton."""
    import pygame

    if len(indices) == 0:
        return
    if verts is not None:
        from .mb_view3d import _project

        sx, sy, dep = _project(verts, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    rgb = (int(color[0]), int(color[1]), int(color[2]))
    for i in indices:
        a = int(edges[int(i), 0])
        b = int(edges[int(i), 1])
        if a < 0 or b < 0 or a >= len(dep) or b >= len(dep):
            continue
        if dep[a] < 0.35 or dep[b] < 0.35:
            continue
        x0, y0 = int(sx[a]), int(sy[a])
        x1, y1 = int(sx[b]), int(sy[b])
        if (x0 < -30 and x1 < -30) or (y0 < -30 and y1 < -30) or (x0 >= w + 30 and x1 >= w + 30) or (y0 >= h + 30 and y1 >= h + 30):
            continue
        pygame.draw.line(surface, rgb, (x0, y0), (x1, y1), 1)


def _overlay(surface, scene, packet, yaw, pitch, dist, pan_x, pan_y, _auto: bool, frames: bool, hint: str | None = None) -> None:
    import pygame

    from .mb_view3d import _project

    w, h = surface.get_size()
    cx, cy, fov = _camera(w, h)
    font = pygame.font.Font(None, 20)
    small = pygame.font.Font(None, 16)
    tiny = pygame.font.Font(None, 15)
    label_font = pygame.font.Font(None, 18)
    surface.blit(font.render("грибовидное тело", True, _INK), (16, 12))
    note = "оболочка FlyWire   скелеты KC γ / αβ / a'b'   MBON   PAM   PPL1"
    surface.blit(small.render(note, True, _INK), (16, 34))
    surface.blit(small.render("глаз Л и глаз П — зрительные доли, светятся при узнавании", True, _INK), (16, 52))
    if hint is None:
        hint = "ЛКМ обзор   ПКМ/СКМ/Shift сдвиг   колёсико зум   стрелки WASD   Home сброс   F каркас"
        if frames:
            hint += " вкл"
    surface.blit(tiny.render(hint, True, _HINT), (12, h - 20))
    rec_l = bool(packet.get("rec_l"))
    rec_r = bool(packet.get("rec_r"))
    for item in lobe_anchors(scene):
        recognized = rec_l if item["side"] == 1 else rec_r
        pos = item["pos"].reshape(1, 3).copy()
        pos[0, 0] += -0.10 if item["side"] == 1 else 0.10
        pos[0, 1] += 0.08
        px, py, dep = _project(pos, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        if float(dep[0]) <= 0.35:
            continue
        color = (186, 220, 238) if recognized else _INK
        text = label_font.render(item["label"], True, color)
        x = int(px[0])
        y = int(py[0])
        if item["side"] == 1:
            x -= text.get_width() + 4
        else:
            x += 4
        surface.blit(text, (x, y - text.get_height() // 2))
    start, tip = forward_arrow(scene)
    pts = np.vstack((start, tip))
    px, py, dep = _project(pts, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    if float(dep[0]) <= 0.35 or float(dep[1]) <= 0.35:
        return
    color = (214, 218, 224)
    x0, y0 = int(px[0]), int(py[0])
    x1, y1 = int(px[1]), int(py[1])
    pygame.draw.line(surface, color, (x0, y0), (x1, y1), 1)
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / norm, dy / norm
    left = (int(x1 - ux * 8 - uy * 4), int(y1 - uy * 8 + ux * 4))
    right = (int(x1 - ux * 8 + uy * 4), int(y1 - uy * 8 - ux * 4))
    pygame.draw.line(surface, color, (x1, y1), left, 1)
    pygame.draw.line(surface, color, (x1, y1), right, 1)
    caption = label_font.render("вперёд", True, _INK)
    surface.blit(caption, (x1 - caption.get_width() // 2, y1 - caption.get_height() - 4))


def paint_scene(surface, packet: dict, yaw: float, pitch: float, dist: float, auto: bool = True, pan_x: float = 0.0, pan_y: float = 0.0, frames: bool = False, fast: bool = False, hint: str | None = None) -> bool:
    scene = load_scene()
    if scene is None:
        return False
    renderer = _gl_renderer(scene)
    if renderer is not None:
        w, h = surface.get_size()
        image = renderer.draw(scene, packet, w, h, yaw, pitch, dist, pan_x, pan_y, frames)
        _blit_rgb(surface, image)
    else:
        _paint_software(surface, scene, packet, yaw, pitch, dist, pan_x, pan_y, frames, fast)
    _overlay(surface, scene, packet, yaw, pitch, dist, pan_x, pan_y, auto, frames, hint)
    return True
