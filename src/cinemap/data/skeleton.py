"""Load precomputed neuroglancer skeletons and turn them into renderable tubes.

A skeleton source's own `info` is a `neuroglancer_skeletons` info (no volume
`scales`), so — exactly like the mesh loader — we point CloudVolume at the parent
dir with a fabricated *volume* info whose `skeletons` key names the subdir, then
read per-segment skeletons via `cv.skeleton.get`.

Skeletons are vertices (nm, x/y/z) + edges. Neuroglancer draws them as screen-space
lines; for a 3D render we sweep each edge into a thin cylinder so the skeleton has
real geometry the camera can orbit. Tubes are built vectorized (one combined mesh
for a whole layer's selected segments) to stay fast across thousands of segments.
"""
from __future__ import annotations

import ast
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh
from cloudvolume import CloudVolume

# concurrent per-segment fetches; logical CPU count (incl. hyperthreading) by default
_FETCH_WORKERS = int(os.environ.get("CINEMAP_FETCH_WORKERS") or (os.cpu_count() or 8))

# Default tube radius (nm). Skeletons are 1D, so this is a render choice, not data;
# tuned to read as a visible strand at EM/organelle scale. Override per call.
DEFAULT_RADIUS_NM = 60.0
# Sides per tube cross-section. 6 is a clean low-poly tube; bump for hero closeups.
_SIDES = 6


class ShaderColormap:
    """A piecewise-smoothstep colormap recovered from a neuroglancer skeleton
    `skeletonRendering.shader`. Matches the common cellmap pattern:

        emitRGB(<map>(min(1.0, <attr>/<norm>)))

    where <map> is built from `const float eN = ...; const vec3 vN = vec3(...)`
    control points and consecutive stops are blended with `smoothstep`. So skeleton
    tubes get the same per-vertex coloring the user sees in neuroglancer instead of
    a flat segment color."""

    def __init__(self, attr: str, norm: float, edges: list[float], colors: list[list[float]]):
        self.attr = attr
        self.norm = norm or 1.0
        self.edges = np.asarray(edges, dtype=np.float64)
        self.colors = np.asarray(colors, dtype=np.float64)  # (S,3) in 0..1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Map normalized values `x` (already divided by norm) to (N,3) rgb."""
        x = np.clip(np.asarray(x, dtype=np.float64), self.edges[0], self.edges[-1])
        out = np.tile(self.colors[0], (len(x), 1))
        for i in range(len(self.edges) - 1):
            e0, e1 = self.edges[i], self.edges[i + 1]
            seg = (x >= e0) & (x <= e1)
            if not seg.any():
                continue
            t = np.clip((x[seg] - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
            a = t * t * (3 - 2 * t)  # smoothstep
            out[seg] = self.colors[i] * (1 - a)[:, None] + self.colors[i + 1] * a[:, None]
        return out


def _turbo(x: np.ndarray) -> np.ndarray:
    """Google/Neuroglancer turbo colormap polynomial used by many GLSL shaders."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    v4 = np.stack([np.ones_like(x), x, x * x, x * x * x], axis=1)
    v2 = np.stack([x ** 4, x ** 5], axis=1)
    red = v4 @ np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234])
    red += v2 @ np.array([-152.94239396, 59.28637943])
    green = v4 @ np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333])
    green += v2 @ np.array([4.27729857, 2.82956604])
    blue = v4 @ np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771])
    blue += v2 @ np.array([-89.90310912, 27.34824973])
    return np.clip(np.stack([red, green, blue], axis=1), 0.0, 1.0)


def _edge_attr_values(skel, edges: np.ndarray, name: str) -> np.ndarray | None:
    """Return a per-edge scalar skeleton attribute matching `prop_<name>()`.

    Neuroglancer skeleton attributes are normally per-vertex.  A tube edge is colored
    by averaging its two endpoint values, which matches how the old hand-coded
    radius path behaved and gives stable colors after tube tessellation.
    """
    candidates = [name]
    if name == "radius":
        candidates.extend(["radii", "vertex_radius", "vertex_radii"])
    else:
        candidates.extend([f"{name}s", f"vertex_{name}", f"vertex_{name}s"])

    vals = None
    for candidate in candidates:
        if hasattr(skel, candidate):
            vals = getattr(skel, candidate)
            break
    if vals is None and hasattr(skel, "attributes"):
        attrs = getattr(skel, "attributes")
        if isinstance(attrs, dict):
            for candidate in candidates:
                if candidate in attrs:
                    vals = attrs[candidate]
                    break
    if vals is None and hasattr(skel, "extra_attributes"):
        attrs = getattr(skel, "extra_attributes")
        if isinstance(attrs, dict):
            for candidate in candidates:
                if candidate in attrs:
                    vals = attrs[candidate]
                    break
    if vals is None:
        return None

    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    if arr.size == len(edges):
        return arr
    if arr.size <= int(np.max(edges)):
        return None
    return (arr[edges[:, 0]] + arr[edges[:, 1]]) * 0.5


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _mix(a, b, t):
    return a * (1.0 - t) + b * t


def _minimum(*args):
    out = args[0]
    for arg in args[1:]:
        out = np.minimum(out, arg)
    return out


def _maximum(*args):
    out = args[0]
    for arg in args[1:]:
        out = np.maximum(out, arg)
    return out


def _vec(n: int, *args):
    if len(args) == 1:
        args = args * n
    if len(args) != n:
        raise ValueError(f"vec{n} expects 1 or {n} arguments")
    arrays = [np.asarray(a, dtype=np.float64) for a in args]
    if any(a.ndim > 0 for a in arrays):
        return np.stack(np.broadcast_arrays(*arrays), axis=-1)
    return np.asarray(args, dtype=np.float64)


_EVAL_FUNCS = {
    "abs": np.abs,
    "clamp": np.clip,
    "exp": np.exp,
    "log": np.log,
    "max": _maximum,
    "min": _minimum,
    "mix": _mix,
    "pow": np.power,
    "sqrt": np.sqrt,
    "smoothstep": _smoothstep,
    "turbo": _turbo,
    "vec2": lambda *args: _vec(2, *args),
    "vec3": lambda *args: _vec(3, *args),
    "vec4": lambda *args: _vec(4, *args),
}


def _eval_expr(expr: str, env: dict[str, object]):
    """Evaluate a side-effect-free GLSL-like expression as NumPy operations."""
    expr = re.sub(r"(?<=\d)f\b", "", expr)
    tree = ast.parse(expr, mode="eval")

    def ev(node):  # noqa: PLR0911
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise NameError(node.id)
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return np.power(left, right)
            raise ValueError("unsupported binary operator")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _EVAL_FUNCS.get(node.func.id)
            if fn is None:
                raise NameError(node.func.id)
            return fn(*(ev(a) for a in node.args))
        raise ValueError(f"unsupported expression: {ast.dump(node, include_attributes=False)}")

    return ev(tree)


class ExpressionShaderColorizer:
    """Evaluate a safe Neuroglancer skeleton-shader subset and bake edge colors.

    This is intentionally a constrained expression evaluator, not a general GLSL
    compiler. It supports the operations commonly used in NG skeleton shaders:
    uicontrol float defaults, `prop_*()` skeleton attributes, float/vector
    assignments in `main`, math functions, `vec*` constructors, and `emitRGB(A)`.
    """

    def __init__(self, shader: str):
        self.shader = shader or ""
        self.controls = {
            name: float(value)
            for name, value in re.findall(
                r"#uicontrol\s+float\s+(\w+)\s+slider\([^)]*default=([\d.eE+-]+)",
                self.shader,
            )
        }
        body = self._main_body(self.shader)
        self.assignments: list[tuple[str, str]] = []
        self.emit_kind = ""
        self.emit_expr = ""
        if body:
            for raw in body.split(";"):
                stmt = raw.strip()
                if not stmt:
                    continue
                m_emit = re.match(r"emit(RGB|RGBA)\s*\((.*)\)\s*$", stmt, re.S)
                if m_emit:
                    self.emit_kind = m_emit.group(1)
                    self.emit_expr = m_emit.group(2).strip()
                    continue
                m_assign = re.match(
                    r"(?:const\s+)?(?:float|int|vec[234]|bool)\s+(\w+)\s*=\s*(.*)\s*$",
                    stmt,
                    re.S,
                )
                if not m_assign:
                    m_assign = re.match(r"(\w+)\s*=\s*(.*)\s*$", stmt, re.S)
                if m_assign:
                    self.assignments.append((m_assign.group(1), m_assign.group(2).strip()))

    @staticmethod
    def _main_body(shader: str) -> str:
        shader = re.sub(r"//.*", "", shader)
        m = re.search(r"void\s+main\s*\([^)]*\)\s*\{(.*?)\}", shader, re.S)
        return m.group(1) if m else ""

    @property
    def usable(self) -> bool:
        return bool(self.emit_expr)

    def _prepare_expr(self, expr: str, skel, edges: np.ndarray, env: dict[str, object]) -> str:
        def repl(match: re.Match) -> str:
            name = match.group(1)
            key = f"__prop_{name}"
            if key not in env:
                values = _edge_attr_values(skel, edges, name)
                if values is None:
                    raise NameError(f"prop_{name}")
                env[key] = values
            return key

        expr = re.sub(r"\bprop_(\w+)\s*\(\s*\)", repl, expr)
        return re.sub(r"(?<=\d)f\b", "", expr)

    @staticmethod
    def _as_rgba(value, n: int, use_alpha: bool) -> np.ndarray | None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1 and arr.size in (3, 4):
            arr = np.tile(arr[None, :], (n, 1))
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] not in (3, 4):
            return None
        rgba = np.empty((n, 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(arr[:, :3] * 255.0, 0, 255).astype(np.uint8)
        alpha = arr[:, 3] if (use_alpha and arr.shape[1] == 4) else 1.0
        rgba[:, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        return rgba

    def edge_rgba(self, skel, edges) -> np.ndarray | None:
        if not self.usable:
            return None
        env: dict[str, object] = {
            "PI": np.pi,
            "false": False,
            "true": True,
            **self.controls,
        }
        try:
            for name, expr in self.assignments:
                env[name] = _eval_expr(self._prepare_expr(expr, skel, edges, env), env)
            value = _eval_expr(self._prepare_expr(self.emit_expr, skel, edges, env), env)
            return self._as_rgba(value, len(edges), self.emit_kind == "RGBA")
        except Exception:  # noqa: BLE001 - unsupported shader subset falls back cleanly
            return None


def _parse_legacy_shader_colormap(shader: str) -> ShaderColormap | None:
    """Recover the older CellMap piecewise-smoothstep colormap pattern."""
    floats = dict(re.findall(r"float\s+(\w+)\s*=\s*([\d.eE+f-]+)", shader))
    edges, colors = [], []
    i = 0
    while f"e{i}" in floats:
        m = re.search(rf"vec3\s+v{i}\s*=\s*vec3\(([^)]+)\)", shader)
        if not m:
            break
        edges.append(float(floats[f"e{i}"].rstrip("f")))
        colors.append([float(c) for c in m.group(1).split(",")[:3]])
        i += 1
    if len(edges) < 2:
        return None
    # attribute and its normalization, from e.g. `lsp_nm/norm` with `float norm = 50000.0f`
    m = re.search(r"(\w+)\s*/\s*(\w+|[\d.eE+f-]+)", shader)
    if not m:
        return None
    attr, denom = m.group(1), m.group(2)
    norm = float(floats.get(denom, denom).rstrip("f")) if denom else 1.0
    return ShaderColormap(attr, norm, edges, colors)


def parse_shader_colormap(shader: str) -> object | None:
    """Recover a skeleton shader colorizer, or None if it cannot be evaluated."""
    if not shader:
        return None
    legacy = _parse_legacy_shader_colormap(shader)
    if legacy is not None:
        return legacy
    expr = ExpressionShaderColorizer(shader)
    if expr.usable:
        return expr
    return None


def _perp_frame(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each unit direction `d` (E,3)."""
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(d), 1))
    # where d is ~parallel to z, use x as the reference instead
    nearly_z = np.abs(d[:, 2]) > 0.9
    ref[nearly_z] = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    v = np.cross(d, u)
    return u, v


def edges_to_tubes(verts: np.ndarray, edges: np.ndarray, radius: float,
                   rgba: np.ndarray | None = None, sides: int = _SIDES) -> trimesh.Trimesh | None:
    """Sweep each (p0,p1) edge into a `sides`-gon cylinder. `rgba` (E,4 uint8) tints
    each edge. Returns one combined Trimesh (vertices in the same nm frame), or None
    when there is no non-degenerate edge."""
    if len(edges) == 0:
        return None
    p0 = verts[edges[:, 0]].astype(np.float64)
    p1 = verts[edges[:, 1]].astype(np.float64)
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    keep = length > 1e-6
    if not keep.any():
        return None
    p0, p1, d, length = p0[keep], p1[keep], d[keep], length[keep]
    if rgba is not None:
        rgba = rgba[keep]
    d /= length[:, None]
    u, v = _perp_frame(d)  # (E,3) each

    e = len(p0)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = (np.cos(theta)[:, None] * u[:, None, :] +
            np.sin(theta)[:, None] * v[:, None, :])  # (E, sides, 3)
    ring0 = p0[:, None, :] + radius * ring           # (E, sides, 3)
    ring1 = p1[:, None, :] + radius * ring
    verts_out = np.concatenate([ring0, ring1], axis=1).reshape(-1, 3)  # (E*2*sides, 3)

    # faces for one edge's cylinder wall (two triangles per side quad)
    k = np.arange(sides)
    kn = (k + 1) % sides
    quad = np.stack([k, kn, sides + kn, sides + k], axis=1)            # (sides, 4)
    tri = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]], axis=0)  # (2*sides, 3)
    offsets = (np.arange(e) * 2 * sides)[:, None, None]
    faces_out = (tri[None] + offsets).reshape(-1, 3)

    colors = None
    if rgba is not None:
        colors = np.repeat(rgba, 2 * sides, axis=0)  # per-vertex from per-edge color
    return trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_colors=colors,
                           process=False)


class SkeletonLoader:
    def __init__(self, skeleton_url: str = "", radius_nm: float = DEFAULT_RADIUS_NM,
                 shader: str = ""):
        self.skeleton_url = (skeleton_url or "").rstrip("/")
        self.radius_nm = radius_nm
        self.parent, self.subdir = self.skeleton_url.rsplit("/", 1) if self.skeleton_url else ("", "")
        self.colormap = parse_shader_colormap(shader)  # None if shader has no colormap
        self._cv = None

    @property
    def cv(self) -> CloudVolume:
        if self._cv is None:
            info = {
                "@type": "neuroglancer_multiscale_volume",
                "type": "segmentation",
                "data_type": "uint64",
                "num_channels": 1,
                "skeletons": self.subdir,
                "scales": [{
                    "key": "s0", "size": [1, 1, 1], "resolution": [1, 1, 1],
                    "chunk_sizes": [[64, 64, 64]], "encoding": "raw", "voxel_offset": [0, 0, 0],
                }],
            }
            self._cv = CloudVolume(
                f"precomputed://{self.parent}", info=info, use_https=True, progress=False
            )
        return self._cv

    def _edge_colors(self, skel, edges, seg_id, colorize) -> np.ndarray | None:
        """Per-edge rgba (E,4 uint8). Uses the shader colormap on the skeleton's
        scalar attribute when available (matching neuroglancer); otherwise the flat
        per-segment color from `colorize`."""
        cm = self.colormap
        if cm is not None and hasattr(cm, "edge_rgba"):
            rgba = cm.edge_rgba(skel, edges)
            if rgba is not None:
                return rgba
        if cm is not None and hasattr(skel, cm.attr):
            attr = np.asarray(getattr(skel, cm.attr), dtype=np.float64).reshape(-1)
            ev = (attr[edges[:, 0]] + attr[edges[:, 1]]) * 0.5 / cm.norm  # per-edge, normalized
            rgb = cm(ev)
            rgba = np.empty((len(edges), 4), dtype=np.uint8)
            rgba[:, :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            rgba[:, 3] = 255
            return rgba
        if colorize is not None:
            r, g, b = colorize(int(seg_id))
            return np.tile((np.array([r, g, b, 1.0]) * 255).astype(np.uint8), (len(edges), 1))
        return None

    def load_many(self, seg_ids, colorize=None, radius_nm: float | None = None) -> trimesh.Trimesh:
        """One combined tube mesh for all `seg_ids`. Colors each segment via the
        shader colormap (per-vertex attribute) when the layer has one, else the flat
        `colorize(seg_id)->rgb`. Skips segments with no skeleton."""
        seg_ids = list(seg_ids)
        if not seg_ids:
            raise ValueError("no segment ids")
        radius = self.radius_nm if radius_nm is None else radius_nm

        def _fetch(s):  # network + tube build per segment, run concurrently
            try:
                skel = self.cv.skeleton.get(int(s))
            except Exception as e:  # noqa: BLE001
                print(f"[skeleton] {s} failed: {e}")
                return None
            verts = np.asarray(skel.vertices, dtype=np.float64)
            edges = np.asarray(skel.edges, dtype=np.int64)
            if len(verts) == 0 or len(edges) == 0:
                return None
            rgba = self._edge_colors(skel, edges, s, colorize)
            return edges_to_tubes(verts, edges, radius, rgba=rgba)

        if len(seg_ids) > 1:
            with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(seg_ids))) as ex:
                parts = list(ex.map(_fetch, seg_ids))
        else:
            parts = [_fetch(seg_ids[0])]
        parts = [p for p in parts if p is not None]
        if not parts:
            raise ValueError("no skeleton geometry for the selected segments")
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
