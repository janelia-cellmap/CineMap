"""Neuroglancer shader replication — evaluate a layer's GLSL on the CPU.

CineMap already rasterizes each cross-section to a PNG before Blender ever sees it, so
the faithful way to reproduce what neuroglancer displays is to run neuroglancer's own
shader over the voxel array in numpy and bake the exact RGB it would show. Blender then
only has to *display* that image with the right color management (see the sRGB note in
`render/blender_script.py:_make_slice`) instead of re-deriving the look.

This is layer-agnostic on purpose: `#uicontrol` directives + `shaderControls` are used by
image, annotation, skeleton and single-mesh layers alike (segmentation MESHES instead use
objectAlpha / meshSilhouetteRendering, handled in `colors.py`). Anything here works for
any of them.

Ported against the neuroglancer source (src/webgl/shader_ui_controls.ts,
src/webgl/lerp.ts, src/webgl/colormaps.ts, src/sliceview/volume/image_renderlayer.ts):

  - invlerp:  v = (x - range[0]) / (range[1] - range[0]), then clamp(v, 0, 1) when the
              control's `clamp` is set (it defaults to TRUE).
  - `window` is a UI-only zoom of the control's slider. It NEVER affects pixels — only
    `range` does. Applying it would silently wreck the contrast.
  - default `range` comes from the volume dtype when the state doesn't override it.
  - default image shader is `#uicontrol invlerp normalized` / `emitGrayscale(normalized())`.
  - emitGrayscale(v) -> vec4(v,v,v,opacity); emitRGB(c) -> vec4(c,opacity);
    emitRGBA(c) -> vec4(c.rgb, c.a*opacity); emitTransparent() -> vec4(0,0,0,0).

Unsupported GLSL degrades to the default invlerp behaviour and reports a reason, so a
shader we can't read shows up as a warning rather than a silently wrong render.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# neuroglancer's defaultDataTypeRange (src/util/lerp.ts). Keyed by numpy dtype name.
DTYPE_RANGES: dict[str, tuple[float, float]] = {
    "uint8": (0, 0xFF),
    "int8": (-0x80, 0x7F),
    "uint16": (0, 0xFFFF),
    "int16": (-0x8000, 0x7FFF),
    "uint32": (0, 0xFFFFFFFF),
    "int32": (-0x80000000, 0x7FFFFFFF),
    "uint64": (0, 0xFFFFFFFFFFFFFFFF),
    "int64": (-0x8000000000000000, 0x7FFFFFFFFFFFFFFF),
    # float32/float64 normalize over the unit interval, NOT the dtype extent.
    "float32": (0.0, 1.0),
    "float64": (0.0, 1.0),
}

DEFAULT_IMAGE_SHADER = "#uicontrol invlerp normalized\nvoid main() {\n  emitGrayscale(normalized());\n}\n"


def default_range_for(dtype) -> tuple[float, float]:
    """neuroglancer's default invlerp range for a volume of this dtype."""
    return DTYPE_RANGES.get(np.dtype(dtype).name, (0.0, 1.0))


# --------------------------------------------------------------------------- controls
@dataclass
class Control:
    """One resolved `#uicontrol`. `value` is whatever the state settled on: a float for
    slider, (lo, hi) for invlerp, an rgb triple for color, bool for checkbox."""

    name: str
    kind: str            # invlerp | slider | color | checkbox | select
    value: Any
    clamp: bool = True   # invlerp only


def _strip_comments(code: str) -> str:
    """Drop // and /* */ comments — neuroglancer does this before matching directives,
    so a commented-out #uicontrol must not register."""
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


# `#uicontrol <type> <name> [<control>](<params>)`; the control kind defaults to the type
# (so `#uicontrol invlerp normalized` is an invlerp). Mirrors NG's innerPattern.
_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*uicontrol[ \t]+(.*)$", re.M)
_INNER = re.compile(
    r"^([_a-zA-Z][_a-zA-Z0-9]*)[ \t]+([a-z_][a-zA-Z0-9_]*)"
    r"(?:[ \t]+([a-z]+))?[ \t]*(?:\([ \t]*(.*?)[ \t]*\))?[ \t]*$",
    re.I,
)


def _split_params(text: str) -> list[str]:
    """Split `a=1, b=[2,3]` on commas that aren't inside brackets/quotes."""
    out, depth, cur, quote = [], 0, "", ""
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _parse_literal(text: str):
    """A directive parameter value: number, bool, [a,b] interval, or bare/quoted string."""
    t = text.strip()
    if not t:
        return None
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    # Tolerate a still-escaped quote (`\"red\"`). A properly JSON-decoded state never has
    # these, but hand-edited and double-encoded states do, and there is no other sensible
    # reading — without this the value silently degrades to the default color.
    if t.startswith('\\"') and t.endswith('\\"'):
        return t[2:-2]
    if t[0] in "\"'" and t[-1] == t[0]:
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        return [_parse_literal(p) for p in _split_params(t[1:-1])]
    try:
        return float(t)
    except ValueError:
        return t


def _parse_params(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    params: dict[str, Any] = {}
    for part in _split_params(text):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        params[k.strip()] = _parse_literal(v)
    return params


_CSS_COLORS = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 0.5019607843137255, 0.0),
    "blue": (0.0, 0.0, 1.0), "white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0),
    "yellow": (1.0, 1.0, 0.0), "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0),
    "gray": (0.5019607843137255,) * 3, "grey": (0.5019607843137255,) * 3,
    "orange": (1.0, 0.6470588235294118, 0.0),
}


def _parse_color(v) -> tuple[float, float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return tuple(float(c) for c in v[:3])  # type: ignore[return-value]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) >= 6:
                return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
        if s.lower() in _CSS_COLORS:
            return _CSS_COLORS[s.lower()]
    return (1.0, 1.0, 1.0)


def parse_directives(code: str, dtype) -> tuple[dict[str, Control], str]:
    """Read every `#uicontrol` out of `code`, returning the controls at their DECLARED
    defaults plus the code with the directive lines removed (as NG does)."""
    code = _strip_comments(code)
    controls: dict[str, Control] = {}

    def _sub(m: re.Match) -> str:
        inner = _INNER.match(m.group(1).strip())
        if inner is None:
            return ""
        type_name, var, ctrl, param_text = inner.groups()
        kind = (ctrl or type_name).lower()
        p = _parse_params(param_text)
        if kind == "invlerp":
            rng = p.get("range")
            lo, hi = ((float(rng[0]), float(rng[1]))
                      if isinstance(rng, (list, tuple)) and len(rng) >= 2
                      else default_range_for(dtype))
            controls[var] = Control(var, "invlerp", (lo, hi),
                                    clamp=bool(p.get("clamp", True)))
        elif kind == "slider":
            controls[var] = Control(var, "slider", float(p.get("default", p.get("min", 0.0)) or 0.0))
        elif kind == "color":
            controls[var] = Control(var, "color", _parse_color(p.get("default", "white")))
        elif kind == "checkbox":
            controls[var] = Control(var, "checkbox", bool(p.get("default", False)))
        elif kind == "select":
            controls[var] = Control(var, "select", p.get("default"))
        return ""

    return controls, _DIRECTIVE.sub(_sub, code)


def apply_shader_controls(controls: dict[str, Control], shader_controls: dict | None,
                          dtype) -> dict[str, Control]:
    """Overlay the state's `shaderControls` JSON onto the declared defaults.

    An invlerp entry is `{"range": [lo, hi], "window": [...]}`. Only `range` is read —
    `window` is the slider's zoom level in the UI and has no effect on rendered pixels.

    A bare `[lo, hi]` array is DISCARDED, matching neuroglancer. Its
    `parseImageInvlerpParameters` starts with `verifyObject`, which rejects arrays
    (util/json.ts), and `ShaderControlState.restoreState` swallows that error
    (shader_ui_controls.ts:1783) — so such an entry leaves the control at its default.
    Honoring it would make us apply a contrast window neuroglancer itself is ignoring;
    verified against a live viewer by spikes/ng_parity/shader_parity.py.
    """
    if not shader_controls:
        return controls
    out = dict(controls)
    for name, raw in shader_controls.items():
        cur = out.get(name)
        if cur is None:
            # A control set in the state but not declared in the shader we were given
            # (e.g. the shader fell back to default). Infer the shape from the value.
            if isinstance(raw, dict) and "range" in raw:
                cur = Control(name, "invlerp", default_range_for(dtype))
            elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
                cur = Control(name, "slider", float(raw))
            elif isinstance(raw, bool):
                cur = Control(name, "checkbox", raw)
            elif isinstance(raw, str):
                cur = Control(name, "color", _parse_color(raw))
            else:
                continue
        if cur.kind == "invlerp":
            if not isinstance(raw, dict):
                continue                      # bare array / scalar: neuroglancer ignores it
            rng = raw.get("range")
            clamp = raw.get("clamp", cur.clamp)
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                cur = Control(name, "invlerp", (float(rng[0]), float(rng[1])),
                              clamp=bool(clamp))
            else:  # e.g. {"window": [...]} only -> keep the declared range
                cur = Control(name, "invlerp", cur.value, clamp=bool(clamp))
        elif cur.kind == "slider":
            try:
                cur = Control(name, "slider", float(raw))
            except (TypeError, ValueError):
                pass
        elif cur.kind == "color":
            cur = Control(name, "color", _parse_color(raw))
        elif cur.kind == "checkbox":
            cur = Control(name, "checkbox", bool(raw))
        else:
            cur = Control(name, cur.kind, raw)
        out[name] = cur
    return out


# ------------------------------------------------------------------ expression eval
class ShaderUnsupported(Exception):
    """The shader uses GLSL beyond the supported subset; caller falls back to invlerp."""


def _as_rgb(v) -> list:
    """Broadcast a scalar/array to a 3-component list (GLSL vec3 promotion)."""
    if isinstance(v, list):
        if len(v) == 1:
            return [v[0]] * 3
        return v[:3]
    return [v, v, v]


def _clamp(x, lo=0.0, hi=1.0):
    if isinstance(x, list):
        return [_clamp(c, lo, hi) for c in x]
    return np.clip(x, lo, hi)


def _colormap_jet(x):
    x = np.asarray(x, dtype=np.float64)
    r = np.where(x < 0.89, (x - 0.35) / 0.31, 1.0 - (x - 0.89) / 0.11 * 0.5)
    g = np.where(x < 0.64, (x - 0.125) * 4.0, 1.0 - (x - 0.64) / 0.27)
    b = np.where(x < 0.34, 0.5 + x * 0.5 / 0.11, 1.0 - (x - 0.34) / 0.31)
    return [np.clip(c, 0.0, 1.0) for c in (r, g, b)]


def _colormap_cubehelix(x):
    xc = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    angle = 2.0 * math.pi * (4.0 / 3.0 + xc)
    amp = xc * (1.0 - xc) / 2.0
    ca, sa = np.cos(angle), np.sin(angle)
    rgb = (-0.14861 * ca + 1.78277 * sa,
           -0.29227 * ca + -0.90649 * sa,
           1.97294 * ca)
    return [np.clip(xc + amp * c, 0.0, 1.0) for c in rgb]


def _mix(a, b, t):
    if isinstance(a, list) or isinstance(b, list) or isinstance(t, list):
        a3, b3, t3 = _as_rgb(a), _as_rgb(b), _as_rgb(t)
        return [a3[i] * (1 - t3[i]) + b3[i] * t3[i] for i in range(3)]
    return a * (1 - t) + b * t


def _smoothstep(e0, e1, x):
    """GLSL smoothstep, component-wise. Edges are applied for vector x too — an earlier
    version passed x straight through when it was a vec, silently ignoring e0/e1."""
    def one(a, b, v):
        t = np.clip((v - a) / ((b - a) or 1e-12), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    if any(isinstance(v, list) for v in (e0, e1, x)):
        a3, b3, x3 = _as_rgb(e0), _as_rgb(e1), _as_rgb(x)
        return [one(a3[i], b3[i], x3[i]) for i in range(3)]
    return one(e0, e1, x)


def _elementwise(fn):
    def wrapped(*a):
        if any(isinstance(x, list) for x in a):
            promoted = [_as_rgb(x) for x in a]
            return [fn(*[p[i] for p in promoted]) for i in range(3)]
        return fn(*a)
    return wrapped


_FUNCS = {
    "clamp": _clamp,
    "min": _elementwise(np.minimum), "max": _elementwise(np.maximum),
    "pow": _elementwise(lambda a, b: np.power(np.maximum(a, 0.0), b)),
    "exp": _elementwise(np.exp), "log": _elementwise(lambda a: np.log(np.maximum(a, 1e-12))),
    "sqrt": _elementwise(lambda a: np.sqrt(np.maximum(a, 0.0))),
    "abs": _elementwise(np.abs), "floor": _elementwise(np.floor),
    "ceil": _elementwise(np.ceil), "sin": _elementwise(np.sin), "cos": _elementwise(np.cos),
    "step": _elementwise(lambda e, x: np.where(x < e, 0.0, 1.0)),
    "mix": _mix, "smoothstep": _smoothstep,
    "colormapJet": _colormap_jet, "colormapCubehelix": _colormap_cubehelix,
}

_TOKEN = re.compile(r"\s*(\d*\.?\d+(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\S)")


class _Expr:
    """Tiny recursive-descent evaluator for the GLSL expression subset that appears in
    real neuroglancer shaders. Operates on numpy arrays so one pass colors the slice."""

    def __init__(self, text: str, env: dict):
        self.toks = _TOKEN.findall(text)
        self.i = 0
        self.env = env

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, t):
        if self.next() != t:
            raise ShaderUnsupported(f"expected {t!r}")

    def parse(self):
        v = self.add()
        if self.peek() is not None:
            raise ShaderUnsupported(f"trailing token {self.peek()!r}")
        return v

    def add(self):
        v = self.mul()
        while self.peek() in ("+", "-"):
            op = self.next()
            r = self.mul()
            v = self._bin(v, r, op)
        return v

    def mul(self):
        v = self.unary()
        while self.peek() in ("*", "/"):
            op = self.next()
            r = self.unary()
            v = self._bin(v, r, op)
        return v

    def unary(self):
        if self.peek() == "-":
            self.next()
            v = self.unary()
            return [-c for c in v] if isinstance(v, list) else -v
        if self.peek() == "+":
            self.next()
        return self.postfix()

    def postfix(self):
        v = self.atom()
        while self.peek() == ".":          # swizzle: .r .rgb .x .xyz
            self.next()
            sw = self.next() or ""
            idx = {"r": 0, "g": 1, "b": 2, "a": 3, "x": 0, "y": 1, "z": 2, "w": 3}
            if not all(c in idx for c in sw):
                raise ShaderUnsupported(f"swizzle {sw!r}")
            src = _as_rgb(v) if not isinstance(v, list) else v
            picked = [src[idx[c]] if idx[c] < len(src) else src[-1] for c in sw]
            v = picked[0] if len(picked) == 1 else picked
        return v

    def _bin(self, a, b, op):
        f = (lambda x, y: x + y) if op == "+" else \
            (lambda x, y: x - y) if op == "-" else \
            (lambda x, y: x * y) if op == "*" else \
            (lambda x, y: x / y)
        if isinstance(a, list) or isinstance(b, list):
            a3, b3 = _as_rgb(a), _as_rgb(b)
            n = max(len(a3), len(b3))
            a3 = a3 + [a3[-1]] * (n - len(a3))
            b3 = b3 + [b3[-1]] * (n - len(b3))
            return [f(a3[i], b3[i]) for i in range(n)]
        return f(a, b)

    def atom(self):
        t = self.next()
        if t is None:
            raise ShaderUnsupported("unexpected end of expression")
        if t == "(":
            v = self.add()
            self.expect(")")
            return v
        try:
            return float(t)
        except ValueError:
            pass
        if not re.match(r"^[A-Za-z_]", t):
            raise ShaderUnsupported(f"token {t!r}")
        if self.peek() == "(":                      # call
            self.next()
            args = []
            if self.peek() != ")":
                args.append(self.add())
                while self.peek() == ",":
                    self.next()
                    args.append(self.add())
            self.expect(")")
            return self._call(t, args)
        if t in self.env:
            return self.env[t]
        raise ShaderUnsupported(f"unknown identifier {t!r}")

    def _call(self, name, args):
        if name in ("vec3", "vec4"):
            n = 3 if name == "vec3" else 4
            if len(args) == 1:
                return _as_rgb(args[0])[:n] if n == 3 else _as_rgb(args[0]) + [1.0]
            flat: list = []
            for a in args:
                flat.extend(a if isinstance(a, list) else [a])
            return flat[:n]
        if name == "float":
            return args[0]
        if name in _FUNCS:
            return _FUNCS[name](*args)
        if name in self.env:                        # a control invoked as `normalized()`
            return self.env[name]
        raise ShaderUnsupported(f"unsupported function {name!r}")


# --------------------------------------------------------------------------- shader
@dataclass
class LayerShader:
    """A neuroglancer layer's resolved shader state, ready to bake onto voxel data."""

    source: str = ""
    controls: dict[str, Control] = field(default_factory=dict)
    opacity: float = 1.0
    warning: str = ""

    @property
    def primary_range(self) -> tuple[float, float] | None:
        """The first invlerp range — the contrast window the UI shows for this layer."""
        for c in self.controls.values():
            if c.kind == "invlerp":
                return c.value
        return None

    def describe(self) -> str:
        r = self.primary_range
        return f"range={r[0]:g}..{r[1]:g}" if r else "no invlerp"

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the shader over `data`, returning uint8 RGB (H, W, 3).

        The result holds display-referred sRGB values, exactly what neuroglancer writes
        to its canvas — so the consumer must load it as sRGB, not Non-Color.
        """
        arr = np.asarray(data)
        env: dict[str, Any] = {}
        for name, c in self.controls.items():
            if c.kind == "invlerp":
                lo, hi = c.value
                span = (hi - lo) or 1.0
                v = (arr.astype(np.float64) - lo) / span
                env[name] = np.clip(v, 0.0, 1.0) if c.clamp else v
            elif c.kind == "color":
                env[name] = [float(x) for x in c.value]
            elif c.kind == "checkbox":
                env[name] = 1.0 if c.value else 0.0
            elif c.kind == "slider":
                env[name] = float(c.value)
            else:
                env[name] = c.value

        rgb = self._emit(env, arr)
        chans = [np.broadcast_to(np.asarray(c, dtype=np.float64), arr.shape) for c in rgb]
        out = np.stack(chans, axis=-1)
        return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def _emit(self, env: dict, arr: np.ndarray) -> list:
        """Find the emit* call in main() and evaluate its argument.

        Only a shader with exactly ONE emit is safe to evaluate this way. A branching
        shader (`if (x) emitTransparent(); else emitGrayscale(v);`) has per-pixel control
        flow we do not model, and silently taking the first emit would bake, say, an
        all-black slice while reporting success. Refuse instead and let the caller fall
        back with a warning.
        """
        _, body = parse_directives(self.source, arr.dtype) if self.source else ({}, "")
        emits = list(re.finditer(r"\bemit(Grayscale|RGBA|RGB|Transparent)\s*\(", body))
        if not emits:
            raise ShaderUnsupported("no emit* call found")
        if len(emits) > 1:
            raise ShaderUnsupported(
                f"{len(emits)} emit* calls (conditional shader not supported)")
        m = emits[0]
        kind = m.group(1)
        if kind == "Transparent":
            return [0.0, 0.0, 0.0]
        # take the balanced argument list
        start = m.end()
        depth, j = 1, start
        while j < len(body) and depth:
            depth += (body[j] == "(") - (body[j] == ")")
            j += 1
        arg = body[start:j - 1]
        val = _Expr(arg, env).parse()
        if kind == "Grayscale":
            g = val[0] if isinstance(val, list) else val
            return [g, g, g]
        return _as_rgb(val)


def _fallback(dtype, opacity: float, reason: str) -> LayerShader:
    """Default-invlerp behaviour, tagged with why we couldn't use the real shader."""
    lo, hi = default_range_for(dtype)
    return LayerShader(source=DEFAULT_IMAGE_SHADER,
                       controls={"normalized": Control("normalized", "invlerp", (lo, hi))},
                       opacity=opacity, warning=reason)


def from_layer(layer: dict, dtype=np.uint8) -> LayerShader:
    """Build the resolved shader state for ANY neuroglancer layer dict.

    Reads `shader`, `shaderControls` and `opacity`. A layer with no shader gets
    neuroglancer's default (plain invlerp over the dtype range), which is exactly what
    neuroglancer itself would show.
    """
    op = layer.get("opacity", 1.0)
    opacity = float(op) if op is not None else 1.0
    src = layer.get("shader") or DEFAULT_IMAGE_SHADER
    try:
        declared, _ = parse_directives(src, dtype)
    except Exception as e:  # noqa: BLE001 — a malformed shader must not kill the render
        return _fallback(dtype, opacity, f"could not parse shader ({e})")
    controls = apply_shader_controls(declared, layer.get("shaderControls"), dtype)
    if not any(c.kind == "invlerp" for c in controls.values()):
        # No invlerp at all (e.g. a pure constant-color shader). Still valid — evaluation
        # will either work or fall back at apply() time.
        pass
    return LayerShader(source=src, controls=controls, opacity=opacity)


def shade(layer_shader: LayerShader, data: np.ndarray) -> tuple[np.ndarray, str]:
    """Apply a shader, falling back to plain invlerp if the GLSL is out of scope.

    Returns (uint8 RGB, warning) — warning is "" when the real shader was used.
    """
    try:
        return layer_shader.apply(data), layer_shader.warning
    except ShaderUnsupported as e:
        fb = _fallback(data.dtype, layer_shader.opacity, str(e))
        # Keep the invlerp range the state DID give us, so contrast still tracks the
        # slider even though the rest of the shader was out of scope. The control must be
        # re-keyed to "normalized": the fallback SOURCE is neuroglancer's default shader,
        # which calls normalized(). Installing it under a custom name (a shader is free to
        # call its control anything) would leave normalized() undefined and raise straight
        # back out of the fallback — dropping the slice from every frame.
        for c in layer_shader.controls.values():
            if c.kind == "invlerp":
                fb.controls = {"normalized": Control("normalized", "invlerp",
                                                     c.value, clamp=c.clamp)}
                break
        return fb.apply(data), f"unsupported shader, using invlerp only: {e}"
