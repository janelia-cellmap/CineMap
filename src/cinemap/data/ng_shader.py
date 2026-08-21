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


def _veclen(v) -> int:
    return len(v) if isinstance(v, list) else 1


def _promote(args: tuple, n: int) -> list[list]:
    """GLSL component-wise promotion: a scalar broadcasts to n components, a vector of
    length n passes through. Anything else is a size mismatch we refuse rather than pad.

    The previous version padded a short vector with its LAST component, so `vec2 * vec3`
    silently produced a plausible-looking wrong answer instead of falling back.
    """
    out = []
    for a in args:
        if not isinstance(a, list):
            out.append([a] * n)
        elif len(a) == 1:
            out.append([a[0]] * n)
        elif len(a) == n:
            out.append(list(a))
        else:
            raise ShaderUnsupported(f"cannot combine vec{len(a)} with vec{n}")
    return out


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
    n = max(_veclen(a), _veclen(b), _veclen(t))
    if n == 1:
        return a * (1 - t) + b * t
    av, bv, tv = _promote((a, b, t), n)
    return [av[i] * (1 - tv[i]) + bv[i] * tv[i] for i in range(n)]


def _smoothstep(e0, e1, x):
    """GLSL smoothstep, component-wise. Edges are applied for vector x too — an earlier
    version passed x straight through when it was a vec, silently ignoring e0/e1."""
    def one(a, b, v):
        t = np.clip((v - a) / ((b - a) or 1e-12), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    n = max(_veclen(e0), _veclen(e1), _veclen(x))
    if n == 1:
        return one(e0, e1, x)
    av, bv, xv = _promote((e0, e1, x), n)
    return [one(av[i], bv[i], xv[i]) for i in range(n)]


def _elementwise(fn):
    def wrapped(*a):
        n = max((_veclen(x) for x in a), default=1)
        if n == 1:
            return fn(*a)
        promoted = _promote(tuple(a), n)
        return [fn(*[p[i] for p in promoted]) for i in range(n)]
    return wrapped


def _dot(a, b):
    """GLSL dot(). Returns a SCALAR, so it cannot go through _elementwise."""
    n = max(_veclen(a), _veclen(b))
    av, bv = _promote((a, b), n)
    acc = av[0] * bv[0]
    for i in range(1, n):
        acc = acc + av[i] * bv[i]
    return acc


def _length(v):
    return np.sqrt(_dot(v, v))


def _normalize(v):
    n = _length(v)
    return [c / n for c in v] if isinstance(v, list) else v / n


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
    "sign": _elementwise(np.sign), "fract": _elementwise(lambda a: a - np.floor(a)),
    "mod": _elementwise(lambda a, b: np.mod(a, b)),
    "tan": _elementwise(np.tan), "atan": _elementwise(np.arctan2),
    "exp2": _elementwise(np.exp2),
    "log2": _elementwise(lambda a: np.log2(np.maximum(a, 1e-12))),
    "inversesqrt": _elementwise(lambda a: 1.0 / np.sqrt(np.maximum(a, 1e-12))),
    "dot": _dot, "length": _length, "normalize": _normalize,
    "distance": lambda a, b: _length([x - y for x, y in zip(_as_rgb(a), _as_rgb(b))]),
    "colormapJet": _colormap_jet, "colormapCubehelix": _colormap_cubehelix,
}

_TOKEN = re.compile(r"\s*(\d*\.?\d+(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\S)")


class _Expr:
    """Tiny recursive-descent evaluator for the GLSL expression subset that appears in
    real neuroglancer shaders. Operates on numpy arrays so one pass colors the slice."""

    def __init__(self, text: str, env: dict, prog: "_Program | None" = None):
        self.toks = _TOKEN.findall(text)
        self.i = 0
        self.env = env
        self.prog = prog

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
            idx = {"r": 0, "g": 1, "b": 2, "a": 3, "x": 0, "y": 1, "z": 2, "w": 3,
                   "s": 0, "t": 1, "p": 2, "q": 3}
            if not all(c in idx for c in sw):
                raise ShaderUnsupported(f"swizzle {sw!r}")
            src = _as_rgb(v) if not isinstance(v, list) else v
            # Out of range means we misread the vector's width; clamping to the last
            # component (the old behavior) would invent a value and render it as fact.
            if any(idx[c] >= len(src) for c in sw):
                raise ShaderUnsupported(f".{sw} on a vec{len(src)}")
            picked = [src[idx[c]] for c in sw]
            v = picked[0] if len(picked) == 1 else picked
        return v

    def _bin(self, a, b, op):
        f = (lambda x, y: x + y) if op == "+" else \
            (lambda x, y: x - y) if op == "-" else \
            (lambda x, y: x * y) if op == "*" else \
            (lambda x, y: x / y)
        n = max(_veclen(a), _veclen(b))
        if n == 1:
            return f(a, b)
        # Arbitrary width, not vec3: a vec4 of polynomial coefficients times a vec4 of
        # powers is ordinary GLSL, and truncating either to 3 components (what _as_rgb
        # did here) drops a term and quietly shifts every colour.
        av, bv = _promote((a, b), n)
        return [f(av[i], bv[i]) for i in range(n)]

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
        if name in ("vec2", "vec3", "vec4"):
            n = {"vec2": 2, "vec3": 3, "vec4": 4}[name]
            if len(args) == 1:
                a = args[0]
                if not isinstance(a, list):         # vecN(scalar) -> all components equal
                    return [a] * n
                if len(a) >= n:                     # vec3(someVec4) truncates
                    return list(a[:n])
                raise ShaderUnsupported(f"{name}(vec{len(a)})")
            flat: list = []
            for a in args:
                flat.extend(a if isinstance(a, list) else [a])
            if len(flat) < n:
                raise ShaderUnsupported(f"{name} given {len(flat)} components")
            return flat[:n]
        if name in ("float", "int", "uint"):
            return args[0]
        if name in _FUNCS:
            return _FUNCS[name](*args)
        if self.prog is not None and name in self.prog.funcs:
            return self.prog.call(name, args)       # a function the shader itself defines
        if name in self.env:                        # a control invoked as `normalized()`
            return self.env[name]
        raise ShaderUnsupported(f"unsupported function {name!r}")


# --------------------------------------------------------------------------- shader
# ------------------------------------------------------------------ statement eval
_TYPES = r"(?:void|bool|int|uint|float|vec2|vec3|vec4|ivec2|ivec3|ivec4|mat2|mat3|mat4)"
_FUNC_HEAD = re.compile(rf"\b{_TYPES}\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{{")
_DECL = re.compile(rf"^(?:const\s+)?{_TYPES}\s+([A-Za-z_]\w*)\s*=\s*(.+)$", re.S)
_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", re.S)
_RETURN = re.compile(r"^return\b\s*(.*)$", re.S)
_EMIT_CALL = re.compile(r"^emit(Grayscale|RGBA|RGB|Transparent)\s*\((.*)\)$", re.S)
_CONTROL_FLOW = re.compile(r"\b(if|else|for|while|do|switch|discard)\b")
_MAX_CALL_DEPTH = 16


@dataclass
class _Func:
    """A function the shader itself defines. Types are dropped: values are numpy arrays
    or python floats and GLSL's static types add nothing we can act on."""

    name: str
    params: list[str]
    body: str


def _match_brace(text: str, open_idx: int) -> int:
    """Index just past the `}` matching the `{` at `open_idx`."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ShaderUnsupported("unbalanced braces")


def _split_statements(body: str) -> list[str]:
    """Top-level `;`-separated statements, ignoring separators nested in (), [] or {}."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == ";" and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return [st for st in out if st]


class _Program:
    """A parsed shader: the functions it defines plus the body of `main`.

    Exists because real neuroglancer shaders are not single expressions. The cellmap
    skeleton shaders, for instance, define their own `turbo()` colormap, declare `const`
    coefficient vectors, compute intermediate locals, and only then call `emitRGB`.
    Matching just a handful of shader TEMPLATES (what the old skeleton colormap parser
    did) silently drops the color for anything else; interpreting the GLSL means a layer
    we have never seen still renders in the viewer's colors.

    Control flow is deliberately NOT supported: per-pixel branching would need a masked
    evaluator, and guessing a branch would bake a confidently wrong image. Such a shader
    raises ShaderUnsupported and the caller falls back with a visible warning.
    """

    def __init__(self, code: str):
        self.funcs: dict[str, _Func] = {}
        self.main: str = ""
        self._parse(code)

    def _parse(self, code: str) -> None:
        pos = 0
        leftover: list[str] = []
        while True:
            m = _FUNC_HEAD.search(code, pos)
            if m is None:
                leftover.append(code[pos:])
                break
            leftover.append(code[pos:m.start()])
            end = _match_brace(code, m.end() - 1)
            body = code[m.end():end - 1]
            name = m.group(1)
            params = []
            for raw in m.group(2).split(","):
                raw = raw.strip()
                if not raw or raw == "void":
                    continue
                params.append(raw.split()[-1])       # "in float x" -> "x"
            if name == "main":
                self.main = body
            else:
                self.funcs[name] = _Func(name, params, body)
            pos = end
        # Anything outside a function body is a global declaration (`const vec3 k = ...`).
        self.globals = "".join(leftover)

    def call(self, name: str, args: list, depth: int = 0):
        if depth >= _MAX_CALL_DEPTH:
            raise ShaderUnsupported(f"call depth exceeded at {name!r}")
        fn = self.funcs[name]
        if len(args) != len(fn.params):
            raise ShaderUnsupported(
                f"{name} takes {len(fn.params)} args, got {len(args)}")
        env = dict(self.base_env)
        env.update(dict(zip(fn.params, args)))
        kind, value = self._run(fn.body, env, depth + 1)
        if kind != "return":
            raise ShaderUnsupported(f"{name} returned nothing")
        return value

    def _run(self, body: str, env: dict, depth: int) -> tuple[str, object]:
        """Execute statements, returning ('return', v), ('emit', (kind, v)) or ('', None)."""
        if _CONTROL_FLOW.search(body):
            raise ShaderUnsupported("control flow (if/for/while) is not supported")
        for st in _split_statements(body):
            m = _RETURN.match(st)
            if m:
                return "return", self._eval(m.group(1), env, depth)
            m = _EMIT_CALL.match(st)
            if m:
                return "emit", (m.group(1), m.group(2))
            m = _DECL.match(st)
            if m:
                env[m.group(1)] = self._eval(m.group(2), env, depth)
                continue
            m = _ASSIGN.match(st)
            if m and m.group(1) in env:
                env[m.group(1)] = self._eval(m.group(2), env, depth)
                continue
            raise ShaderUnsupported(f"unsupported statement {st.strip()[:60]!r}")
        return "", None

    def _eval(self, expr: str, env: dict, depth: int):
        return _Expr(expr, env, _ProgramAtDepth(self, depth)).parse()

    def emit(self, env: dict) -> tuple[str, object]:
        """Run `main` and return (emit kind, evaluated argument)."""
        self.base_env = dict(env)
        genv = dict(env)
        if self.globals.strip():
            self._run(self.globals, genv, 0)         # const/global declarations
            self.base_env = dict(genv)
        kind, payload = self._run(self.main, genv, 0)
        if kind != "emit":
            raise ShaderUnsupported("main() has no emit* call")
        emit_kind, arg = payload
        if emit_kind == "Transparent":
            return emit_kind, [0.0, 0.0, 0.0]
        return emit_kind, self._eval(arg, genv, 0)


class _ProgramAtDepth:
    """Threads the current call depth into nested `_Expr` evaluation, so a shader that
    calls itself hits the depth limit instead of recursing until Python dies."""

    def __init__(self, prog: _Program, depth: int):
        self._prog = prog
        self._depth = depth

    @property
    def funcs(self):
        return self._prog.funcs

    def call(self, name, args):
        return self._prog.call(name, args, self._depth)


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
        """Run main() and return the emitted RGB as a 3-list.

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
        kind, val = _Program(body).emit(env)
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


def effective_image_opacity(state: dict | None, layer: dict | None) -> float:
    """What a neuroglancer image layer's `opacity` ACTUALLY does to displayed pixels.

    Not `layer["opacity"]`. Neuroglancer disables GL blending entirely for the
    bottom-most image render layer when its blend mode is the default:

        // sliceview/volume/image_renderlayer.ts
        if (blendModeValue === BLEND_MODES.ADDITIVE || renderLayerNum > 0) {
          gl.enable(gl.BLEND); BLEND_FUNCTIONS.get(blendModeValue)!(gl);
        } else {
          gl.disable(WebGL2RenderingContext.BLEND);
        }

    So that layer's `uOpacity` reaches only the framebuffer's alpha channel and never
    modulates its RGB. The panel then composites with an if/else, not a blend:

        // sliceview/frontend.ts
        if (sampledColor.a == 0.0) { sampledColor = uBackgroundColor; }
        emit(sampledColor * uColorFactor, 0u);

    -- so the background shows only where alpha is exactly zero. Net effect for a single
    image layer: it renders at FULL strength for any opacity > 0, and vanishes at 0.

    This matters a lot here, because neuroglancer's image `opacity` DEFAULTS to 0.5
    (layer/image/index.ts:129). Feeding that value into Blender's alpha renders every EM
    slice at half strength while neuroglancer shows it whole -- the long-standing
    "EM looks much worse in Blender / we have to shift opacity to match" mismatch.

    Measured against a live viewer: opacity 0.25/0.5/0.75/1.0 all render an identical
    full-strength ramp, and only 0.0 shows the background.
    """
    layer = layer or {}
    raw = layer.get("opacity", 0.5)          # neuroglancer's default, not 1.0
    opacity = float(raw) if raw is not None else 0.5
    if opacity <= 0.0:
        return 0.0
    if str(layer.get("blend", "default")).lower() == "additive":
        return opacity                        # additive always blends, even at index 0
    if _is_bottom_image_layer(state, layer):
        return 1.0
    return opacity


def _is_bottom_image_layer(state: dict | None, layer: dict | None) -> bool:
    """True if `layer` is the first image layer neuroglancer draws (renderLayerNum 0).

    With no state to compare against we assume it is: CineMap bakes exactly one EM slice
    per keyframe, so the single-layer case is the norm, and it is the case where getting
    this wrong halves the EM.
    """
    if not state or not layer:
        return True
    name = layer.get("name")
    for l in state.get("layers", []) or []:
        if l.get("type") != "image" or l.get("archived"):
            continue
        if l.get("visible", True) is False:
            continue
        return l.get("name") == name
    return True


# ------------------------------------------------- property-driven (skeleton) shaders
_PROP_CALL = re.compile(r"\bprop_([A-Za-z_]\w*)\s*\(")


def shader_property_names(shader_src: str) -> set[str]:
    """The vertex properties a shader reads, e.g. `prop_radius()` -> {"radius"}.

    Neuroglancer exposes each of a skeleton's vertex attributes to its
    `skeletonRendering.shader` as a zero-argument `prop_<name>()` accessor.
    """
    return set(_PROP_CALL.findall(_strip_comments(shader_src or "")))


def shade_properties(shader_src: str, props: dict[str, np.ndarray],
                     shader_controls: dict | None = None) -> tuple[np.ndarray, str]:
    """Evaluate a property-driven shader (skeletons, annotations) over per-element values.

    `props` maps a bare property name to a 1-D array, one entry per element to colour.
    Returns (uint8 RGB of shape (N, 3), warning); the warning is "" on success.

    This replaces pattern-matching a handful of known shader shapes. The cellmap skeleton
    shaders define their own `turbo()` colormap over `log(prop_radius())`, which matched
    no template, so every such layer silently fell back to one flat per-segment colour --
    the "skeletons have no colour" bug. Interpreting the GLSL means a layer nobody
    anticipated still renders in the viewer's colours.
    """
    n = len(next(iter(props.values()))) if props else 0
    try:
        declared, body = parse_directives(shader_src, np.float32)
        controls = apply_shader_controls(declared, shader_controls, np.float32)
        env: dict[str, Any] = {}
        for name, c in controls.items():
            if c.kind == "color":
                env[name] = [float(x) for x in c.value]
            elif c.kind == "checkbox":
                env[name] = 1.0 if c.value else 0.0
            elif c.kind == "invlerp":
                env[name] = 0.0                    # no data channel here; keep it defined
            else:
                env[name] = float(c.value) if c.kind == "slider" else c.value
        for name, arr in props.items():
            env[f"prop_{name}"] = np.asarray(arr, dtype=np.float64)
        kind, val = _Program(body).emit(env)
        rgb = [val] * 3 if kind == "Grayscale" and not isinstance(val, list) else _as_rgb(val)
        chans = [np.broadcast_to(np.asarray(c, dtype=np.float64), (n,)) for c in rgb]
        out = np.stack(chans, axis=-1)
        return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), ""
    except Exception as e:  # noqa: BLE001 — a bad shader must never kill a render
        return np.zeros((n, 3), dtype=np.uint8), f"{type(e).__name__}: {e}"
