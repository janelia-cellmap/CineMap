"""Claude agent — the hybrid director from plan.md.

Claude calls structured tools that mutate the project through the same
operations/scouting funnel the UI uses, so agent edits and manual edits compose
on one project.json. Tool definitions + a static system prompt are prompt-cached.
"""
from __future__ import annotations

import json
import os
from typing import Callable

from . import operations as ops
from . import scouting, store
from .models import Project, RenderSettings

MODEL = os.environ.get("CINEMAP_AGENT_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 2048
MAX_STEPS = 10

SYSTEM = """You are the directing assistant inside CineMap, a tool that makes \
videos of large EM (electron-microscopy) datasets and their segmentation meshes.

How a video works here:
- A video is an ordered list of KEYFRAMES. Each keyframe is a scene state: a \
camera, EM cross-section SLICE planes, and segmentation MESHES (chosen segment \
ids per layer). The video interpolates between consecutive keyframes; a keyframe's \
`duration_in_s` is the transition time INTO it.
- The user scouts the data in an embedded Neuroglancer viewer. "Bake" captures \
the current 3D view (camera + visible layers/segments) as a keyframe.
- Meshes are generated from the label volume; many segments render as one colored \
field. The EM slice gives spatial context.

Your job: translate plain requests ("orbit the nuclei", "sweep a slice through, \
then zoom to a few cells") into tool calls that build/edit the keyframe timeline. \
Call `get_state` first when you need to know the current scene. Prefer the preset \
shot tools (make_orbit, sweep_slice) for motion. After editing, briefly tell the \
user what you changed and that they can Preview/Export (or use the render tool). \
Keep replies short. Keyframes are referenced by their integer index from get_state."""

TOOLS = [
    {"name": "get_state", "description": "Return the dataset manifest (EM + mesh layers, available segments) and the current keyframes (index, label, camera, slices, meshes). Call this first to understand the scene.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "make_orbit", "description": "Append keyframes orbiting the current camera target (preserves framing). Use for 'rotate/spin around'.",
     "input_schema": {"type": "object", "properties": {
         "degrees": {"type": "number", "default": 360}, "n": {"type": "integer", "default": 12},
         "elevation_deg": {"type": "number", "default": 22}}}},
    {"name": "sweep_slice", "description": "Append keyframes sweeping an EM cross-section slice through the volume along an axis.",
     "input_schema": {"type": "object", "properties": {
         "axis": {"type": "string", "enum": ["x", "y", "z"], "default": "z"},
         "n": {"type": "integer", "default": 12}}}},
    {"name": "bake_keyframe", "description": "Capture the current Neuroglancer 3D view (camera + visible layers/segments) as a new keyframe.",
     "input_schema": {"type": "object", "properties": {"label": {"type": "string"}}}},
    {"name": "duplicate_last_keyframe", "description": "Append a copy of the last keyframe (to then tweak).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "delete_keyframe", "description": "Delete the keyframe at the given index.",
     "input_schema": {"type": "object", "properties": {"index": {"type": "integer"}}, "required": ["index"]}},
    {"name": "reorder_keyframes", "description": "Reorder all keyframes; give the new order as a list of current indices.",
     "input_schema": {"type": "object", "properties": {"order": {"type": "array", "items": {"type": "integer"}}}, "required": ["order"]}},
    {"name": "set_keyframe", "description": "Edit a keyframe's transition timing.",
     "input_schema": {"type": "object", "properties": {
         "index": {"type": "integer"}, "duration_in_s": {"type": "number"},
         "easing": {"type": "string", "enum": ["linear", "ease-in-out"]}}, "required": ["index"]}},
    {"name": "set_keyframe_segments", "description": "Set which segment ids of a mesh layer are shown in a keyframe.",
     "input_schema": {"type": "object", "properties": {
         "index": {"type": "integer"}, "mesh_name": {"type": "string"},
         "segment_ids": {"type": "array", "items": {"type": "integer"}}}, "required": ["index", "mesh_name", "segment_ids"]}},
    {"name": "set_mesh_opacity", "description": "Set the 3D mesh opacity for a keyframe (0..1). Lower it (e.g. 0) so the EM slice and its segmentation overlay are visible without the 3D meshes occluding them.",
     "input_schema": {"type": "object", "properties": {
         "index": {"type": "integer"}, "opacity": {"type": "number"}}, "required": ["index", "opacity"]}},
    {"name": "set_slice", "description": "Set a keyframe's EM slice axis/position/visibility.",
     "input_schema": {"type": "object", "properties": {
         "index": {"type": "integer"}, "axis": {"type": "string", "enum": ["x", "y", "z"]},
         "position_nm": {"type": "number"}, "visible": {"type": "boolean"}}, "required": ["index"]}},
    {"name": "goto_keyframe", "description": "Navigate the Neuroglancer viewer to a keyframe's view.",
     "input_schema": {"type": "object", "properties": {"index": {"type": "integer"}}, "required": ["index"]}},
    {"name": "render", "description": "Render the timeline to a video. Returns a job id; the UI shows progress.",
     "input_schema": {"type": "object", "properties": {
         "width": {"type": "integer", "default": 1280}, "height": {"type": "integer", "default": 720},
         "fps": {"type": "integer", "default": 30}, "samples": {"type": "integer", "default": 48}}}},
]


def _kf_summary(p: Project) -> list[dict]:
    out = []
    for i, k in enumerate(p.keyframes):
        out.append({
            "index": i, "label": k.label, "duration_in_s": k.duration_in_s,
            "look_at_nm": [round(x) for x in k.camera.look_at_nm],
            "slices": [{"axis": s.axis, "position_nm": round(s.position_nm), "visible": s.visible} for s in k.slices],
            "meshes": [{"layer": m.mesh_name, "n_segments": len(m.segment_ids), "visible": m.visible} for m in k.meshes],
        })
    return out


def _state(p: Project) -> dict:
    man = p.manifest
    return {
        "dataset": man.title,
        "em_layer": man.em.name if man.em else None,
        "mesh_layers": [{"name": m.name, "n_selected": len(m.segment_ids)} for m in man.meshes],
        "voxel_size_nm": man.voxel_size_nm,
        "keyframes": _kf_summary(p),
    }


def _kid(p: Project, index: int) -> str:
    if not (0 <= index < len(p.keyframes)):
        raise ValueError(f"keyframe index {index} out of range (0..{len(p.keyframes) - 1})")
    return p.keyframes[index].id


def _dispatch(name: str, args: dict, p: Project, render_fn: Callable | None) -> dict:
    if name == "get_state":
        return _state(p)
    if name == "make_orbit":
        kfs = ops.make_orbit(p, degrees=args.get("degrees", 360), n=args.get("n", 12),
                             elevation_deg=args.get("elevation_deg", 22))
        return {"added": len(kfs)}
    if name == "sweep_slice":
        kfs = ops.sweep_slice(p, axis=args.get("axis", "z"), n=args.get("n", 12))
        return {"added": len(kfs)}
    if name == "bake_keyframe":
        kf = scouting.bake_keyframe(p, label=args.get("label", "scouted"))
        return {"added": 1, "index": len(p.keyframes) - 1, "meshes": [m.mesh_name for m in kf.meshes]}
    if name == "duplicate_last_keyframe":
        ops.add_keyframe(p)
        return {"added": 1, "index": len(p.keyframes) - 1}
    if name == "delete_keyframe":
        ops.delete_keyframe(p, _kid(p, args["index"]))
        return {"deleted": args["index"]}
    if name == "reorder_keyframes":
        ids = [p.keyframes[i].id for i in args["order"]]
        ops.reorder_keyframes(p, ids)
        return {"ok": True}
    if name == "set_keyframe":
        fields = {k: args[k] for k in ("duration_in_s", "easing") if k in args}
        ops.update_keyframe(p, _kid(p, args["index"]), **fields)
        return {"ok": True}
    if name == "set_keyframe_segments":
        kf = p.keyframes[args["index"]]
        from .models import MeshInstance

        meshes = [m for m in kf.meshes if m.mesh_name != args["mesh_name"]]
        meshes.append(MeshInstance(mesh_name=args["mesh_name"], segment_ids=args["segment_ids"]))
        ops.update_keyframe(p, kf.id, meshes=meshes)
        return {"ok": True, "n_segments": len(args["segment_ids"])}
    if name == "set_mesh_opacity":
        kf = p.keyframes[args["index"]]
        op = max(0.0, min(1.0, args["opacity"]))
        meshes = [m.model_copy(update={"opacity": op, "visible": op > 0.001}) for m in kf.meshes]
        ops.update_keyframe(p, kf.id, meshes=meshes)
        return {"ok": True, "opacity": op}
    if name == "set_slice":
        kf = p.keyframes[args["index"]]
        from .models import SlicePlane

        s = kf.slices[0] if kf.slices else SlicePlane(em_name=p.manifest.em.name if p.manifest.em else "em")
        s = s.model_copy(update={k: args[k] for k in ("axis", "position_nm", "visible") if k in args})
        ops.update_keyframe(p, kf.id, slices=[s])
        return {"ok": True}
    if name == "goto_keyframe":
        scouting.goto_keyframe(p, _kid(p, args["index"]))
        return {"ok": True}
    if name == "render":
        if render_fn is None:
            return {"error": "rendering not available in this context"}
        job_id = render_fn(RenderSettings(width=args.get("width", 1280), height=args.get("height", 720),
                                          fps=args.get("fps", 30), samples=args.get("samples", 48)))
        return {"job_id": job_id, "note": "render started; progress shows in the UI"}
    return {"error": f"unknown tool {name}"}


def run_agent(project_id: str, message: str, history: list[dict] | None = None,
              render_fn: Callable | None = None) -> dict:
    """Run one chat turn. Returns {text, actions, render_job_id, error?}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"text": "The agent needs an Anthropic API key. Set ANTHROPIC_API_KEY "
                        "in the environment before launching CineMap (e.g. "
                        "`export ANTHROPIC_API_KEY=sk-...` then ./run.sh).",
                "actions": [], "render_job_id": None, "error": "no_api_key"}
    import anthropic

    client = anthropic.Anthropic()
    p = store.load(project_id)

    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": message})

    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    actions: list[dict] = []
    render_job_id = None

    for _ in range(MAX_STEPS):
        try:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=system,
                                          tools=TOOLS, messages=messages)
        except anthropic.AuthenticationError:
            return {"text": "That Anthropic API key was rejected. Please set a valid key.",
                    "actions": actions, "render_job_id": render_job_id, "error": "bad_api_key"}
        except anthropic.APIError as e:  # noqa: PERF203
            return {"text": f"Claude API error: {e}", "actions": actions,
                    "render_job_id": render_job_id, "error": "api_error"}
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out = _dispatch(block.name, block.input or {}, p, render_fn)
            except Exception as e:  # noqa: BLE001
                out = {"error": str(e)}
            if isinstance(out, dict) and out.get("job_id"):
                render_job_id = out["job_id"]
            actions.append({"tool": block.name, "input": block.input, "result": out})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})

    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return {"text": text, "actions": actions, "render_job_id": render_job_id}
