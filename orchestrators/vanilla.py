# orchestrators/vanilla.py
import os
import re
import json
import copy
import chainlit as cl
import asyncio
from typing import Any, Dict, List
from core.state import Task, ConversationState


def _collect_latest_segmentations(state: ConversationState) -> List[str]:
    """
    Return a clean list of existing segmentation file paths from memory.
    Supports:
      - state.memory["segmentations"]      
      - state.memory["segmentation"]      
      - state.memory["segmentations_map"]  
    """
    mem = state.memory or {}
    segs: List[str] = []

    if isinstance(mem.get("segmentations"), list):
        segs = mem["segmentations"]

    elif mem.get("segmentation") is not None:
        val = mem["segmentation"]
        segs = val if isinstance(val, list) else [val]

    elif isinstance(mem.get("segmentations_map"), dict):
        segs = list(mem["segmentations_map"].values())

    segs = [p for p in segs if isinstance(p, str) and os.path.exists(p)]
    return segs


def _ensure_image_path(child: Task, state: ConversationState, root_task: Task):
    if not child.kwargs.get("image_path"):
        ip = state.memory.get("image_path")
        if not ip and root_task.files:
            ip = root_task.files[0]
        if ip:
            child.kwargs["image_path"] = ip


_FULL_RE    = re.compile(r"^\$\{([^}]+)\}$")
_PARTIAL_RE = re.compile(r"\$\{([^}]+)\}")
_TOKEN_RE   = re.compile(r"([^. \[\]]+)|\[(\d+)\]")


def _get_from_ctx(ctx: Dict[str, Any], expr: str) -> Any:
    cur: Any = ctx
    for key, idx in _TOKEN_RE.findall(expr):
        if key:
            if isinstance(cur, (list, tuple)) and key.isdigit():
                cur = cur[int(key)]
            else:
                cur = cur[key]
        else:
            cur = cur[int(idx)]
    return cur


def _resolve_placeholders(value: Any, ctx: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v, ctx) for v in value]
    if isinstance(value, str):
        m = _FULL_RE.match(value)
        if m:
            try:
                return _get_from_ctx(ctx, m.group(1))
            except Exception:
                return None

        def repl(match):
            try:
                v = _get_from_ctx(ctx, match.group(1))
                return str(v)
            except Exception:
                return match.group(0)

        return _PARTIAL_RE.sub(repl, value)
    return value


def _route_to_plan(route: Dict[str, Any], next_tasks: List[Task]) -> List[Dict[str, Any]]:
    if isinstance(route, dict) and isinstance(route.get("plan"), list):
        plan = []
        for i, step in enumerate(route["plan"]):
            agent = step.get("agent")
            if not agent:
                continue
            plan.append({
                "id": step.get("id") or f"{agent}_{i+1}",
                "agent": agent,
                "kwargs": copy.deepcopy(step.get("kwargs", {}))
            })
        if plan:
            return plan

    return []


async def run_pipeline(router, agents, task: Task, state: ConversationState):

    async with cl.Step(name="Routing") as step:
        router_res = await router.run(task, state)
        route_dict = router_res.output if isinstance(router_res.output, dict) else {}
        plan = _route_to_plan(route_dict, getattr(router_res, "next_tasks", []))
        step.output = f"**Plan**:\n```json\n{json.dumps({'plan': plan}, indent=2)}\n```"
        await cl.Message(content="Plan created!").send()
        #await cl.Message(content=f"**Plan**:\n```json\n{json.dumps({'plan': plan}, indent=2)}\n```").send()

    # Build a context available to placeholders
    ctx: Dict[str, Any] = {
        "input": {"files": task.files, "kwargs": task.kwargs},
        "state": state.memory,
    }

    result_payload = None

    # Execute steps in order
    for idx, stage in enumerate(plan):
        step_id  = stage.get("id") or f"step{idx+1}"
        agent_nm = stage["agent"]
        raw_kwargs = stage.get("kwargs", {})

        raw_kwargs.setdefault("input_files",  task.files or [])
        raw_kwargs.setdefault("chained_files", state.memory.get("files", []))

        kwargs = _resolve_placeholders(raw_kwargs, ctx)
        if agent_nm == "viz_slider":
            img_candidate = kwargs.get("image_path")
            print(img_candidate)
            unresolved = isinstance(img_candidate, str) and "${" in img_candidate
            if img_candidate is None or unresolved:
                tmp_child = Task(user_msg="", files=task.files, kwargs=kwargs)
                _ensure_image_path(tmp_child, state, task)  
                kwargs = tmp_child.kwargs 
                print(kwargs)  

        if agent_nm == "viz_slider" and kwargs.pop("use_latest_segmentations", False):
            segs = _collect_latest_segmentations(state)
            if segs:
                kwargs.setdefault("mask_paths", segs)
                kwargs.setdefault("mask_path", segs[0])
            tmp_child = Task(user_msg="", files=task.files, kwargs=kwargs)
            _ensure_image_path(tmp_child, state, task)
            kwargs = tmp_child.kwargs  

        if agent_nm == "viz_slider":
            print("VIZ_SLIDER kwargs:", kwargs)

        child = Task(user_msg=task.user_msg, files=task.files, intent=step_id, kwargs=kwargs)
        agent = agents[agent_nm]

        is_llm_agent = hasattr(agent, 'model') and agent.model is not None

        async with cl.Step(name=f"{agent_nm} ({idx+1}/{len(plan)})") as step:
            if is_llm_agent:
                step.output = "**LLM Agent** - Generating and executing code...\n\n```python\n"
            else:
                step.output = f"🔧 **{agent_nm}** - Processing...\n\n"
            await step.update()
                
            if is_llm_agent:    
                try:
                    res = await agent.run(child, state)
                    
                    if hasattr(res, 'artifacts') and res.artifacts and 'code' in res.artifacts:
                        code = res.artifacts['code']
                        # Add code to step output
                        step.output += code + "\n```\n\n**LLM Agent completed successfully!**"
                    else:
                        step.output += "\n```\n\n**LLM Agent completed successfully!**"
                    
                except Exception as e:
                    step.output += f"\n\n**{agent_nm}** failed: {str(e)}"
                    raise
                    
            else:
                if agent_nm == "universeg":
                    step.output += "Loading UniverSeg model...\n"
                    step.output += "Running few-shot segmentation...\n"
                            
                try:
                    res = await agent.run(child, state)
                    step.output += f"**{agent_nm}** completed successfully!"
                        
                except Exception as e:
                    step.output += f"\n\n**{agent_nm}** failed: {str(e)}"
                    raise

            result_payload = res.output

            ctx[step_id] = {
                "output": res.output,
                "artifacts": res.artifacts,
                "debug": getattr(res, "debug", None),
            }

            arts = res.artifacts or {}

            if agent_nm == "code_exec" or agent_nm == "monai":
                print(arts["code"])

            if "segmentations" in arts:
                state.memory["segmentations"] = arts["segmentations"]
            if "segmentations_map" in arts:
                state.memory["segmentations_map"] = arts["segmentations_map"]
            if "output_dir" in arts:
                state.memory["last_output_dir"] = arts["output_dir"]
            if "image_path" in kwargs:
                state.memory["image_path"] = kwargs["image_path"]

            if hasattr(result_payload, "to_csv"):
                state.memory["last_df"] = result_payload

            #msg.content = f"**{agent_nm}** done."
            #await msg.update()

    return result_payload
