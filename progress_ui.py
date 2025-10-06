# progress_ui.py
import chainlit as cl

_KEY_EL = "voxel_progress_el"
_KEY_MSG = "voxel_progress_msg"

async def _ensure_el():
    el = cl.user_session.get(_KEY_EL)
    msg = cl.user_session.get(_KEY_MSG)
    if el is None or msg is None:
        el = cl.CustomElement(
            name="VoxelProgress",
            props={"pct": 0, "label": "Starting"},
            display="inline",
        )
        msg = cl.Message(content="", author="VoxelInsight", elements=[el])
        await msg.send()
        cl.user_session.set(_KEY_EL, el)
        cl.user_session.set(_KEY_MSG, msg)
    return el, msg  

async def update_progress(pct: int, label: str):
    el, msg = await _ensure_el()  
    el.props = {"pct": int(max(0, min(100, pct))), "label": str(label)}
    msg.elements = [el]
    await msg.update()  

async def progress_done():
    await update_progress(100, "Done")
