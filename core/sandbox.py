import os
import tempfile
import textwrap
import subprocess
from typing import Dict, Any

def run_user_code_inproc(code: str, local_env: Dict[str, Any]) -> Dict[str, Any]:
    env = {"__builtins__": __builtins__}
    if local_env:
        env.update(local_env)
    exec(code, env, env)
    return {
        "res_query": env.get("res_query"),
    }

#def run_user_code_subprocess(code: str, local_env: Dict[str, Any]) -> Dict[str, Any]:


def run_user_code(code: str, local_env: Dict[str, Any]):
    mode = os.getenv("EXECUTION_MODE", "inproc")
    #if mode == "subprocess":
        #return run_user_code_subprocess(code, local_env)
    return run_user_code_inproc(code, local_env)
