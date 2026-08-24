import importlib
import multiprocessing
import os
import re
import traceback
import types
from typing import Any, Dict, Tuple

# Applies to every env var name; anything matching is dropped from the child's
# environment before user code runs, so a generated snippet reading os.environ
# (which several tools intentionally expose) can't exfiltrate provider/DB
# credentials. Broad on purpose: better to over-redact than miss a variable.
_SECRET_ENV_PATTERN = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|_AUTH)", re.IGNORECASE)

# Wall-clock budget for a single run_user_code call. This is the only timeout
# that can actually stop runaway/malicious generated code: callers invoke
# run_user_code synchronously from inside an `async def`, so the asyncio-level
# per-tool timeout can't preempt it (asyncio can only cancel at await points).
DEFAULT_TIMEOUT_S = float(os.getenv("SANDBOX_TIMEOUT_S", "600"))

# Resource limits are opt-in (unset by default). The tools behind run_user_code
# span lightweight pandas queries and GPU torch/MONAI inference, so a single
# default RLIMIT_AS/RLIMIT_CPU would either be too loose to matter or tight
# enough to break legitimate multi-threaded/CUDA workloads (CUDA context
# creation in particular tends to fail under a strict RLIMIT_AS regardless of
# actual memory used). Set these only if your deployment's workload profile
# is known and homogeneous. `tools/idc_python_worker.py` shows a working
# example (RLIMIT_CPU + a 4GB RLIMIT_AS) for a narrower, CPU-only worker.
_MEMORY_LIMIT_MB = os.getenv("SANDBOX_MEMORY_LIMIT_MB")
_CPU_LIMIT_S = os.getenv("SANDBOX_CPU_LIMIT_S")
_NPROC_LIMIT = os.getenv("SANDBOX_NPROC_LIMIT")


def _redact_secret_env() -> None:
    for name in list(os.environ):
        if _SECRET_ENV_PATTERN.search(name):
            os.environ.pop(name, None)


def _apply_resource_limits() -> None:
    try:
        import resource
    except ImportError:
        return  # not available on Windows

    if _CPU_LIMIT_S:
        try:
            cpu_s = int(_CPU_LIMIT_S)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass
    if _MEMORY_LIMIT_MB:
        try:
            mem_bytes = int(_MEMORY_LIMIT_MB) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
    if _NPROC_LIMIT:
        try:
            nproc = int(_NPROC_LIMIT)
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError):
            pass


def _split_modules(local_env: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Plain `pickle` can't serialize module objects, and every tool that
    calls run_user_code stuffs modules (pd, os, plt, nib, ...) into
    local_env. Send module names across the process boundary instead and
    re-import them in the child, where imports are cheap (already loaded)."""
    module_names: Dict[str, str] = {}
    data: Dict[str, Any] = {}
    for key, value in (local_env or {}).items():
        if isinstance(value, types.ModuleType):
            module_names[key] = value.__name__
        else:
            data[key] = value
    return module_names, data


def _sandboxed_worker(code: str, module_names: Dict[str, str], data_env: Dict[str, Any], conn) -> None:
    _redact_secret_env()
    _apply_resource_limits()
    try:
        env: Dict[str, Any] = {"__builtins__": __builtins__}
        for key, name in module_names.items():
            env[key] = importlib.import_module(name)
        env.update(data_env)
        exec(code, env, env)
        result: Dict[str, Any] = {"ok": True, "res_query": env.get("res_query")}
    except BaseException as exc:
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    try:
        conn.send(result)
    except Exception as exc:
        # res_query wasn't picklable (e.g. an object holding an open handle/lock) -
        # report that clearly instead of hanging the parent's poll() forever.
        conn.send({"ok": False, "error": f"Result could not be returned from the sandboxed process: {exc}"})
    finally:
        conn.close()


def run_user_code_subprocess(code: str, local_env: Dict[str, Any]) -> Dict[str, Any]:
    """Run generated code in an isolated child process.

    Unlike the in-process mode, this gives the code its own address space (no
    inherited DB/API client objects to reach via sys.modules), a redacted
    environment, and a wall-clock timeout that's actually enforceable via
    process termination.
    """
    module_names, data_env = _split_modules(local_env)
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_sandboxed_worker,
        args=(code, module_names, data_env, child_conn),
        daemon=True,
    )
    process.start()
    child_conn.close()

    if parent_conn.poll(DEFAULT_TIMEOUT_S):
        result = parent_conn.recv()
    else:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        result = {"ok": False, "error": f"Execution timed out after {DEFAULT_TIMEOUT_S:.0f}s and was terminated."}

    process.join(5)
    parent_conn.close()

    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Sandboxed code execution failed."))
    return {"res_query": result.get("res_query")}


def run_user_code_inproc(code: str, local_env: Dict[str, Any]) -> Dict[str, Any]:
    """Execute generated code directly in this process. No isolation, no
    timeout, full access to the server's own environment and memory - only
    use this in a trusted, single-tenant, already-sandboxed deployment
    (e.g. the whole app itself running inside a locked-down container)."""
    env = {"__builtins__": __builtins__}
    if local_env:
        env.update(local_env)
    exec(code, env, env)
    return {"res_query": env.get("res_query")}


def run_user_code(code: str, local_env: Dict[str, Any]) -> Dict[str, Any]:
    mode = os.getenv("EXECUTION_MODE", "subprocess")
    if mode == "inproc":
        return run_user_code_inproc(code, local_env)
    return run_user_code_subprocess(code, local_env)
