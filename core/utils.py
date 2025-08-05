import re

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code_block(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if not m:
        return ""
    code = m.group(1)
    if code.strip().startswith("python"):
        code = code.split("python", 1)[1]
    return code.strip()
