import re

CODE_BLOCK_RE = re.compile(r"```(?:[ \t]*python[ \t]*)?\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code_block(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if not m:
        return ""
    code = m.group(1)
    # A model that puts the language tag on its own line (```\npython\n...)
    # rather than right after the fence leaves it as the first captured
    # line; drop it by exact match only, so real code that merely starts
    # with a "python"-prefixed identifier (e.g. python_version = ...) isn't
    # corrupted by a substring match.
    first_line, sep, rest = code.partition("\n")
    if first_line.strip().lower() == "python":
        code = rest
    return code.strip()
