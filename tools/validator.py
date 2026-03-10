"""Code validation via the E2B cloud sandbox."""

from __future__ import annotations

from e2b_code_interpreter import Sandbox

from config.settings import settings


async def validate_code(code: str, language: str = "python") -> dict:
    """Execute *code* inside an E2B sandbox and return the result.

    Returns a dict with:
        success (bool) - whether execution completed without errors
        output  (str)  - stdout produced by the snippet
        error   (str)  - stderr / exception message, empty on success
    """
    sbx = Sandbox(api_key=settings.e2b_api_key)
    try:
        execution = sbx.run_code(code)
        stdout = "\n".join(r.text for r in execution.results if r.text)
        if execution.error:
            return {
                "success": False,
                "output": stdout,
                "error": f"{execution.error.name}: {execution.error.value}",
            }
        return {"success": True, "output": stdout, "error": ""}
    finally:
        sbx.kill()
