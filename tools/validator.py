"""Code validation via E2B sandbox with static-analysis fallback.

Supports Swift, Kotlin, Dart, JavaScript/TypeScript, and Python.
Falls back to static analysis (bracket matching + RevenueCat method
registry) when E2B is unavailable, so the agent is never blocked.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from anthropic import Anthropic

from config.settings import settings

logger = logging.getLogger("validator")

# ---------------------------------------------------------------------------
# RevenueCat SDK method registry
# ---------------------------------------------------------------------------

VALID_RC_METHODS: dict[str, list[str]] = {
    "ios": [
        "configure",
        "getOfferings",
        "purchase",
        "restorePurchases",
        "getCustomerInfo",
        "invalidateCustomerInfoCache",
        "logIn",
        "logOut",
        "setAttributes",
        "setEmail",
        "setPhoneNumber",
        "collectDeviceIdentifiers",
        "enableAdServicesAttributionTokenCollection",
    ],
    "android": [
        "configure",
        "getOfferings",
        "purchasePackage",
        "purchaseProduct",
        "restorePurchases",
        "getCustomerInfo",
        "invalidateCustomerInfoCache",
        "logIn",
        "logOut",
        "setAttributes",
        "setEmail",
        "setPhoneNumber",
        "collectDeviceIdentifiers",
    ],
    "flutter": [
        "configure",
        "getOfferings",
        "purchasePackage",
        "purchaseProduct",
        "restorePurchases",
        "getCustomerInfo",
        "invalidateCustomerInfoCache",
        "logIn",
        "logOut",
        "setAttributes",
        "setEmail",
        "setPhoneNumber",
        "collectDeviceIdentifiers",
    ],
}

# Map language aliases to platform keys
_LANG_TO_PLATFORM: dict[str, str] = {
    "swift": "ios",
    "objc": "ios",
    "objective-c": "ios",
    "kotlin": "android",
    "java": "android",
    "dart": "flutter",
    "flutter": "flutter",
}

# Bracket pairs for matching
_BRACKETS = {"(": ")", "[": "]", "{": "}"}


# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------


def _check_brackets(code: str) -> str | None:
    """Return an error message if brackets are unbalanced, else None."""
    stack: list[str] = []
    in_string = False
    string_char = ""
    prev = ""

    for ch in code:
        if in_string:
            if ch == string_char and prev != "\\":
                in_string = False
        else:
            if ch in ('"', "'", "`"):
                in_string = True
                string_char = ch
            elif ch in _BRACKETS:
                stack.append(_BRACKETS[ch])
            elif ch in _BRACKETS.values():
                if not stack or stack[-1] != ch:
                    return f"Unmatched closing bracket: '{ch}'"
                stack.pop()
        prev = ch

    if stack:
        return f"Unclosed bracket(s): expected {', '.join(repr(b) for b in reversed(stack))}"
    return None


def _detect_platform(code: str, language: str) -> str:
    """Guess the RevenueCat platform from code + language hint."""
    lang = language.lower().strip()
    if lang in _LANG_TO_PLATFORM:
        return _LANG_TO_PLATFORM[lang]
    # Heuristic detection from code patterns
    if "Purchases.shared" in code or "import StoreKit" in code:
        return "ios"
    if "Purchases.sharedInstance" in code or "BillingClient" in code:
        return "android"
    if "Purchases.configure" in code and ("await " in code or "async " in code):
        return "flutter"
    return ""


def _check_rc_methods(code: str, platform: str) -> list[str]:
    """Check that referenced RevenueCat methods exist in the SDK registry.

    Returns a list of error strings for invalid methods.
    """
    if not platform or platform not in VALID_RC_METHODS:
        return []

    valid = set(VALID_RC_METHODS[platform])
    errors: list[str] = []

    # Match patterns like: Purchases.shared.methodName  or  Purchases.methodName
    patterns = [
        r"Purchases\.shared\.(\w+)",
        r"Purchases\.sharedInstance\.(\w+)",
        r"Purchases\.(\w+)\s*\(",
        r"await\s+Purchases\.(\w+)",
    ]
    found_methods: set[str] = set()
    for pat in patterns:
        found_methods.update(re.findall(pat, code))

    # Filter out non-method matches (e.g., "shared", "sharedInstance")
    skip = {"shared", "sharedInstance"}
    found_methods -= skip

    for method in sorted(found_methods):
        if method not in valid:
            errors.append(
                f"Unknown RevenueCat {platform} method: '{method}'. "
                f"Valid methods: {', '.join(sorted(valid))}"
            )

    return errors


def _check_api_key_placeholder(code: str) -> str | None:
    """Warn if the code contains a placeholder API key."""
    placeholders = [
        "your_api_key",
        "YOUR_API_KEY",
        "<api_key>",
        "<your_api_key>",
        "api_key_here",
        "REPLACE_ME",
        "sk_",
    ]
    for p in placeholders:
        if p in code:
            return (
                f"Code contains placeholder API key '{p}'. "
                "Remind the developer to replace it with their actual key."
            )
    return None


def _check_swift_syntax(code: str) -> list[str]:
    """Swift-specific checks."""
    errors: list[str] = []
    if "Purchases.sharedInstance" in code:
        errors.append(
            "Swift uses 'Purchases.shared', not 'Purchases.sharedInstance'. "
            "Use: Purchases.shared.getOfferings { ... }"
        )
    return errors


def _check_kotlin_syntax(code: str) -> list[str]:
    """Kotlin-specific checks."""
    errors: list[str] = []
    if "Purchases.shared." in code and "Purchases.sharedInstance" not in code:
        # Could be wrong — Kotlin uses sharedInstance
        if "import com.revenuecat" in code or "Purchases.shared." in code:
            errors.append(
                "Kotlin uses 'Purchases.sharedInstance', not 'Purchases.shared'. "
                "Use: Purchases.sharedInstance.getOfferings(...)"
            )
    return errors


def _check_dart_syntax(code: str) -> list[str]:
    """Dart/Flutter-specific checks."""
    errors: list[str] = []
    if "Purchases.shared." in code:
        errors.append(
            "Flutter SDK uses 'Purchases.configure' and 'await Purchases.getOfferings()', "
            "not 'Purchases.shared'. Check the Flutter SDK docs."
        )
    return errors


def _static_analysis(code: str, language: str) -> dict[str, Any]:
    """Run static analysis without E2B. Returns a validation result dict."""
    t0 = time.monotonic()
    errors: list[str] = []
    warnings: list[str] = []

    # Bracket matching
    bracket_err = _check_brackets(code)
    if bracket_err:
        errors.append(bracket_err)

    # Platform detection and RC method validation
    platform = _detect_platform(code, language)
    method_errors = _check_rc_methods(code, platform)
    errors.extend(method_errors)

    # Language-specific checks
    lang = language.lower().strip()
    if lang in ("swift", "objc", "objective-c"):
        errors.extend(_check_swift_syntax(code))
    elif lang in ("kotlin", "java"):
        errors.extend(_check_kotlin_syntax(code))
    elif lang in ("dart", "flutter"):
        errors.extend(_check_dart_syntax(code))

    # API key placeholder warning
    api_warn = _check_api_key_placeholder(code)
    if api_warn:
        warnings.append(api_warn)

    elapsed = int((time.monotonic() - t0) * 1000)

    if errors:
        return {
            "valid": False,
            "error": "; ".join(errors),
            "suggestion": None,
            "execution_time_ms": elapsed,
            # Legacy compat keys
            "success": False,
        }

    return {
        "valid": True,
        "error": "; ".join(warnings) if warnings else None,
        "suggestion": None,
        "execution_time_ms": elapsed,
        "success": True,
    }


# ---------------------------------------------------------------------------
# CodeValidator class
# ---------------------------------------------------------------------------


class CodeValidator:
    """Validates code snippets via E2B sandbox with static-analysis fallback."""

    def __init__(self, anthropic_client: Anthropic | None = None):
        self._client = anthropic_client or Anthropic(
            api_key=settings.anthropic_api_key,
        )

    async def validate_snippet(
        self, code: str, language: str = "python"
    ) -> dict[str, Any]:
        """Validate a code snippet.

        Tries E2B sandbox first; falls back to static analysis if E2B
        is unavailable or the API key is not set.

        Returns:
            {
                "valid": bool,
                "error": str | None,
                "suggestion": str | None,
                "execution_time_ms": int,
            }
        """
        # Always run static analysis first for RC-specific checks
        static_result = _static_analysis(code, language)
        if not static_result["valid"]:
            # If static analysis already found errors, return immediately
            # and optionally generate a fix suggestion
            suggestion = await self._suggest_fix(
                code, static_result["error"], language
            )
            static_result["suggestion"] = suggestion
            return static_result

        # Try E2B sandbox for runtime validation
        if settings.e2b_api_key:
            try:
                return await self._validate_e2b(code, language, static_result)
            except Exception as exc:
                logger.warning("E2B unavailable, using static analysis: %s", exc)

        # Fallback: static analysis passed
        return static_result

    async def _validate_e2b(
        self,
        code: str,
        language: str,
        static_result: dict,
    ) -> dict[str, Any]:
        """Run code in E2B sandbox."""
        from e2b_code_interpreter import Sandbox

        t0 = time.monotonic()
        sbx = Sandbox(api_key=settings.e2b_api_key)
        try:
            execution = sbx.run_code(code)
            elapsed = int((time.monotonic() - t0) * 1000)
            stdout = "\n".join(r.text for r in execution.results if r.text)

            if execution.error:
                error_msg = f"{execution.error.name}: {execution.error.value}"
                suggestion = await self._suggest_fix(code, error_msg, language)
                return {
                    "valid": False,
                    "error": error_msg,
                    "suggestion": suggestion,
                    "execution_time_ms": elapsed,
                    "success": False,
                }

            return {
                "valid": True,
                "error": static_result.get("error"),  # pass through warnings
                "suggestion": None,
                "execution_time_ms": elapsed,
                "success": True,
            }
        finally:
            sbx.kill()

    async def _suggest_fix(
        self, code: str, error: str, language: str
    ) -> str | None:
        """Use Claude Haiku to suggest a fix for the broken code."""
        try:
            response = self._client.messages.create(
                model=settings.claude_model_haiku,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": (
                            "You fix broken code snippets. Return ONLY the "
                            "corrected code, no explanation. If the code uses "
                            "RevenueCat SDK, ensure correct method names."
                        ),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Language: {language}\n"
                            f"Error: {error}\n\n"
                            f"Code:\n```\n{code}\n```\n\n"
                            "Return the fixed code only."
                        ),
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Failed to generate fix suggestion: %s", exc)
            return None

    async def fix_code(self, code: str, error: str, language: str) -> str:
        """Fix broken code using Claude Haiku. Returns corrected code.

        Public API used by architect_node for auto-correction.
        """
        suggestion = await self._suggest_fix(code, error, language)
        return suggestion or code


# ---------------------------------------------------------------------------
# Module-level convenience function (legacy compat)
# ---------------------------------------------------------------------------

_default_validator: CodeValidator | None = None


def _get_validator() -> CodeValidator:
    global _default_validator
    if _default_validator is None:
        _default_validator = CodeValidator()
    return _default_validator


async def validate_code(code: str, language: str = "python") -> dict:
    """Module-level convenience wrapper.

    Returns a dict compatible with the legacy API:
        success (bool), error (str), output (str)
    Also includes the new keys: valid, suggestion, execution_time_ms.
    """
    validator = _get_validator()
    result = await validator.validate_snippet(code, language)
    # Ensure legacy keys exist
    result.setdefault("success", result.get("valid", False))
    result.setdefault("output", "")
    if result.get("error") is None:
        result["error"] = ""
    return result
