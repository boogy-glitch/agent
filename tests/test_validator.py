"""Tests for the CodeValidator and static analysis.

All external dependencies (E2B, Anthropic) are mocked so tests run
without API keys or network access.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Patch settings before importing validator
_mock_settings = SimpleNamespace(
    anthropic_api_key="test-key",
    claude_model_haiku="claude-haiku-4-5-20251001",
    e2b_api_key="",  # disable E2B by default -> static analysis
    embedding_model="voyage-3",
)
_settings_mod = MagicMock()
_settings_mod.settings = _mock_settings
sys.modules.setdefault("config", MagicMock())
sys.modules["config.settings"] = _settings_mod

from tools.validator import (
    CodeValidator,
    _check_brackets,
    _check_rc_methods,
    _static_analysis,
)


# ---------------------------------------------------------------------------
# 1. Valid Swift snippet — should pass
# ---------------------------------------------------------------------------

VALID_SWIFT = """\
import StoreKit

Purchases.shared.getOfferings { offerings, error in
    if let offerings = offerings {
        let package = offerings.current?.availablePackages.first
        Purchases.shared.purchase(package: package!) { transaction, customerInfo, error, userCancelled in
            print("Purchase complete")
        }
    }
}
"""


class TestValidSwift:
    @pytest.mark.asyncio
    async def test_valid_swift_passes(self):
        validator = CodeValidator(anthropic_client=MagicMock())
        result = await validator.validate_snippet(VALID_SWIFT, "swift")
        assert result["valid"] is True
        assert result["error"] is None or "placeholder" not in (result["error"] or "")


# ---------------------------------------------------------------------------
# 2. Invalid method name — should fail
# ---------------------------------------------------------------------------

INVALID_METHOD_SWIFT = """\
Purchases.shared.fetchAllProducts { products, error in
    print(products)
}
"""


class TestInvalidMethod:
    @pytest.mark.asyncio
    async def test_invalid_method_fails(self):
        validator = CodeValidator(anthropic_client=MagicMock())
        result = await validator.validate_snippet(INVALID_METHOD_SWIFT, "swift")
        assert result["valid"] is False
        assert "fetchAllProducts" in (result["error"] or "")


# ---------------------------------------------------------------------------
# 3. Missing API key — should warn
# ---------------------------------------------------------------------------

CODE_WITH_PLACEHOLDER_KEY = """\
Purchases.configure(withAPIKey: "your_api_key")
Purchases.shared.getOfferings { offerings, error in
    print(offerings)
}
"""


class TestMissingApiKey:
    @pytest.mark.asyncio
    async def test_placeholder_api_key_warns(self):
        validator = CodeValidator(anthropic_client=MagicMock())
        result = await validator.validate_snippet(
            CODE_WITH_PLACEHOLDER_KEY, "swift"
        )
        # Should still pass (it's a warning, not an error)
        assert result["valid"] is True
        # But should have a warning about the placeholder
        assert result["error"] is not None
        assert "placeholder" in result["error"].lower() or "your_api_key" in result["error"]


# ---------------------------------------------------------------------------
# 4. Valid Flutter snippet — should pass
# ---------------------------------------------------------------------------

VALID_FLUTTER = """\
final offerings = await Purchases.getOfferings();
if (offerings.current != null) {
    final package = offerings.current!.availablePackages.first;
    final customerInfo = await Purchases.purchasePackage(package);
    print(customerInfo.entitlements.all);
}
"""


class TestValidFlutter:
    @pytest.mark.asyncio
    async def test_valid_flutter_passes(self):
        validator = CodeValidator(anthropic_client=MagicMock())
        result = await validator.validate_snippet(VALID_FLUTTER, "dart")
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# 5. Malformed code — should fail with fix suggestion
# ---------------------------------------------------------------------------

MALFORMED_CODE = """\
Purchases.shared.getOfferings { offerings, error in
    if let offerings = offerings {
        print(offerings)
    // missing closing brackets
"""


class TestMalformedCode:
    @pytest.mark.asyncio
    async def test_malformed_code_fails_with_suggestion(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=(
                    'Purchases.shared.getOfferings { offerings, error in\n'
                    '    if let offerings = offerings {\n'
                    '        print(offerings)\n'
                    '    }\n'
                    '}'
                )
            )
        ]
        mock_client.messages.create.return_value = mock_response

        validator = CodeValidator(anthropic_client=mock_client)
        result = await validator.validate_snippet(MALFORMED_CODE, "swift")

        assert result["valid"] is False
        assert "bracket" in (result["error"] or "").lower()
        # Should have a fix suggestion from Haiku
        assert result["suggestion"] is not None
        assert "}" in result["suggestion"]


# ---------------------------------------------------------------------------
# Static analysis unit tests
# ---------------------------------------------------------------------------


class TestBracketMatching:
    def test_balanced(self):
        assert _check_brackets("func() { if (x) { [1, 2] } }") is None

    def test_unclosed_brace(self):
        err = _check_brackets("func() {")
        assert err is not None
        assert "}" in err

    def test_mismatched(self):
        err = _check_brackets("func() { (x] }")
        assert err is not None

    def test_ignores_strings(self):
        assert _check_brackets('let s = "hello { world"') is None


class TestRCMethodCheck:
    def test_valid_ios_methods(self):
        code = "Purchases.shared.getOfferings { }\nPurchases.shared.purchase(package:)"
        errors = _check_rc_methods(code, "ios")
        assert errors == []

    def test_invalid_ios_method(self):
        code = "Purchases.shared.doSomethingWrong()"
        errors = _check_rc_methods(code, "ios")
        assert len(errors) == 1
        assert "doSomethingWrong" in errors[0]

    def test_valid_flutter_methods(self):
        code = "await Purchases.getOfferings()\nawait Purchases.purchasePackage(pkg)"
        errors = _check_rc_methods(code, "flutter")
        assert errors == []

    def test_unknown_platform_skips(self):
        code = "Purchases.shared.anything()"
        errors = _check_rc_methods(code, "unknown")
        assert errors == []


class TestFixCode:
    @pytest.mark.asyncio
    async def test_fix_code_returns_corrected(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="fixed code here")]
        mock_client.messages.create.return_value = mock_response

        validator = CodeValidator(anthropic_client=mock_client)
        result = await validator.fix_code("broken", "some error", "swift")
        assert result == "fixed code here"

    @pytest.mark.asyncio
    async def test_fix_code_returns_original_on_failure(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API down")

        validator = CodeValidator(anthropic_client=mock_client)
        result = await validator.fix_code("broken code", "error", "swift")
        assert result == "broken code"
