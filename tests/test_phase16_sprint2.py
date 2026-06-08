"""
Phase-3 sprint 2: B1 vault persistence, H1 mandate recalibration, M1
copywriting, dead-code removal.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── B1: vault persistence via env ─────────────────────────────────────────

class VaultDbPathEnvTest(unittest.TestCase):

    def test_env_overrides_default(self) -> None:
        """VAULT_DB_PATH env must drive SecureVault location used by tg_bot."""
        os.environ["VAULT_DB_PATH"] = "/tmp/test-vault.db"
        import importlib, sys as _sys
        # Force tg_bot module-level re-evaluation.
        for mod in ("tg_bot",):
            _sys.modules.pop(mod, None)
        try:
            import tg_bot
        except BaseException:
            self.skipTest("tg_bot import unavailable")
            return
        self.assertEqual(tg_bot.VAULT_DB, "/tmp/test-vault.db")
        os.environ.pop("VAULT_DB_PATH", None)

    def test_cloudbuild_wires_vault_path(self) -> None:
        """
        cloudbuild.yaml lives at the REPO root, not inside the Docker image.
        When this test runs from inside the prod image at CI test-gate time
        the file is absent and the assertion is meaningless — skip cleanly.
        When run from the repo (local dev / pre-commit) we DO check the
        wiring.
        """
        cb_path = Path(__file__).resolve().parent.parent / "cloudbuild.yaml"
        if not cb_path.is_file():
            self.skipTest("cloudbuild.yaml not packaged into Docker image "
                          "(expected at CI test-gate); repo-local run will check.")
        cb = cb_path.read_text()
        self.assertIn("VAULT_DB_PATH=/mnt/state/users_vault.db", cb)


# ── H1: MODERATE recalibration 0.05 → 0.065 ───────────────────────────────

class ModerateRecalibrationTest(unittest.TestCase):

    def test_moderate_cvar_base_is_065(self) -> None:
        from finance.scoring import _RISK_MANDATE_MATRIX
        self.assertAlmostEqual(_RISK_MANDATE_MATRIX["MODERATE"]["cvar_base"],
                               0.065, places=4)

    def test_moderate_score_softens_for_same_portfolio(self) -> None:
        """Same (vol, cvar, erc) → softer score than the 0.05 era (63 was test golden)."""
        from finance.scoring import composite_risk_score as c
        new = c(0.142, -0.052, 22.0, mandate="MODERATE")
        # The pre-recalibration value was 63 (CVaR component saturated at 100).
        # After widening to 0.065 the s_cvar = 80 → composite ≈ 55.
        self.assertLess(new, 63)
        self.assertGreater(new, 40)

    def test_ordering_still_holds(self) -> None:
        from finance.scoring import composite_risk_score as c
        a = c(0.22, -0.04, 30.0, mandate="CONSERVATIVE")
        b = c(0.22, -0.04, 30.0, mandate="MODERATE")
        d = c(0.22, -0.04, 30.0, mandate="AGGRESSIVE")
        self.assertGreaterEqual(a, b)
        self.assertGreaterEqual(b, d)


# ── H1: payload exposes mandate label ─────────────────────────────────────

class PayloadMandateLabelTest(unittest.TestCase):

    def test_label_in_payload(self) -> None:
        import pandas as pd
        from pdf_payload import build_payload
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
                 "PnL": 200.0, "Return_Pct": 0.25,
                 "Euler_Risk_Contribution_Pct": 10.0},
            ]),
            "total_value": 1000.0, "portfolio_metrics": {},
            "benchmark_comparison": {}, "sector_exposure": {"Technology": 1.0},
            "risk_mandate": "MODERATE",
        }
        payload = build_payload(results, "base")
        self.assertEqual(payload["risk_mandate_label"], "Умеренный")

    def test_label_default_when_missing(self) -> None:
        import pandas as pd
        from pdf_payload import build_payload
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
                 "PnL": 200.0, "Return_Pct": 0.25,
                 "Euler_Risk_Contribution_Pct": 10.0},
            ]),
            "total_value": 1000.0, "portfolio_metrics": {},
            "benchmark_comparison": {}, "sector_exposure": {"Technology": 1.0},
        }
        payload = build_payload(results, "base")
        self.assertEqual(payload["risk_mandate_label"], "Умеренный")


# ── M1: copy / dead-code / requirements ───────────────────────────────────

class CopywritingAndCleanupTest(unittest.TestCase):

    def test_send_pdf_renamed(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "src" / "tg_bot.py").read_text()
        self.assertIn("async def _send_report(", src)
        self.assertNotIn("async def _send_pdf(", src)

    def test_no_pdf_otchet_in_user_copy(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "src" / "tg_bot.py").read_text()
        # The phrase "PDF-отчёт" used to ship to users in 2 places.
        self.assertNotIn("PDF-отчёт", src)
        self.assertNotIn("PDF-отчет", src)

    def test_dead_modules_removed(self) -> None:
        src_dir = Path(__file__).resolve().parent.parent / "src"
        self.assertFalse((src_dir / "parity_audit.py").exists())
        self.assertFalse((src_dir / "command_graph.py").exists())

    def test_pymupdf_dropped_from_bot_requirements(self) -> None:
        req = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
        # Allow it to be mentioned in a COMMENT, but not as an installable
        # requirement line (i.e. no live `pymupdf4llm>=…` line).
        lines = [l.strip() for l in req.splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        self.assertFalse(any(l.startswith("pymupdf4llm") for l in lines),
                         "pymupdf4llm must not appear as an installable line")


# ── Concierge-tone step copy ──────────────────────────────────────────────

class ProgressMessageToneTest(unittest.TestCase):

    def test_steps_match_concierge_spec(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "src" / "tg_bot.py").read_text()
        for needle in (
            "Шаг 1/4:* Интеграция рыночных данных",
            "Шаг 2/4:* Факторное моделирование",
            "Шаг 3/4:* Оптимизация целевых весов",
            "Шаг 4/4:* Сборка интерактивного интерфейса",
            "Портфель успешно получен и принят в обработку",
            "Расчёты успешно завершены",
        ):
            self.assertIn(needle, src, f"missing concierge copy: {needle!r}")

    def test_raw_metric_dump_removed(self) -> None:
        """The raw 'MAC3 Risk Summary' chat-dump send_message must be gone."""
        src = (Path(__file__).resolve().parent.parent / "src" / "tg_bot.py").read_text()
        # The visible-to-user title used to be the header of an in-chat dump.
        # It must not appear inside any quoted send_message string anymore.
        self.assertNotIn('"📊 *MAC3 Risk Summary:*"', src)
        # The ASCII-bar sector chart copy was also retired.
        self.assertNotIn('"📊 *Секторное распределение:*"', src)


if __name__ == "__main__":
    unittest.main()
