"""Read-only local dashboard API for the paper-trading bot.

The dashboard intentionally exposes no order-placement or configuration-write endpoint.
Run with: python -m src.dashboard
"""
from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

from src.paths import PROJECT_ROOT, project_path

load_dotenv()

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


def _read_json(path: Path) -> dict[str, Any]:
    """Read an optional JSON artifact without letting corrupt data break the UI."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any, default: float | None = None) -> float | None:
    """Convert untrusted persisted values to finite numbers only."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _json_object(value: Any) -> dict[str, Any]:
    """Decode stored JSON objects defensively; old/corrupt rows stay visible."""
    try:
        decoded = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _query_memory(path: Path, query: str, params: tuple = ()) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(query, params).fetchall()]
    except sqlite3.Error:
        # A dashboard must remain usable while the bot initializes its database.
        return []


def _request_limit() -> int:
    try:
        value = int(request.args.get("limit", 25))
    except (TypeError, ValueError):
        value = 25
    return min(max(value, 1), 100)


def create_app(state_path: str | Path | None = None, memory_path: str | Path | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    state_file = Path(state_path or os.getenv("PAPER_STATE_PATH", "data/paper_state.json"))
    state_file = project_path(state_file)
    memory_file = Path(memory_path or os.getenv("MEMORY_DB_PATH", "data/memory.sqlite3"))
    memory_file = project_path(memory_file)

    @app.get("/")
    def dashboard():
        return send_from_directory(DASHBOARD_DIR, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "mode": "paper", "read_only": True})

    @app.get("/api/status")
    def status():
        state = _read_json(state_file)
        position = state.get("position")
        position = position if isinstance(position, dict) else None
        snapshot = state.get("last_market")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        mark_price = _finite_number(snapshot.get("price"))
        if mark_price is None and position:
            mark_price = _finite_number(position.get("entry"))
        balance = _finite_number(state.get("balance"), 0.0) or 0.0
        quantity = _finite_number(position.get("quantity"), 0.0) if position else 0.0
        equity = balance + (
            quantity * mark_price
            if position and mark_price is not None and quantity is not None else 0.0
        )
        summary = _query_memory(
            memory_file,
            "SELECT 'decisions' AS name, COUNT(*) AS count FROM decisions "
            "UNION ALL SELECT 'trades', COUNT(*) FROM trades "
            "UNION ALL SELECT 'patterns', COUNT(*) FROM patterns "
            "UNION ALL SELECT 'evaluations', COUNT(*) FROM evaluations",
        )
        return jsonify({
            "mode": "paper",
            "read_only": True,
            "state_available": bool(state),
            "balance": balance,
            "equity": equity,
            "start_balance": _finite_number(state.get("start_balance"), balance),
            "day": state.get("day"),
            "position": position,
            "market": snapshot,
            "memory": {item["name"]: item["count"] for item in summary},
        })

    @app.get("/api/integration")
    def integration():
        """Read-only diagnostic of the files and stores expected by the paper system."""
        checks: list[dict[str, str]] = []

        mode_ok = os.getenv("MODE", "paper").strip().lower() == "paper"
        checks.append({
            "id": "mode", "label": "حالت معامله",
            "status": "ok" if mode_ok else "error",
            "detail": "فقط شبیه‌سازی فعال است" if mode_ok else "MODE باید paper باشد",
        })

        state_exists = state_file.is_file()
        state_valid = False
        if state_exists:
            try:
                parsed_state = json.loads(state_file.read_text(encoding="utf-8"))
                state_valid = isinstance(parsed_state, dict) and bool(parsed_state)
            except (OSError, json.JSONDecodeError):
                state_valid = False
        checks.append({
            "id": "state", "label": "وضعیت حساب",
            "status": "ok" if state_valid else ("error" if state_exists else "missing"),
            "detail": "فایل وضعیت قابل خواندن است" if state_valid else
                "فایل وضعیت وجود دارد اما JSON معتبر نیست" if state_exists else
                "هنوز فایل وضعیت ساخته نشده است",
        })

        memory_ok = False
        missing_tables: list[str] = []
        if memory_file.is_file():
            try:
                uri = memory_file.resolve().as_uri() + "?mode=ro"
                with sqlite3.connect(uri, uri=True, timeout=2) as conn:
                    tables = {row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()}
                required = {"decisions", "trades", "patterns", "evaluations", "model_versions"}
                missing_tables = sorted(required - tables)
                memory_ok = not missing_tables
            except sqlite3.Error:
                memory_ok = False
        checks.append({
            "id": "memory", "label": "حافظه SQLite",
            "status": "ok" if memory_ok else ("error" if memory_file.exists() else "missing"),
            "detail": "پایگاه داده و جدول‌های اصلی آماده‌اند" if memory_ok else (
                "جدول‌های ناقص: " + ", ".join(missing_tables) if missing_tables else
                "پایگاه داده قابل خواندن نیست" if memory_file.exists() else "پایگاه داده هنوز ساخته نشده است"
            ),
        })

        trades_file = project_path(os.getenv("PAPER_TRADES_PATH", "data/trades.csv"))
        ledger_ok = False
        if trades_file.is_file():
            try:
                with trades_file.open("r", encoding="utf-8", newline="") as stream:
                    header = stream.readline().strip().split(",")
                ledger_ok = header == ["time_utc", "side", "price", "quantity", "fee", "balance", "reason"]
            except OSError:
                ledger_ok = False
        checks.append({
            "id": "ledger", "label": "دفتر معاملات CSV",
            "status": "ok" if ledger_ok else ("error" if trades_file.exists() else "missing"),
            "detail": "سرستون‌های دفتر معاملات معتبرند" if ledger_ok else
                "فایل موجود است اما سرستون‌ها معتبر نیستند" if trades_file.exists() else
                "دفتر معاملات هنوز ساخته نشده است",
        })

        ledger_sync_status = "missing"
        ledger_sync_detail = "برای تطبیق باید دفتر CSV و حافظه آماده باشند"
        if ledger_ok and memory_ok:
            try:
                with trades_file.open("r", encoding="utf-8", newline="") as stream:
                    csv_trade_count = sum(1 for _ in csv.DictReader(stream))
                trade_count_rows = _query_memory(
                    memory_file, "SELECT COUNT(*) AS count FROM trades"
                )
                if not trade_count_rows:
                    ledger_sync_status = "error"
                    ledger_sync_detail = "تعداد معاملات SQLite قابل خواندن نیست"
                else:
                    db_trade_count = int(trade_count_rows[0]["count"])
                    if csv_trade_count == db_trade_count:
                        ledger_sync_status = "ok"
                        ledger_sync_detail = f"تعداد رکوردها برابر است: {csv_trade_count}"
                    else:
                        ledger_sync_status = "warning"
                        ledger_sync_detail = (
                            f"تعداد CSV برابر {csv_trade_count} و تعداد SQLite برابر {db_trade_count} است؛ "
                            "احتمال ناهماهنگی پس از توقف ناگهانی وجود دارد"
                        )
            except (OSError, csv.Error, ValueError, TypeError):
                ledger_sync_status = "error"
                ledger_sync_detail = "تطبیق دفتر معاملات و حافظه با خطا مواجه شد"
        checks.append({
            "id": "ledger_sync", "label": "تطبیق معاملات",
            "status": ledger_sync_status,
            "detail": ledger_sync_detail,
        })

        model_file = project_path(os.getenv("MODEL_PATH", "data/model.joblib"))
        model_ok = model_file.is_file() and model_file.stat().st_size > 0
        checks.append({
            "id": "model", "label": "فایل مدل ML",
            "status": "ok" if model_ok else "missing",
            "detail": "فایل مدل موجود است؛ سازگاری آن هنگام بارگذاری بررسی می‌شود" if model_ok else
                "مدل موجود نیست؛ برای اجرای ربات باید آموزش داده شود",
        })

        errors = sum(item["status"] == "error" for item in checks)
        missing = sum(item["status"] == "missing" for item in checks)
        warnings = sum(item["status"] == "warning" for item in checks)
        overall = "error" if errors else "warning" if missing or warnings else "ok"
        return jsonify({"overall": overall, "checks": checks})

    @app.get("/api/trades")
    def trades():
        limit = _request_limit()
        rows = _query_memory(
            memory_file,
            "SELECT id, timestamp, symbol, side, price, quantity, fee, realized_pnl, reason "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return jsonify({"items": rows})

    @app.get("/api/backtest")
    def backtest():
        report_path = Path(os.getenv("BACKTEST_REPORT_PATH", "data/backtest_report.json"))
        report_path = project_path(report_path)
        report = _read_json(report_path)
        metrics = report.get("metrics", {})
        return jsonify({
            "available": isinstance(metrics, dict) and bool(metrics),
            "metrics": metrics if isinstance(metrics, dict) else {},
            "trades": report.get("trades", []) if isinstance(report.get("trades", []), list) else [],
            "equity_curve": report.get("equity_curve", []) if isinstance(report.get("equity_curve", []), list) else [],
        })

    @app.get("/api/decisions")
    def decisions():
        limit = _request_limit()
        rows = _query_memory(
            memory_file,
            "SELECT id, timestamp, symbol, timeframe, decision, price, confidence, context_json, outcome_json "
            "FROM decisions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        for row in rows:
            row["context"] = _json_object(row.pop("context_json", "{}"))
            row["outcome"] = _json_object(row.get("outcome_json")) if row.get("outcome_json") else None
            row.pop("outcome_json", None)
        return jsonify({"items": rows})

    return app


app = create_app()


if __name__ == "__main__":
    # Loopback only: do not expose this read-only dashboard to a public network by default.
    app.run(host="127.0.0.1", port=int(os.getenv("DASHBOARD_PORT", "8765")), debug=False)
