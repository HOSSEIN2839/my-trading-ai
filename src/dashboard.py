"""Read-only local dashboard API for the paper-trading bot.

The dashboard intentionally exposes no order-placement or configuration-write endpoint.
Run with: python -m src.dashboard
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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
    if not state_file.is_absolute():
        state_file = PROJECT_ROOT / state_file
    memory_file = Path(memory_path or os.getenv("MEMORY_DB_PATH", "data/memory.sqlite3"))
    if not memory_file.is_absolute():
        memory_file = PROJECT_ROOT / memory_file

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
        snapshot = state.get("last_market") or {}
        mark_price = snapshot.get("price")
        if mark_price is None and position:
            mark_price = position.get("entry")
        balance = float(state.get("balance", 0.0))
        equity = balance + (
            float(position.get("quantity", 0.0)) * float(mark_price)
            if position and mark_price is not None else 0.0
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
            "start_balance": float(state.get("start_balance", balance)),
            "day": state.get("day"),
            "position": position,
            "market": snapshot,
            "memory": {item["name"]: item["count"] for item in summary},
        })

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
            row["context"] = json.loads(row.pop("context_json") or "{}")
            row["outcome"] = json.loads(row.pop("outcome_json")) if row.get("outcome_json") else None
            row.pop("outcome_json", None)
        return jsonify({"items": rows})

    return app


app = create_app()


if __name__ == "__main__":
    # Loopback only: do not expose this read-only dashboard to a public network by default.
    app.run(host="127.0.0.1", port=int(os.getenv("DASHBOARD_PORT", "8765")), debug=False)
