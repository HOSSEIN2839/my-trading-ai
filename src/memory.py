"""Persistent SQLite memory for the independent paper-trading agent.

This module stores observations and learning artifacts; it does not place orders,
call external AI services, or claim that stored patterns are profitable.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.paths import project_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class TradingMemory:
    """Durable, local memory for decisions, trades, patterns, evaluations and models."""

    def __init__(self, db_path: str | Path = "data/memory.sqlite3") -> None:
        self.db_path = Path(":memory:") if str(db_path) == ":memory:" else project_path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=15)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('BUY','SELL','HOLD')),
            price REAL,
            confidence REAL,
            context_json TEXT NOT NULL,
            outcome_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol, timeframe, timestamp);

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            realized_pnl REAL,
            reason TEXT NOT NULL DEFAULT '',
            details_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(timestamp);

        CREATE TABLE IF NOT EXISTS patterns (
            pattern_key TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            average_return REAL,
            updated_at TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            dataset_start TEXT,
            dataset_end TEXT,
            accepted INTEGER NOT NULL DEFAULT 0 CHECK(accepted IN (0,1)),
            metrics_json TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_evaluations_model ON evaluations(model_version, timestamp);

        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            parent_version TEXT,
            accepted INTEGER NOT NULL DEFAULT 0 CHECK(accepted IN (0,1)),
            metrics_json TEXT NOT NULL,
            FOREIGN KEY(parent_version) REFERENCES model_versions(version)
        );
        """)
        self._conn.commit()

    def record_decision(
        self, *, symbol: str, timeframe: str, decision: str,
        price: float | None = None, confidence: float | None = None,
        context: Mapping[str, Any] | None = None, timestamp: str | None = None,
    ) -> int:
        decision = decision.upper()
        if decision not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("decision must be BUY, SELL or HOLD")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        cur = self._conn.execute(
            """INSERT INTO decisions(timestamp,symbol,timeframe,decision,price,confidence,context_json)
               VALUES(?,?,?,?,?,?,?)""",
            (timestamp or _utc_now(), symbol, timeframe, decision, price, confidence, _json(context or {})),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def annotate_decision(self, decision_id: int, outcome: Mapping[str, Any]) -> None:
        cur = self._conn.execute(
            "UPDATE decisions SET outcome_json=? WHERE id=?", (_json(outcome), decision_id)
        )
        if cur.rowcount != 1:
            raise KeyError(f"decision id not found: {decision_id}")
        self._conn.commit()

    def record_trade(
        self, *, symbol: str, side: str, price: float, quantity: float,
        fee: float = 0.0, realized_pnl: float | None = None, reason: str = "",
        details: Mapping[str, Any] | None = None, timestamp: str | None = None,
    ) -> int:
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if price <= 0 or quantity <= 0 or fee < 0:
            raise ValueError("price and quantity must be positive; fee cannot be negative")
        cur = self._conn.execute(
            """INSERT INTO trades(timestamp,symbol,side,price,quantity,fee,realized_pnl,reason,details_json)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (timestamp or _utc_now(), symbol, side, price, quantity, fee, realized_pnl, reason, _json(details or {})),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def remember_pattern(
        self, *, pattern_key: str, description: str, evidence_count: int,
        success_count: int, average_return: float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not pattern_key.strip() or evidence_count < 0 or success_count < 0 or success_count > evidence_count:
            raise ValueError("invalid pattern key or evidence counts")
        self._conn.execute(
            """INSERT INTO patterns(pattern_key,description,evidence_count,success_count,average_return,updated_at,details_json)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(pattern_key) DO UPDATE SET
                 description=excluded.description, evidence_count=excluded.evidence_count,
                 success_count=excluded.success_count, average_return=excluded.average_return,
                 updated_at=excluded.updated_at, details_json=excluded.details_json""",
            (pattern_key, description, evidence_count, success_count, average_return, _utc_now(), _json(details or {})),
        )
        self._conn.commit()

    def record_evaluation(
        self, *, model_version: str, metrics: Mapping[str, Any],
        dataset_start: str | None = None, dataset_end: str | None = None,
        accepted: bool = False, notes: str = "",
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO evaluations(timestamp,model_version,dataset_start,dataset_end,accepted,metrics_json,notes)
               VALUES(?,?,?,?,?,?,?)""",
            (_utc_now(), model_version, dataset_start, dataset_end, int(accepted), _json(metrics), notes),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def save_model_version(
        self, *, version: str, artifact_path: str | Path,
        metrics: Mapping[str, Any], accepted: bool = False,
        parent_version: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO model_versions(version,created_at,artifact_path,parent_version,accepted,metrics_json)
               VALUES(?,?,?,?,?,?)""",
            (version, _utc_now(), str(artifact_path), parent_version, int(accepted), _json(metrics)),
        )
        self._conn.commit()

    def recent_decisions(self, limit: int = 20, symbol: str | None = None) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["context"] = json.loads(item.pop("context_json"))
            item["outcome"] = json.loads(item.pop("outcome_json")) if item["outcome_json"] else None
            result.append(item)
        return result

    def summary(self) -> dict[str, int]:
        tables = ("decisions", "trades", "patterns", "evaluations", "model_versions")
        return {
            table: int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TradingMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
