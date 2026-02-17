"""Scoped chat agent for dashboard actions."""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(slots=True)
class ChatAction:
    """Parsed chat action."""

    action: str
    args: dict[str, Any]


class ChatAgent:
    """Command + optional LLM action parser."""

    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env

    def parse(self, message: str) -> ChatAction:
        """Parse user message to structured action."""
        parsed = self._parse_local(message)
        if parsed.action != "unknown":
            return parsed
        llm_action = self._parse_with_llm(message)
        return llm_action or parsed

    def _parse_local(self, message: str) -> ChatAction:
        text = message.strip()
        low = text.lower()
        if low.startswith("run ticker "):
            ticker = text.split(" ", 2)[-1].strip().upper()
            return ChatAction("run_scan", {"ticker": ticker})
        if low.startswith("set "):
            body = text[4:].strip()
            if "=" in body:
                key, value = body.split("=", 1)
                return ChatAction("set_param", {"key": key.strip(), "value": value.strip()})
        if low in {"show top trade", "top trade", "show best trade"}:
            return ChatAction("show_top_trade", {})
        if low in {"list outputs", "show outputs", "list runs"}:
            return ChatAction("list_outputs", {})
        if low.startswith("delete output "):
            run_id = text.split(" ", 2)[-1].strip()
            return ChatAction("delete_output", {"run_id": run_id})
        if low.startswith("explain trade "):
            trade_id = text.split(" ", 2)[-1].strip()
            return ChatAction("explain_trade", {"trade_id": trade_id})
        if low.startswith("confirm "):
            token = text.split(" ", 1)[-1].strip()
            return ChatAction("confirm", {"token": token})
        ticker_match = re.fullmatch(r"[A-Za-z]{1,10}", text)
        if ticker_match:
            return ChatAction("run_scan", {"ticker": text.upper()})
        return ChatAction("unknown", {"message": text})

    def _parse_with_llm(self, message: str) -> ChatAction | None:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None
        system = (
            "You convert user input into one JSON action object.\n"
            "Allowed actions: run_scan, set_param, show_top_trade, list_outputs, delete_output, explain_trade, confirm, unknown.\n"
            "Return JSON with keys: action (string), args (object).\n"
            "For run_scan args require ticker uppercase if present."
        )
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": message},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            action = str(payload.get("action", "unknown"))
            args = payload.get("args", {})
            if not isinstance(args, dict):
                args = {}
            return ChatAction(action=action, args=args)
        except Exception:
            return None

    def handle(
        self,
        message: str,
        tools: dict[str, Callable[..., Any]],
        pending: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Handle one chat message with scoped tools."""
        action = self.parse(message)
        if action.action == "confirm":
            token = str(action.args.get("token", ""))
            if token not in pending:
                return {"ok": False, "message": "Invalid or expired confirmation token."}
            payload = pending.pop(token)
            if payload["action"] == "delete_output":
                run_id = payload["run_id"]
                deleted = bool(tools["delete_output"](run_id))
                return {"ok": deleted, "message": f"Delete output {run_id}: {'done' if deleted else 'not found'}."}
            return {"ok": False, "message": "Unsupported confirmation action."}

        if action.action == "run_scan":
            ticker = str(action.args.get("ticker", "")).upper().strip()
            if not ticker:
                return {"ok": False, "message": "Missing ticker symbol."}
            result = tools["run_scan"](ticker)
            return {"ok": True, "message": f"Run complete for {ticker}.", "result": result}

        if action.action == "set_param":
            key = str(action.args.get("key", "")).strip()
            value = action.args.get("value")
            if not key:
                return {"ok": False, "message": "Missing parameter key."}
            out = tools["set_param"](key, value)
            return {"ok": True, "message": str(out)}

        if action.action == "show_top_trade":
            result = tools["show_top_trade"]()
            return {"ok": True, "message": "Top trade returned.", "result": result}

        if action.action == "list_outputs":
            result = tools["list_outputs"]()
            if isinstance(result, pd.DataFrame):
                text = f"{len(result)} outputs found."
            else:
                text = "Output list returned."
            return {"ok": True, "message": text, "result": result}

        if action.action == "delete_output":
            run_id = str(action.args.get("run_id", "")).strip()
            if not run_id:
                return {"ok": False, "message": "Missing run_id for delete."}
            token = secrets.token_hex(4).upper()
            pending[token] = {"action": "delete_output", "run_id": run_id}
            return {
                "ok": True,
                "message": f"Confirm deletion with: confirm {token}",
                "requires_confirmation": True,
                "token": token,
            }

        if action.action == "explain_trade":
            trade_id = str(action.args.get("trade_id", "")).strip()
            if not trade_id:
                return {"ok": False, "message": "Missing trade_id."}
            result = tools["explain_trade"](trade_id)
            return {"ok": True, "message": "Trade explanation ready.", "result": result}

        return {"ok": False, "message": "I can run ticker, set param, list outputs, show top trade, explain trade, or delete output."}
