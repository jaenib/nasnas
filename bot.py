"""
Simple Telegram bot to track shared travel expenses between two people.

Usage (DM the bot):
- `/add 23.50 lunch` to record an expense paid by you, split 50/50.
- `/balance` to see who owes whom.
- `/history` to see the last few entries.

All expenses are split evenly between the two configured users.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ----------------------------- configuration ----------------------------- #


@dataclass(frozen=True)
class UserConfig:
    id: int
    name: str


@dataclass(frozen=True)
class BotConfig:
    token: str
    users: List[UserConfig]
    data_path: Path
    base_currency: str
    rates_to_base: Dict[str, float]
    healthcheck_chat_id: Optional[int] = None
    instance_id: str = "default"
    instance_name: str = "Default"


def _env_int(key: str) -> int:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Environment variable {key} is required")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {key} must be an integer") from exc


def load_config() -> BotConfig:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is required. Set it in .env or the environment.")

    user_one = UserConfig(
        id=_env_int("USER_ONE_ID"), name=os.getenv("USER_ONE_NAME", "User 1")
    )
    user_two = UserConfig(
        id=_env_int("USER_TWO_ID"), name=os.getenv("USER_TWO_NAME", "User 2")
    )
    data_path = Path(os.getenv("DATA_PATH", "data/expenses.json"))
    health_raw = os.getenv("HEALTHCHECK_CHAT_ID")
    health_id = int(health_raw) if health_raw else None
    base_currency = os.getenv("BASE_CURRENCY", "MAD").strip().upper()

    rates: Dict[str, float] = {}
    for key, value in os.environ.items():
        if not key.startswith("RATE_") or not key.endswith("_TO_BASE"):
            continue
        code = key[len("RATE_") : -len("_TO_BASE")]
        code = code.strip().upper()
        if not code:
            continue
        try:
            rates[code] = float(value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid rate for {key}. Must be a number.") from exc

    return BotConfig(
        token=token,
        users=[user_one, user_two],
        data_path=data_path,
        base_currency=base_currency,
        rates_to_base=rates,
        healthcheck_chat_id=health_id,
    )


# ----------------------------- persistence ------------------------------ #


def empty_ledger_state() -> Dict[str, Any]:
    return {
        "expenses": [],
        "settlements": [],
        "pushups": {},
        "agreements": [],
        "challenges": [],
    }


class Ledger:
    def __init__(
        self,
        path: Path,
        users: List[UserConfig],
        state: Optional[Dict[str, Any]] = None,
        save_callback: Optional[Callable[[], None]] = None,
    ):
        self.path = path
        self.users = {user.id: user for user in users}
        self._save_callback = save_callback
        if state is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.state = empty_ledger_state()
            self._load()
        else:
            self.state = state
            self._ensure_shape()

    def _ensure_shape(self) -> None:
        defaults = empty_ledger_state()
        for key, value in defaults.items():
            if key not in self.state:
                self.state[key] = value.copy() if isinstance(value, dict) else list(value)

    def _load(self) -> None:
        if not self.path.exists():
            self.state = empty_ledger_state()
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            self._ensure_shape()
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load ledger from %s: %s", self.path, exc)
            self.state = empty_ledger_state()

    def _save(self) -> None:
        if self._save_callback:
            self._save_callback()
            return
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        tmp_path.replace(self.path)

    def log_pushups(
        self, user_id: int, count: int, when: Optional[datetime] = None
    ) -> dict:
        if count <= 0:
            raise ValueError("Count must be positive.")
        if user_id not in self.users:
            raise ValueError("Unknown user.")
        timestamp = when or datetime.now(timezone.utc)
        day_key = timestamp.date().isoformat()
        self.state.setdefault("pushups", {})
        day_totals: Dict[str, int] = self.state["pushups"].setdefault(day_key, {})
        user_key = str(user_id)
        day_totals[user_key] = int(day_totals.get(user_key, 0)) + int(count)
        record = {
            "user_id": user_id,
            "user_name": self.users[user_id].name,
            "count": int(count),
            "total_for_day": day_totals[user_key],
            "date": day_key,
            "created_at": timestamp.isoformat(),
        }
        self._save()
        return record

    def pushups_for_date(self, day_key: str) -> Dict[int, int]:
        pushups = self.state.get("pushups", {}).get(day_key, {})
        totals: Dict[int, int] = {}
        for raw_user_id, count in pushups.items():
            try:
                uid = int(raw_user_id)
            except (TypeError, ValueError):
                continue
            totals[uid] = int(count)
        return totals

    def add_expense(
        self,
        payer_id: int,
        amount: float,
        currency: str,
        description: str,
        amount_base: float,
        base_currency: str,
    ) -> dict:
        expense = {
            "payer_id": payer_id,
            "payer_name": self.users[payer_id].name,
            "amount": round(amount, 2),
            "currency": currency,
            "amount_base": round(amount_base, 2),
            "base_currency": base_currency,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "expense",
        }
        self.state.setdefault("expenses", []).append(expense)
        self._save()
        return expense

    def record_settlement(
        self,
        payer_id: int,
        receiver_id: int,
        amount_base: float,
        base_currency: str,
        comment: str,
        balances_before: Dict[int, float],
    ) -> dict:
        settlement = {
            "payer_id": payer_id,
            "payer_name": self.users[payer_id].name,
            "receiver_id": receiver_id,
            "receiver_name": self.users[receiver_id].name,
            "amount": round(amount_base, 2),
            "currency": base_currency,
            "amount_base": round(amount_base, 2),
            "base_currency": base_currency,
            "comment": comment,
            "balances_before": balances_before,
            "cleared_expenses": len(self.state.get("expenses", [])),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "settlement",
        }
        self.state.setdefault("settlements", []).append(settlement)
        self.state["expenses"] = []
        self._save()
        return settlement

    def balances(self) -> Dict[int, float]:
        net: Dict[int, float] = {uid: 0.0 for uid in self.users}
        expenses = self.state.get("expenses", [])
        if not expenses or not net:
            return net

        split_count = len(net)
        for exp in expenses:
            amount = float(exp.get("amount_base", exp.get("amount", 0.0)))
            payer_id = int(exp["payer_id"])
            share = amount / split_count
            for uid in net:
                if uid == payer_id:
                    net[uid] += amount - share
                else:
                    net[uid] -= share
        return net

    def last_entries(self, limit: int = 5) -> List[dict]:
        expenses = self.state.get("expenses", [])
        settlements = self.state.get("settlements", [])
        enriched: List[dict] = []
        for exp in expenses:
            copy = {**exp}
            copy.setdefault("type", "expense")
            enriched.append(copy)
        for sett in settlements:
            copy = {**sett}
            copy.setdefault("type", "settlement")
            enriched.append(copy)
        return sorted(enriched, key=lambda e: e.get("created_at", ""))[-limit:]

    def create_agreement(self, creator: UserConfig, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Use: /agree <agreement text>")
        agreements = self.state.setdefault("agreements", [])
        agreement = {
            "id": next_item_id(agreements, "a"),
            "text": text,
            "creator_id": creator.id,
            "creator_name": creator.name,
            "accepted_by": [creator.id],
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        agreements.append(agreement)
        self._update_agreement_status(agreement)
        self._save()
        return agreement

    def accept_agreement(self, agreement_id: str, user: UserConfig) -> dict:
        agreement = self.find_agreement(agreement_id)
        if not agreement:
            raise ValueError(f"Agreement {agreement_id} was not found.")
        if agreement.get("status") == "cancelled":
            raise ValueError(f"Agreement {agreement_id} is cancelled.")
        accepted = agreement.setdefault("accepted_by", [])
        if user.id not in accepted:
            accepted.append(user.id)
        self._update_agreement_status(agreement)
        self._save()
        return agreement

    def latest_pending_agreement(self) -> Optional[dict]:
        for agreement in reversed(self.state.get("agreements", [])):
            if agreement.get("status") == "pending":
                return agreement
        return None

    def find_agreement(self, agreement_id: str) -> Optional[dict]:
        wanted = agreement_id.strip().lower()
        for agreement in self.state.get("agreements", []):
            if str(agreement.get("id", "")).lower() == wanted:
                return agreement
        return None

    def _update_agreement_status(self, agreement: dict) -> None:
        participant_ids = set(self.users)
        accepted = {int(uid) for uid in agreement.get("accepted_by", [])}
        agreement["status"] = "active" if participant_ids and participant_ids <= accepted else "pending"

    def create_challenge(self, creator: UserConfig, title: str, target: Optional[int]) -> dict:
        title = title.strip()
        if not title:
            raise ValueError("Use: /challenge [target] <title>")
        challenges = self.state.setdefault("challenges", [])
        challenge = {
            "id": next_item_id(challenges, "c"),
            "title": title,
            "target": target,
            "creator_id": creator.id,
            "creator_name": creator.name,
            "scores": {},
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        challenges.append(challenge)
        self._save()
        return challenge

    def add_challenge_score(self, challenge_id: str, user: UserConfig, amount: int) -> dict:
        if amount <= 0:
            raise ValueError("Score must be positive.")
        challenge = self.find_challenge(challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} was not found.")
        if challenge.get("status") != "active":
            raise ValueError(f"Challenge {challenge_id} is not active.")
        scores = challenge.setdefault("scores", {})
        user_key = str(user.id)
        scores[user_key] = int(scores.get(user_key, 0)) + amount
        self._save()
        return challenge

    def latest_active_challenge(self) -> Optional[dict]:
        for challenge in reversed(self.state.get("challenges", [])):
            if challenge.get("status") == "active":
                return challenge
        return None

    def find_challenge(self, challenge_id: str) -> Optional[dict]:
        wanted = challenge_id.strip().lower()
        for challenge in self.state.get("challenges", []):
            if str(challenge.get("id", "")).lower() == wanted:
                return challenge
        return None


class InstanceStore:
    def __init__(self, path: Path, default_users: List[UserConfig]):
        self.path = path
        self.default_users = default_users
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = {
            "schema_version": 2,
            "default_instance_id": "default",
            "instances": {},
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._ensure_default_instance(empty_ledger_state())
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load instance store from %s: %s", self.path, exc)
            self._ensure_default_instance(empty_ledger_state())
            return

        if isinstance(raw, dict) and isinstance(raw.get("instances"), dict):
            self.state = raw
        else:
            legacy = raw if isinstance(raw, dict) else empty_ledger_state()
            self.state = {
                "schema_version": 2,
                "default_instance_id": "default",
                "instances": {},
            }
            self._ensure_default_instance(legacy)

        self.state.setdefault("schema_version", 2)
        self.state.setdefault("default_instance_id", "default")
        self.state.setdefault("instances", {})
        self._ensure_default_instance()

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        tmp_path.replace(self.path)

    def _ensure_default_instance(self, ledger_state: Optional[Dict[str, Any]] = None) -> None:
        self.ensure_instance(
            "default",
            "Default",
            chat_id=None,
            users=self.default_users,
            ledger_state=ledger_state,
            save=False,
        )

    def ensure_instance(
        self,
        instance_id: str,
        name: str,
        chat_id: Optional[int],
        users: Optional[List[UserConfig]] = None,
        ledger_state: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        instances = self.state.setdefault("instances", {})
        instance = instances.get(instance_id)
        if instance is None:
            instance = ledger_state or empty_ledger_state()
            instance["id"] = instance_id
            instance["name"] = name
            instance["chat_id"] = chat_id
            instance["members"] = [
                {"id": user.id, "name": user.name} for user in (users or [])
            ]
            instances[instance_id] = instance
        else:
            instance.setdefault("id", instance_id)
            instance.setdefault("name", name)
            instance.setdefault("chat_id", chat_id)
            instance.setdefault("members", [])
            if users and not instance["members"]:
                instance["members"] = [
                    {"id": user.id, "name": user.name} for user in users
                ]
        for key, value in empty_ledger_state().items():
            if key not in instance:
                instance[key] = value.copy() if isinstance(value, dict) else list(value)
        if save:
            self._save()
        return instance

    def instance_id_for_update(self, update: Any, context: ContextTypes.DEFAULT_TYPE) -> str:
        chat = getattr(update, "effective_chat", None)
        if chat is None and getattr(update, "message", None):
            chat = update.message.chat
        if chat and chat.type != "private":
            instance_id = f"chat:{chat.id}"
            self.ensure_instance(instance_id, chat.title or str(chat.id), chat.id)
            return instance_id
        return context.user_data.get("active_instance_id", self.state["default_instance_id"])

    def add_member(self, instance_id: str, user_id: int, name: str) -> None:
        instance = self.state["instances"][instance_id]
        members = instance.setdefault("members", [])
        for member in members:
            if int(member["id"]) == user_id:
                member["name"] = name
                self._save()
                return
        members.append({"id": user_id, "name": name})
        self._save()

    def remove_member(self, instance_id: str, user_id: int) -> None:
        instance = self.state["instances"][instance_id]
        members = instance.setdefault("members", [])
        instance["members"] = [m for m in members if int(m["id"]) != user_id]
        self._save()

    def users_for(self, instance_id: str) -> List[UserConfig]:
        instance = self.state["instances"][instance_id]
        users = []
        for member in instance.get("members", []):
            try:
                users.append(UserConfig(id=int(member["id"]), name=str(member["name"])))
            except (KeyError, TypeError, ValueError):
                continue
        return users

    def ledger_for(self, instance_id: str) -> Ledger:
        instance = self.state["instances"][instance_id]
        return Ledger(self.path, self.users_for(instance_id), state=instance, save_callback=self._save)

    def config_for(self, base_config: BotConfig, instance_id: str) -> BotConfig:
        instance = self.state["instances"][instance_id]
        return BotConfig(
            token=base_config.token,
            users=self.users_for(instance_id),
            data_path=base_config.data_path,
            base_currency=base_config.base_currency,
            rates_to_base=base_config.rates_to_base,
            healthcheck_chat_id=base_config.healthcheck_chat_id,
            instance_id=instance_id,
            instance_name=str(instance.get("name") or instance_id),
        )

    def user_instances(self, user_id: int) -> List[Dict[str, Any]]:
        found = []
        for instance in self.state.get("instances", {}).values():
            for member in instance.get("members", []):
                if int(member.get("id", 0)) == user_id:
                    found.append(instance)
                    break
        return found

    def all_instance_ids(self) -> List[str]:
        return list(self.state.get("instances", {}).keys())


# ------------------------------ utilities ------------------------------- #


AMOUNT_PATTERN = re.compile(
    r"^\s*([-+]?\d+(?:[.,]\d{1,2})?)(?:\s+([A-Za-z]{3}))?\s+(.+)$"
)
SETTLEMENT_PATTERN = re.compile(r"^\s*(.*)$")
PUSHUPS_PATTERN = re.compile(r"^\s*(\d+)\s*(?:push[-\s]?ups?)?\s*$", re.IGNORECASE)
CHALLENGE_PATTERN = re.compile(r"^\s*(?:(\d+)\s+)?(.+?)\s*$")


def next_item_id(items: List[dict], prefix: str) -> str:
    highest = 0
    for item in items:
        raw = str(item.get("id", ""))
        if not raw.startswith(prefix):
            continue
        try:
            highest = max(highest, int(raw[len(prefix) :]))
        except ValueError:
            continue
    return f"{prefix}{highest + 1}"


def normalize_currency(code: Optional[str]) -> str:
    if not code:
        return ""
    return code.strip().upper()


def parse_expense_text(text: str) -> Tuple[float, Optional[str], str]:
    match = AMOUNT_PATTERN.match(text or "")
    if not match:
        raise ValueError("Use: <amount> [CUR] <description> (e.g. 23.50 MAD dinner)")
    amount_raw, currency_raw, description = match.groups()
    currency = normalize_currency(currency_raw) or None
    amount = float(amount_raw.replace(",", "."))
    if amount <= 0:
        raise ValueError("Amount must be positive.")
    description = description.strip()
    if not description:
        raise ValueError("Add a short description after the amount.")
    return round(amount, 2), currency, description


def parse_settlement_text(text: str) -> str:
    match = SETTLEMENT_PATTERN.match(text or "")
    if not match:
        raise ValueError("Use: /settle [comment]")
    comment_raw = match.group(1)
    comment = (comment_raw or "").strip() or "settlement"
    return comment


def parse_pushups_text(text: str) -> int:
    match = PUSHUPS_PATTERN.match(text or "")
    if not match:
        raise ValueError("Send push-ups as a number (e.g. 25).")
    count = int(match.group(1))
    if count <= 0:
        raise ValueError("Count must be positive.")
    return count


def parse_challenge_text(text: str) -> Tuple[Optional[int], str]:
    match = CHALLENGE_PATTERN.match(text or "")
    if not match:
        raise ValueError("Use: /challenge [target] <title>")
    target_raw, title = match.groups()
    target = int(target_raw) if target_raw else None
    return target, title.strip()


def to_base(amount: float, currency: str, config: BotConfig) -> float:
    cur = normalize_currency(currency)
    if not cur:
        raise ValueError("Currency code is missing.")
    if cur == config.base_currency:
        return amount
    rate = config.rates_to_base.get(cur)
    if rate is None:
        raise ValueError(
            f"No rate for {cur}->{config.base_currency}. "
            f"Set RATE_{cur}_TO_BASE in the environment."
        )
    return amount * rate


def format_balance_lines(net: Dict[int, float], users: List[UserConfig], base_currency: str) -> str:
    lines = []
    for user in users:
        balance = net.get(user.id, 0.0)
        if balance > 0.01:
            lines.append(f"{user.name} is owed {balance:.2f} {base_currency}.")
        elif balance < -0.01:
            lines.append(f"{user.name} owes {-balance:.2f} {base_currency}.")
        else:
            lines.append(f"{user.name} is settled.")

    if len(users) == 2:
        diff = net.get(users[0].id, 0.0)
        if diff > 0.01:
            lines.append(
                f"{users[1].name} owes {users[0].name} {diff:.2f} {base_currency}."
            )
        elif diff < -0.01:
            lines.append(
                f"{users[0].name} owes {users[1].name} {abs(diff):.2f} {base_currency}."
            )
        else:
            lines.append("All square. 🎉")
    return "\n".join(lines)


def format_expense_line(expense: dict, base_currency: str) -> str:
    created = expense.get("created_at", "")
    try:
        timestamp = datetime.fromisoformat(created).astimezone(timezone.utc)
        created_str = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        created_str = created or "unknown time"
    desc = expense.get("description", "")
    payer = expense.get("payer_name") or str(expense.get("payer_id"))
    amount = float(expense.get("amount", 0))
    currency = expense.get("currency", base_currency)
    amount_base = expense.get("amount_base")
    base_note = ""
    if amount_base is not None and currency != base_currency:
        base_note = f" (base {amount_base:.2f} {base_currency})"
    return f"{created_str}: {payer} paid {amount:.2f} {currency}{base_note} for {desc}"


def format_entry_line(entry: dict, base_currency: str) -> str:
    entry_type = entry.get("type", "expense")
    if entry_type == "settlement":
        created = entry.get("created_at", "")
        try:
            ts = datetime.fromisoformat(created).astimezone(timezone.utc)
            created_str = ts.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            created_str = created or "unknown time"
        payer = entry.get("payer_name") or str(entry.get("payer_id"))
        receiver = entry.get("receiver_name") or str(entry.get("receiver_id", ""))
        amount = float(entry.get("amount", 0))
        currency = entry.get("currency", base_currency)
        amount_base = entry.get("amount_base")
        base_note = ""
        if amount_base is not None and currency != base_currency:
            base_note = f" (base {amount_base:.2f} {base_currency})"
        comment = entry.get("comment", "")
        cleared = entry.get("cleared_expenses", 0)
        return (
            f"{created_str}: Settlement {payer} paid {receiver} {amount:.2f} {currency}{base_note}; "
            f"cleared {cleared} expenses. Note: {comment or '—'}"
        )
    return format_expense_line(entry, base_currency)


def format_pushup_standings(
    pushups: Dict[int, int], config: BotConfig, date_label: str
) -> str:
    if not pushups:
        return f"No push-ups logged for {date_label} yet."

    lines = [f"Push-ups for {date_label}:"]
    sorted_users = sorted(
        config.users, key=lambda u: pushups.get(u.id, 0), reverse=True
    )
    top = max(pushups.values())
    winners = [u.name for u in config.users if pushups.get(u.id, 0) == top]
    for user in sorted_users:
        lines.append(f"- {user.name}: {pushups.get(user.id, 0)}")
    if top > 0:
        if len(winners) == 1:
            lines.append(f"🏅 Winner: {winners[0]} ({top})")
        else:
            lines.append(f"🤝 Tie: {', '.join(winners)} ({top})")
    return "\n".join(lines)


def format_agreement(agreement: dict, config: BotConfig) -> str:
    accepted = {int(uid) for uid in agreement.get("accepted_by", [])}
    accepted_names = [
        user.name for user in config.users if user.id in accepted
    ] or ["Nobody yet"]
    status = agreement.get("status", "pending")
    return (
        f"{agreement.get('id')}: {agreement.get('text')}\n"
        f"Status: {status}. Accepted by: {', '.join(accepted_names)}"
    )


def format_challenge(challenge: dict, config: BotConfig) -> str:
    scores = challenge.get("scores", {})
    lines = [f"{challenge.get('id')}: {challenge.get('title')}"]
    target = challenge.get("target")
    if target:
        lines[0] += f" (target {target})"
    for user in sorted(config.users, key=lambda u: int(scores.get(str(u.id), 0)), reverse=True):
        lines.append(f"- {user.name}: {int(scores.get(str(user.id), 0))}")
    return "\n".join(lines)


def get_runtime(context: ContextTypes.DEFAULT_TYPE) -> Tuple[Ledger, BotConfig]:
    return (
        context.application.bot_data["ledger"],
        context.application.bot_data["config"],
    )


def get_runtime_for_update(
    update: Any, context: ContextTypes.DEFAULT_TYPE
) -> Tuple[Ledger, BotConfig]:
    store: InstanceStore = context.application.bot_data["store"]
    base_config: BotConfig = context.application.bot_data["base_config"]
    instance_id = store.instance_id_for_update(update, context)
    return store.ledger_for(instance_id), store.config_for(base_config, instance_id)


def user_state_key(config: BotConfig, user_id: int, name: str) -> str:
    return f"{name}:{config.instance_id}:{user_id}"


def user_from_id(user_id: int, config: BotConfig) -> Optional[UserConfig]:
    return next((u for u in config.users if u.id == user_id), None)

def available_currencies(config: BotConfig) -> List[str]:
    codes = set([config.base_currency])
    codes.update(config.rates_to_base.keys())
    return sorted(codes)


def build_currency_keyboard(config: BotConfig) -> InlineKeyboardMarkup:
    codes = available_currencies(config)
    buttons = [
        InlineKeyboardButton(code, callback_data=f"{CB_CURRENCY_PREFIX}{code}")
        for code in codes
    ]
    # one row per button for clarity
    return InlineKeyboardMarkup([[b] for b in buttons])


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("➕ Expense", callback_data=f"{CB_MENU_PREFIX}add_expense")],
            [
                InlineKeyboardButton("📊 Balance", callback_data=f"{CB_MENU_PREFIX}balance"),
                InlineKeyboardButton("🕑 History", callback_data=f"{CB_MENU_PREFIX}history"),
            ],
            [InlineKeyboardButton("✅ Settle", callback_data=f"{CB_MENU_PREFIX}settle")],
            [InlineKeyboardButton("💪 Push-ups", callback_data=f"{CB_MENU_PREFIX}pushups")],
            [InlineKeyboardButton("🏅 Standings", callback_data=f"{CB_MENU_PREFIX}pushups_standings")],
            [InlineKeyboardButton("ℹ️ Help", callback_data=f"{CB_MENU_PREFIX}help")],
        ]
    )


def pushups_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("+5", callback_data=f"{CB_PUSHUPS_ADD_PREFIX}5"),
                InlineKeyboardButton("+10", callback_data=f"{CB_PUSHUPS_ADD_PREFIX}10"),
            ],
            [
                InlineKeyboardButton("+20", callback_data=f"{CB_PUSHUPS_ADD_PREFIX}20"),
                InlineKeyboardButton("+30", callback_data=f"{CB_PUSHUPS_ADD_PREFIX}30"),
            ],
            [InlineKeyboardButton("Enter custom", callback_data=f"{CB_PUSHUPS_PREFIX}custom")],
            [InlineKeyboardButton("Back to menu", callback_data=f"{CB_MENU_PREFIX}home")],
        ]
    )


CB_CURRENCY_PREFIX = "currency:"
CB_MENU_PREFIX = "menu:"
CB_PUSHUPS_PREFIX = "pushups:"
CB_PUSHUPS_ADD_PREFIX = f"{CB_PUSHUPS_PREFIX}add:"


async def currency_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    if not data.startswith(CB_CURRENCY_PREFIX):
        return
    currency = data[len(CB_CURRENCY_PREFIX) :].strip().upper()
    pending = context.user_data.pop(
        user_state_key(config, query.from_user.id, "pending_currency"), None
    )
    if not pending:
        await query.edit_message_text("No pending entry to apply this currency to.")
        return

    kind = pending.get("kind")
    amount = pending.get("amount")
    description = pending.get("description")
    comment = pending.get("comment")
    payer_id = pending.get("payer_id")
    receiver_id = pending.get("receiver_id")
    actor = user_from_id(query.from_user.id, config)
    if not actor:
        await query.edit_message_text("You're not on the traveler list for this bot.")
        return

    if kind == "expense":
        if amount is None or description is None:
            await query.edit_message_text("Incomplete pending expense data.")
            return
        await finalize_expense(query, context, actor, amount, currency, description)
    elif kind == "settlement":
        if amount is None or comment is None or receiver_id is None or payer_id is None:
            await query.edit_message_text("Incomplete pending settlement data.")
            return
        receiver = user_from_id(receiver_id, config)
        payer = user_from_id(payer_id, config)
        if not receiver or not payer:
            await query.edit_message_text("Participants not found.")
            return
        await finalize_settlement(query, context, payer, receiver, amount, currency, comment)
    else:
        await query.edit_message_text("Unknown pending action.")

# ------------------------------ handlers -------------------------------- #


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _, config = get_runtime_for_update(update, context)
    names = " & ".join([u.name for u in config.users]) or "No participants yet"
    help_text = (
        "Shared travel bot is online.\n"
        "Add expenses by sending: `<amount> [CUR] <description>`\n"
        f"If no currency is given, I'll ask (options: {', '.join(available_currencies(config))}).\n"
        "or `/add <amount> [CUR] <description>`.\n"
        "Push-ups: `/pushups <count>` or tap the push-up buttons.\n\n"
        "Commands:\n"
        "- /add 23.50 dinner\n"
        "- /balance\n"
        "- /history\n\n"
        "- /settle [comment]  (records who owed whom, marks it paid, and clears expenses)\n\n"
        "- /join  (subscribe to this group instance)\n"
        "- /agree <text> and /accept [id]\n"
        "- /challenge [target] <title> and /score [id] <amount>\n\n"
        f"Instance: {config.instance_name} (`{config.instance_id}`)\n"
        f"Participants: {names}\n"
        "Every entry is split evenly between both people.\n\n"
        "Use the inline menu below for quick actions."
    )
    await update.message.reply_text(help_text, reply_markup=main_menu_keyboard())


async def add_expense_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("You're not on the traveler list for this bot.")
        return

    text = " ".join(context.args) if context.args else ""
    try:
        amount, currency, description = parse_expense_text(text)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    if not currency:
        context.user_data[user_state_key(config, actor.id, "pending_currency")] = {
            "kind": "expense",
            "amount": amount,
            "description": description,
            "payer_id": actor.id,
            "instance_id": config.instance_id,
        }
        await update.message.reply_text(
            "Choose a currency:",
            reply_markup=build_currency_keyboard(config),
        )
        return

    await finalize_expense(update, context, actor, amount, currency, description)


async def add_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Handle plain text messages (without commands) as expense inputs.
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    _, config = get_runtime_for_update(update, context)
    user_id = update.effective_user.id if update.effective_user else 0
    if context.user_data.pop(user_state_key(config, user_id, "awaiting_pushups_custom"), False):
        await pushups_text_handler(update, context)
        return
    # Avoid treating casual group text as bot input unless a command/button asked for it.
    if update.message.chat.type != "private":
        return
    if PUSHUPS_PATTERN.match(text):
        await pushups_text_handler(update, context)
        return
    await add_expense_handler(update, context)


async def finalize_expense(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    actor: UserConfig,
    amount: float,
    currency: str,
    description: str,
) -> None:
    ledger, config = get_runtime_for_update(update, context)
    try:
        amount_base = to_base(amount, currency, config)
    except ValueError as exc:
        if hasattr(update, "message") and update.message:
            await update.message.reply_text(str(exc))
        elif hasattr(update, "edit_message_text"):
            await update.edit_message_text(str(exc))
        return

    expense = ledger.add_expense(
        actor.id,
        amount,
        currency,
        description,
        amount_base,
        config.base_currency,
    )
    net = ledger.balances()
    balance_text = format_balance_lines(net, config.users, config.base_currency)
    reply_text = (
        f"Logged {amount:.2f} {currency} for '{description}' as paid by {actor.name}.\n\n{balance_text}"
    )
    if hasattr(update, "message") and update.message:
        await update.message.reply_text(reply_text, reply_markup=main_menu_keyboard())
    elif hasattr(update, "edit_message_text"):
        await update.edit_message_text(reply_text, reply_markup=main_menu_keyboard())


async def balance_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    net = ledger.balances()
    text = format_balance_lines(net, config.users, config.base_currency)
    await update.message.reply_text(text, reply_markup=main_menu_keyboard())


async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    entries = ledger.last_entries(limit=10)
    if not entries:
        await update.message.reply_text("No expenses yet.", reply_markup=main_menu_keyboard())
        return
    lines = [format_entry_line(exp, config.base_currency) for exp in entries]
    await update.message.reply_text(
        "Recent activity:\n" + "\n".join(lines), reply_markup=main_menu_keyboard()
    )


async def join_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message:
        return
    store: InstanceStore = context.application.bot_data["store"]
    instance_id = store.instance_id_for_update(update, context)
    if update.effective_chat and update.effective_chat.type == "private" and instance_id == "default":
        _, config = get_runtime_for_update(update, context)
        if not user_from_id(update.effective_user.id, config):
            await update.message.reply_text("Use /join inside a group instance first.")
            return
    user = update.effective_user
    name = user.full_name or user.username or str(user.id)
    store.add_member(instance_id, user.id, name)
    _, config = get_runtime_for_update(update, context)
    await update.message.reply_text(
        f"{name} joined {config.instance_name}. Participants: "
        f"{', '.join([u.name for u in config.users])}",
        reply_markup=main_menu_keyboard(),
    )


async def leave_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message:
        return
    store: InstanceStore = context.application.bot_data["store"]
    instance_id = store.instance_id_for_update(update, context)
    if instance_id == "default":
        await update.message.reply_text("The default DM instance keeps its configured users.")
        return
    store.remove_member(instance_id, update.effective_user.id)
    await update.message.reply_text("You left this instance.")


async def instance_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _, config = get_runtime_for_update(update, context)
    names = ", ".join([u.name for u in config.users]) or "No participants yet"
    await update.message.reply_text(
        f"Instance: {config.instance_name}\nID: {config.instance_id}\nParticipants: {names}"
    )


async def instances_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message:
        return
    store: InstanceStore = context.application.bot_data["store"]
    instances = store.user_instances(update.effective_user.id)
    if not instances:
        await update.message.reply_text("You are not subscribed to any instances yet.")
        return
    lines = ["Your instances:"]
    for instance in instances:
        lines.append(f"- {instance.get('name')} (`{instance.get('id')}`)")
    lines.append("In a DM, use `/use <id>` to make one active.")
    await update.message.reply_text("\n".join(lines))


async def use_instance_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.message:
        return
    if update.effective_chat and update.effective_chat.type != "private":
        await update.message.reply_text("Group chats use their own group instance automatically.")
        return
    if not context.args:
        await update.message.reply_text("Use: /use <instance_id>")
        return
    instance_id = context.args[0]
    store: InstanceStore = context.application.bot_data["store"]
    if instance_id not in store.state.get("instances", {}):
        await update.message.reply_text(f"Instance {instance_id} was not found.")
        return
    if not any(i.get("id") == instance_id for i in store.user_instances(update.effective_user.id)):
        await update.message.reply_text("You are not subscribed to that instance.")
        return
    context.user_data["active_instance_id"] = instance_id
    _, config = get_runtime_for_update(update, context)
    await update.message.reply_text(f"Active DM instance set to {config.instance_name}.")


async def agree_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("Join this instance first with /join.")
        return
    text = " ".join(context.args)
    try:
        agreement = ledger.create_agreement(actor, text)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    await update.message.reply_text(
        f"Agreement proposed.\n{format_agreement(agreement, config)}\n"
        f"Participants can accept with /accept {agreement['id']}."
    )


async def accept_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("Join this instance first with /join.")
        return
    agreement_id = context.args[0] if context.args else None
    agreement = ledger.find_agreement(agreement_id) if agreement_id else ledger.latest_pending_agreement()
    if not agreement:
        await update.message.reply_text("No pending agreement found.")
        return
    try:
        agreement = ledger.accept_agreement(str(agreement["id"]), actor)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    await update.message.reply_text(format_agreement(agreement, config))


async def agreements_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    agreements = ledger.state.get("agreements", [])[-10:]
    if not agreements:
        await update.message.reply_text("No agreements yet.")
        return
    await update.message.reply_text("\n\n".join(format_agreement(a, config) for a in agreements))


async def challenge_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("Join this instance first with /join.")
        return
    try:
        target, title = parse_challenge_text(" ".join(context.args))
        challenge = ledger.create_challenge(actor, title, target)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    await update.message.reply_text(
        f"Challenge created.\n{format_challenge(challenge, config)}\n"
        f"Log progress with /score {challenge['id']} <amount>."
    )


async def score_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("Join this instance first with /join.")
        return
    if len(context.args) == 1:
        challenge = ledger.latest_active_challenge()
        if not challenge:
            await update.message.reply_text("No active challenge found.")
            return
        challenge_id = str(challenge["id"])
        raw_amount = context.args[0]
    elif len(context.args) >= 2:
        challenge_id = context.args[0]
        raw_amount = context.args[1]
    else:
        await update.message.reply_text("Use: /score [challenge_id] <amount>")
        return
    try:
        amount = int(raw_amount)
        challenge = ledger.add_challenge_score(challenge_id, actor, amount)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    await update.message.reply_text(format_challenge(challenge, config))


async def challenges_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    challenges = ledger.state.get("challenges", [])[-10:]
    if not challenges:
        await update.message.reply_text("No challenges yet.")
        return
    await update.message.reply_text("\n\n".join(format_challenge(c, config) for c in challenges))


async def healthcheck(app: Application, config: BotConfig) -> None:
    if not config.healthcheck_chat_id:
        return
    try:
        await app.bot.send_message(
            chat_id=config.healthcheck_chat_id, text="Travel bot is running."
        )
    except Exception as exc:
        logger.warning("Healthcheck message failed: %s", exc)


async def settle_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("You're not on the traveler list for this bot.")
        return

    if len(config.users) != 2:
        await update.message.reply_text("Settlement reset currently needs exactly two subscribed users.")
        return
    other = next((u for u in config.users if u.id != actor.id), None)

    # Compute balances before clearing
    net_before = ledger.balances()
    if all(abs(v) < 0.01 for v in net_before.values()):
        await update.message.reply_text("Balances are already settled.")
        return

    diff = net_before.get(actor.id, 0.0)
    if diff > 0:  # actor is owed -> other pays actor
        payer = other
        receiver = actor
        amount_base = diff
    elif diff < 0:  # actor owes -> actor pays other
        payer = actor
        receiver = other
        amount_base = abs(diff)
    else:
        # actor is even, so other must be owed
        payer = actor
        receiver = other
        amount_base = abs(net_before.get(other.id, 0.0))

    text = " ".join(context.args) if context.args else (update.message.text or "")
    try:
        comment = parse_settlement_text(text)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    ledger.record_settlement(
        payer_id=payer.id,
        receiver_id=receiver.id,
        amount_base=round(amount_base, 2),
        base_currency=config.base_currency,
        comment=comment,
        balances_before=net_before,
    )
    await update.message.reply_text(
        f"Recorded settlement: {payer.name} paid {receiver.name} {amount_base:.2f} {config.base_currency} "
        f"('{comment}'). Expenses reset. Balances are now 0 for everyone.",
        reply_markup=main_menu_keyboard(),
    )


async def finalize_settlement(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    actor: UserConfig,
    receiver: UserConfig,
    amount: float,
    currency: str,
    comment: str,
) -> None:
    # Deprecated path (kept for inline currency callback safety); now settlements are auto base currency
    ledger, config = get_runtime_for_update(update, context)
    net_before = ledger.balances()
    try:
        amount_base = to_base(amount, currency, config)
    except ValueError as exc:
        if hasattr(update, "message") and update.message:
            await update.message.reply_text(str(exc))
        elif hasattr(update, "edit_message_text"):
            await update.edit_message_text(str(exc))
        return
    ledger.record_settlement(
        payer_id=actor.id,
        receiver_id=receiver.id,
        amount_base=amount_base,
        base_currency=config.base_currency,
        comment=comment,
        balances_before=net_before,
    )
    reply = (
        f"Recorded settlement: {actor.name} paid {receiver.name} {amount_base:.2f} {config.base_currency} "
        f"('{comment}'). Expenses reset. Balances are now 0 for everyone."
    )
    if hasattr(update, "message") and update.message:
        await update.message.reply_text(reply, reply_markup=main_menu_keyboard())
    elif hasattr(update, "edit_message_text"):
        await update.edit_message_text(reply, reply_markup=main_menu_keyboard())


# ------------------------- push-up and menu flows ----------------------- #


async def pushups_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("You're not on the traveler list for this bot.")
        return

    text = " ".join(context.args) if context.args else (update.message.text or "")
    try:
        count = parse_pushups_text(text)
    except ValueError as exc:
        await update.message.reply_text(str(exc), reply_markup=pushups_keyboard())
        return

    record = ledger.log_pushups(actor.id, count)
    day_key = record["date"]
    totals = ledger.pushups_for_date(day_key)
    standings = format_pushup_standings(totals, config, day_key)
    await update.message.reply_text(
        f"Logged {count} push-ups for today.\n"
        f"Total for you today: {record['total_for_day']}.\n\n{standings}",
        reply_markup=pushups_keyboard(),
    )


async def pushups_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("You're not on the traveler list for this bot.")
        return
    text = update.message.text or ""
    try:
        count = parse_pushups_text(text)
    except ValueError as exc:
        await update.message.reply_text(
            f"{exc}\nUse /pushups <count> or the buttons.",
            reply_markup=pushups_keyboard(),
        )
        return
    record = ledger.log_pushups(actor.id, count)
    day_key = record["date"]
    totals = ledger.pushups_for_date(day_key)
    standings = format_pushup_standings(totals, config, day_key)
    await update.message.reply_text(
        f"Logged {count} push-ups for today.\n"
        f"Total for you today: {record['total_for_day']}.\n\n{standings}",
        reply_markup=pushups_keyboard(),
    )


async def pushups_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    if not data.startswith(CB_PUSHUPS_PREFIX):
        return
    actor = user_from_id(query.from_user.id, config)
    if not actor:
        if query.message:
            await query.message.reply_text("You're not on the traveler list for this bot.")
        return

    action = data[len(CB_PUSHUPS_PREFIX) :]
    if action.startswith("add:"):
        _, _, raw_count = action.partition(":")
        try:
            count = int(raw_count)
        except ValueError:
            if query.message:
                await query.message.reply_text("Invalid push-up amount.", reply_markup=pushups_keyboard())
            return
        record = ledger.log_pushups(actor.id, count)
        day_key = record["date"]
        totals = ledger.pushups_for_date(day_key)
        standings = format_pushup_standings(totals, config, day_key)
        if query.message:
            await query.message.reply_text(
                f"Added {count} push-ups for today.\n"
                f"Your total: {record['total_for_day']}.\n\n{standings}",
                reply_markup=main_menu_keyboard(),
            )
    elif action == "custom":
        context.user_data[user_state_key(config, actor.id, "awaiting_pushups_custom")] = True
        if query.message:
            await query.message.reply_text(
                "Send the number of push-ups to log for today (e.g. 25).",
                reply_markup=pushups_keyboard(),
            )


async def settle_via_menu(
    update: Update, context: ContextTypes.DEFAULT_TYPE, actor: UserConfig
) -> None:
    query = update.callback_query
    ledger, config = get_runtime_for_update(update, context)
    if len(config.users) != 2:
        if query.message:
            await query.message.reply_text(
                "Settlement reset currently needs exactly two subscribed users.",
                reply_markup=main_menu_keyboard(),
            )
        return
    other = next((u for u in config.users if u.id != actor.id), None)

    net_before = ledger.balances()
    if all(abs(v) < 0.01 for v in net_before.values()):
        if query.message:
            await query.message.reply_text(
                "Balances are already settled.", reply_markup=main_menu_keyboard()
            )
        return

    diff = net_before.get(actor.id, 0.0)
    if diff > 0:
        payer = other
        receiver = actor
        amount_base = diff
    elif diff < 0:
        payer = actor
        receiver = other
        amount_base = abs(diff)
    else:
        payer = actor
        receiver = other
        amount_base = abs(net_before.get(other.id, 0.0))

    ledger.record_settlement(
        payer_id=payer.id,
        receiver_id=receiver.id,
        amount_base=round(amount_base, 2),
        base_currency=config.base_currency,
        comment="settled via menu",
        balances_before=net_before,
    )
    if query.message:
        await query.message.reply_text(
            f"Recorded settlement: {payer.name} paid {receiver.name} {amount_base:.2f} {config.base_currency}. "
            "Balances reset to 0 for everyone.",
            reply_markup=main_menu_keyboard(),
        )


async def send_pushups_prompt(
    messageable, ledger: Ledger, config: BotConfig
) -> None:
    today_key = datetime.now(timezone.utc).date().isoformat()
    totals = ledger.pushups_for_date(today_key)
    standings = format_pushup_standings(totals, config, today_key)
    await messageable.reply_text(
        f"{standings}\n\nUse the buttons to add push-ups for today.",
        reply_markup=pushups_keyboard(),
    )


async def send_pushups_standings(
    messageable, ledger: Ledger, config: BotConfig, day_key: Optional[str] = None
) -> None:
    target_day = day_key or datetime.now(timezone.utc).date().isoformat()
    totals = ledger.pushups_for_date(target_day)
    standings = format_pushup_standings(totals, config, target_day)
    await messageable.reply_text(standings, reply_markup=main_menu_keyboard())


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    if not data.startswith(CB_MENU_PREFIX):
        return
    actor = user_from_id(query.from_user.id, config)
    if not actor:
        if query.message:
            await query.message.reply_text("You're not on the traveler list for this bot.")
        return

    action = data[len(CB_MENU_PREFIX) :]
    if action == "add_expense":
        if query.message:
            await query.message.reply_text(
                "Send: <amount> [CUR] <description> to log an expense.",
                reply_markup=main_menu_keyboard(),
            )
    elif action == "balance":
        net = ledger.balances()
        text = format_balance_lines(net, config.users, config.base_currency)
        if query.message:
            await query.message.reply_text(text, reply_markup=main_menu_keyboard())
    elif action == "history":
        entries = ledger.last_entries(limit=10)
        if query.message:
            if not entries:
                await query.message.reply_text("No expenses yet.", reply_markup=main_menu_keyboard())
            else:
                lines = [format_entry_line(exp, config.base_currency) for exp in entries]
                await query.message.reply_text(
                    "Recent activity:\n" + "\n".join(lines),
                    reply_markup=main_menu_keyboard(),
                )
    elif action == "settle":
        await settle_via_menu(update, context, actor)
    elif action == "pushups":
        if query.message:
            await send_pushups_prompt(query.message, ledger, config)
    elif action == "pushups_standings":
        if query.message:
            await send_pushups_standings(query.message, ledger, config)
    elif action in {"help", "home"}:
        if query.message:
            await query.message.reply_text(
                "Menu ready. Pick an action:",
                reply_markup=main_menu_keyboard(),
            )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Menu:", reply_markup=main_menu_keyboard()
        )


async def pushups_daily_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    store: InstanceStore = context.application.bot_data["store"]
    base_config: BotConfig = context.application.bot_data["base_config"]
    day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    day_key = day.isoformat()
    for instance_id in store.all_instance_ids():
        ledger = store.ledger_for(instance_id)
        config = store.config_for(base_config, instance_id)
        totals = ledger.pushups_for_date(day_key)
        if not totals:
            continue
        standings = format_pushup_standings(totals, config, day_key)
        recipients = [u.id for u in config.users]
        if config.healthcheck_chat_id and instance_id == "default":
            recipients.append(config.healthcheck_chat_id)

        for chat_id in recipients:
            try:
                await context.bot.send_message(chat_id=chat_id, text=standings)
            except Exception as exc:
                logger.warning("Failed to send daily push-up report to %s: %s", chat_id, exc)


# ------------------------------ bootstrap -------------------------------- #


async def on_startup(app: Application) -> None:
    config: BotConfig = app.bot_data["base_config"]
    await healthcheck(app, config)
    store: InstanceStore = app.bot_data["store"]
    logger.info("Bot started with %d instance(s).", len(store.all_instance_ids()))


def build_application(config: BotConfig, store: InstanceStore) -> Application:
    application = (
        Application.builder()
        .token(config.token)
        .post_init(on_startup)
        .build()
    )
    application.bot_data["base_config"] = config
    application.bot_data["store"] = store
    application.bot_data["config"] = store.config_for(config, "default")
    application.bot_data["ledger"] = store.ledger_for("default")

    application.job_queue.run_daily(
        pushups_daily_report,
        time=time(hour=0, minute=0, tzinfo=timezone.utc),
        name="pushups-daily-report",
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("join", join_handler))
    application.add_handler(CommandHandler("leave", leave_handler))
    application.add_handler(CommandHandler("instance", instance_handler))
    application.add_handler(CommandHandler("instances", instances_handler))
    application.add_handler(CommandHandler("use", use_instance_handler))
    application.add_handler(CommandHandler("agree", agree_handler))
    application.add_handler(CommandHandler("accept", accept_handler))
    application.add_handler(CommandHandler("agreements", agreements_handler))
    application.add_handler(CommandHandler("challenge", challenge_handler))
    application.add_handler(CommandHandler("score", score_handler))
    application.add_handler(CommandHandler("challenges", challenges_handler))
    application.add_handler(CommandHandler("add", add_expense_handler))
    application.add_handler(CommandHandler("balance", balance_handler))
    application.add_handler(CommandHandler("history", history_handler))
    application.add_handler(CommandHandler("settle", settle_handler))
    application.add_handler(CommandHandler("pushups", pushups_command_handler))
    application.add_handler(
        CallbackQueryHandler(
            menu_callback,
            pattern=f"^{CB_MENU_PREFIX}",
        )
    )
    application.add_handler(
        CallbackQueryHandler(
            pushups_callback,
            pattern=f"^{CB_PUSHUPS_PREFIX}",
        )
    )
    application.add_handler(
        CallbackQueryHandler(
            currency_selected,
            pattern=f"^{CB_CURRENCY_PREFIX}",
        )
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, add_text_handler))
    return application


def main() -> None:
    config = load_config()
    store = InstanceStore(config.data_path, config.users)
    application = build_application(config, store)
    application.run_polling(stop_signals=None)


if __name__ == "__main__":
    main()
