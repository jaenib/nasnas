"""
Telegram bot for shared group life: expenses, agreements, and challenges.

Each chat gets its own isolated instance (the legacy two-user DM setup is the
`default` instance). Per instance:
- Expenses: `/add 23.50 lunch`, `/balance`, `/history`, `/settle` — split
  evenly between members who opted in with `/join`.
- Agreements: `/agree <text>` proposes a pact; it activates once two people
  accept (inline buttons or /accept, /decline). `/breach` tracks strikes.
- Challenges: `/challenge 100 push-ups for 7d` with live leaderboards,
  streaks, progress bars, winner detection, and deadline sweeps.
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
            "declined_by": [],
            "breaches": [],
            "participants": {str(creator.id): creator.name},
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        agreements.append(agreement)
        self._update_agreement_status(agreement)
        self._save()
        return agreement

    def respond_agreement(self, agreement_id: str, user: UserConfig, accept: bool) -> dict:
        agreement = self.find_agreement(agreement_id)
        if not agreement:
            raise ValueError(f"Agreement {agreement_id} was not found.")
        if agreement.get("status") in ("revoked", "cancelled"):
            raise ValueError(f"Agreement {agreement_id} was revoked.")
        accepted = agreement.setdefault("accepted_by", [])
        declined = agreement.setdefault("declined_by", [])
        agreement.setdefault("participants", {})[str(user.id)] = user.name
        if accept:
            if user.id in declined:
                declined.remove(user.id)
            if user.id not in accepted:
                accepted.append(user.id)
        else:
            if user.id in accepted:
                accepted.remove(user.id)
            if user.id not in declined:
                declined.append(user.id)
        self._update_agreement_status(agreement)
        self._save()
        return agreement

    def revoke_agreement(self, agreement_id: str, user: UserConfig) -> dict:
        agreement = self.find_agreement(agreement_id)
        if not agreement:
            raise ValueError(f"Agreement {agreement_id} was not found.")
        if agreement.get("status") in ("revoked", "cancelled"):
            raise ValueError(f"Agreement {agreement_id} is already revoked.")
        if int(agreement.get("creator_id", 0)) != user.id:
            raise ValueError("Only the creator can revoke an agreement.")
        agreement["status"] = "revoked"
        agreement["revoked_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        return agreement

    def record_breach(
        self, agreement_id: str, offender: UserConfig, reporter: UserConfig, note: str
    ) -> dict:
        agreement = self.find_agreement(agreement_id)
        if not agreement:
            raise ValueError(f"Agreement {agreement_id} was not found.")
        if agreement.get("status") != "active":
            raise ValueError("Breaches can only be recorded against active agreements.")
        participants = agreement.setdefault("participants", {})
        participants.setdefault(str(offender.id), offender.name)
        participants.setdefault(str(reporter.id), reporter.name)
        agreement.setdefault("breaches", []).append(
            {
                "user_id": offender.id,
                "user_name": offender.name,
                "reported_by": reporter.id,
                "reporter_name": reporter.name,
                "note": note.strip(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save()
        return agreement

    def latest_open_agreement(self) -> Optional[dict]:
        for agreement in reversed(self.state.get("agreements", [])):
            if agreement.get("status") == "pending":
                return agreement
        for agreement in reversed(self.state.get("agreements", [])):
            if agreement.get("status") == "active":
                return agreement
        return None

    def latest_active_agreement(self) -> Optional[dict]:
        for agreement in reversed(self.state.get("agreements", [])):
            if agreement.get("status") == "active":
                return agreement
        return None

    def find_agreement(self, agreement_id: str) -> Optional[dict]:
        wanted = str(agreement_id or "").strip().lower()
        if not wanted:
            return None
        for agreement in self.state.get("agreements", []):
            if str(agreement.get("id", "")).lower() == wanted:
                return agreement
        return None

    def _update_agreement_status(self, agreement: dict) -> None:
        # An agreement binds its acceptors; it takes effect once at least two
        # people are in. Revoked agreements never change status again.
        if agreement.get("status") in ("revoked", "cancelled"):
            return
        accepted = {int(uid) for uid in agreement.get("accepted_by", [])}
        agreement["status"] = "active" if len(accepted) >= 2 else "pending"
        if agreement["status"] == "active" and not agreement.get("activated_at"):
            agreement["activated_at"] = datetime.now(timezone.utc).isoformat()

    def create_challenge(
        self,
        creator: UserConfig,
        title: str,
        target: Optional[int],
        deadline: Optional[str] = None,
    ) -> dict:
        title = title.strip()
        if not title:
            raise ValueError("Use: /challenge [target] <title> [for 7d | until YYYY-MM-DD]")
        challenges = self.state.setdefault("challenges", [])
        challenge = {
            "id": next_item_id(challenges, "c"),
            "title": title,
            "target": target,
            "deadline": deadline,
            "creator_id": creator.id,
            "creator_name": creator.name,
            "scores": {},
            "daily": {},
            "participants": {str(creator.id): creator.name},
            "winner_ids": [],
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        challenges.append(challenge)
        self._save()
        return challenge

    def add_challenge_score(
        self, challenge_id: str, user: UserConfig, amount: int
    ) -> Tuple[dict, bool]:
        """Apply a score change. Returns (challenge, completed_now).

        Negative amounts correct mistakes; totals never go below zero.
        Hitting the target completes the challenge on the spot.
        """
        if amount == 0:
            raise ValueError("Score change cannot be zero.")
        challenge = self.find_challenge(challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} was not found.")
        today = datetime.now(timezone.utc).date()
        if challenge.get("status") == "active" and challenge_expired(challenge, today):
            self._finalize_challenge(challenge, "ended")
            self._save()
            raise ValueError(
                f"Challenge {challenge['id']} hit its deadline "
                f"({challenge.get('deadline')}) and is now closed. See /challenges done."
            )
        if challenge.get("status") != "active":
            raise ValueError(f"Challenge {challenge_id} is not active.")
        scores = challenge.setdefault("scores", {})
        user_key = str(user.id)
        old_total = int(scores.get(user_key, 0))
        new_total = max(0, old_total + int(amount))
        applied = new_total - old_total
        scores[user_key] = new_total
        day_key = today.isoformat()
        daily = challenge.setdefault("daily", {}).setdefault(user_key, {})
        daily[day_key] = max(0, int(daily.get(day_key, 0)) + applied)
        challenge.setdefault("participants", {})[user_key] = user.name
        completed = False
        target = challenge.get("target")
        if target and new_total >= int(target):
            self._finalize_challenge(challenge, "completed", winner_ids=[user.id])
            completed = True
        self._save()
        return challenge, completed

    def finish_challenge(self, challenge_id: str, user: UserConfig) -> dict:
        challenge = self.find_challenge(challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} was not found.")
        if challenge.get("status") != "active":
            raise ValueError(f"Challenge {challenge_id} is not active.")
        if int(challenge.get("creator_id", 0)) != user.id:
            raise ValueError("Only the creator can end a challenge early.")
        self._finalize_challenge(challenge, "ended")
        self._save()
        return challenge

    def expire_due_challenges(self, today) -> List[dict]:
        ended = []
        for challenge in self.state.get("challenges", []):
            if challenge.get("status") == "active" and challenge_expired(challenge, today):
                self._finalize_challenge(challenge, "ended")
                ended.append(challenge)
        if ended:
            self._save()
        return ended

    def _finalize_challenge(
        self, challenge: dict, status: str, winner_ids: Optional[List[int]] = None
    ) -> None:
        if winner_ids is None:
            scores: Dict[int, int] = {}
            for raw_uid, value in (challenge.get("scores", {}) or {}).items():
                try:
                    scores[int(raw_uid)] = int(value)
                except (TypeError, ValueError):
                    continue
            top = max(scores.values(), default=0)
            winner_ids = [uid for uid, value in scores.items() if value == top and top > 0]
        challenge["status"] = status
        challenge["winner_ids"] = winner_ids
        challenge["finished_at"] = datetime.now(timezone.utc).isoformat()

    def latest_active_challenge(self) -> Optional[dict]:
        for challenge in reversed(self.state.get("challenges", [])):
            if challenge.get("status") == "active":
                return challenge
        return None

    def find_challenge(self, challenge_id: str) -> Optional[dict]:
        wanted = str(challenge_id or "").strip().lower()
        if not wanted:
            return None
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
CHALLENGE_UNTIL_PATTERN = re.compile(r"\s+until\s+(\d{4}-\d{2}-\d{2})\s*$", re.IGNORECASE)
CHALLENGE_FOR_PATTERN = re.compile(r"\s+for\s+(\d+)\s*(d|days?|w|weeks?)\s*$", re.IGNORECASE)
CHALLENGE_TARGET_PATTERN = re.compile(r"^(\d+)\s+(.+)$")
SCORE_AMOUNT_PATTERN = re.compile(r"^[+-]?\d+$")


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


CHALLENGE_USAGE = "Use: /challenge [target] <title> [for 7d | until YYYY-MM-DD]"


def parse_challenge_text(text: str, today) -> Tuple[Optional[int], str, Optional[str]]:
    """Parse '/challenge [target] <title> [for Nd|Nw | until YYYY-MM-DD]'.

    Returns (target, title, deadline_iso). The deadline is the last day
    scoring is allowed (inclusive).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError(CHALLENGE_USAGE)

    deadline: Optional[str] = None
    match = CHALLENGE_UNTIL_PATTERN.search(text)
    if match:
        try:
            deadline_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("Deadline must be a valid date like 2026-07-20.") from exc
        if deadline_date < today:
            raise ValueError("That deadline is already in the past.")
        deadline = deadline_date.isoformat()
        text = text[: match.start()].strip()
    else:
        match = CHALLENGE_FOR_PATTERN.search(text)
        if match:
            count = int(match.group(1))
            if count <= 0:
                raise ValueError("Duration must be positive.")
            days = count * 7 if match.group(2).lower().startswith("w") else count
            deadline = (today + timedelta(days=days)).isoformat()
            text = text[: match.start()].strip()

    target: Optional[int] = None
    match = CHALLENGE_TARGET_PATTERN.match(text)
    if match:
        target = int(match.group(1))
        text = match.group(2).strip()
    if not text or text.isdigit():
        raise ValueError(CHALLENGE_USAGE)
    return target, text, deadline


def challenge_expired(challenge: dict, today) -> bool:
    deadline = challenge.get("deadline")
    if not deadline:
        return False
    try:
        deadline_date = datetime.strptime(str(deadline), "%Y-%m-%d").date()
    except ValueError:
        return False
    return today > deadline_date


def challenge_streak(challenge: dict, user_id: int, today) -> int:
    """Consecutive days with progress, counting back from today (or yesterday
    if today has no entry yet, so an unbroken streak isn't shown as 0)."""
    daily = challenge.get("daily", {}).get(str(user_id), {})
    days = {day for day, value in daily.items() if int(value) > 0}
    day = today
    if day.isoformat() not in days:
        day = day - timedelta(days=1)
    streak = 0
    while day.isoformat() in days:
        streak += 1
        day -= timedelta(days=1)
    return streak


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


STATUS_ICONS = {
    "pending": "🕓",
    "active": "🟢",
    "revoked": "🗑",
    "cancelled": "🗑",
    "completed": "🏁",
    "ended": "🏁",
}


def display_name(user_id: int, participants: Dict[str, Any], config: BotConfig) -> str:
    user = user_from_id(user_id, config)
    if user:
        return user.name
    name = participants.get(str(user_id))
    if name:
        return str(name)
    return f"user {user_id}"


def progress_bar(value: int, target: int, width: int = 10) -> str:
    if target <= 0:
        return ""
    ratio = min(1.0, max(0.0, value / target))
    filled = int(round(ratio * width))
    return "▰" * filled + "▱" * (width - filled) + f" {int(ratio * 100)}%"


def format_agreement(agreement: dict, config: BotConfig) -> str:
    participants = agreement.get("participants", {}) or {}
    accepted = [int(uid) for uid in agreement.get("accepted_by", [])]
    declined = [int(uid) for uid in agreement.get("declined_by", [])]
    status = agreement.get("status", "pending")
    icon = STATUS_ICONS.get(status, "")
    lines = [f"📜 {agreement.get('id')} — {agreement.get('text')}"]
    lines.append(f"Status: {icon} {status}".rstrip())
    if accepted:
        lines.append("✅ " + ", ".join(display_name(uid, participants, config) for uid in accepted))
    else:
        lines.append("✅ nobody yet")
    if declined:
        lines.append("❌ " + ", ".join(display_name(uid, participants, config) for uid in declined))
    if status == "pending":
        waiting = [
            user.name
            for user in config.users
            if user.id not in accepted and user.id not in declined
        ]
        if waiting:
            lines.append("Waiting on: " + ", ".join(waiting))
    breaches = agreement.get("breaches", [])
    if breaches:
        counts: Dict[int, int] = {}
        for breach in breaches:
            offender = int(breach.get("user_id", 0))
            counts[offender] = counts.get(offender, 0) + 1
        strikes = ", ".join(
            f"{display_name(uid, participants, config)} ×{count}"
            for uid, count in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"⚠️ Breaches: {strikes}")
        last = breaches[-1]
        note = str(last.get("note") or "").strip()
        latest = f"   latest: {display_name(int(last.get('user_id', 0)), participants, config)}"
        if note:
            latest += f" — “{note}”"
        lines.append(latest)
    return "\n".join(lines)


def format_challenge(challenge: dict, config: BotConfig, today=None) -> str:
    today = today or datetime.now(timezone.utc).date()
    participants = challenge.get("participants", {}) or {}
    scores: Dict[int, int] = {}
    for raw_uid, value in (challenge.get("scores", {}) or {}).items():
        try:
            scores[int(raw_uid)] = int(value)
        except (TypeError, ValueError):
            continue
    for user in config.users:
        scores.setdefault(user.id, 0)

    status = challenge.get("status", "active")
    icon = STATUS_ICONS.get(status, "")
    target = challenge.get("target")
    deadline = challenge.get("deadline")
    meta = []
    if target:
        meta.append(f"target {target}")
    if deadline:
        if status == "active":
            try:
                days_left = (datetime.strptime(str(deadline), "%Y-%m-%d").date() - today).days
                meta.append(
                    f"ends {deadline} ({days_left}d left)" if days_left >= 0 else f"ended {deadline}"
                )
            except ValueError:
                meta.append(f"ends {deadline}")
        else:
            meta.append(f"deadline {deadline}")
    header = f"🏆 {challenge.get('id')} — {challenge.get('title')}"
    if meta:
        header += " (" + ", ".join(meta) + ")"

    lines = [header, f"Status: {icon} {status}".rstrip()]
    if target:
        bar = progress_bar(max(scores.values(), default=0), int(target))
        if bar:
            lines.append(bar)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    for rank, (uid, total) in enumerate(ranked, start=1):
        streak = challenge_streak(challenge, uid, today) if status == "active" else 0
        streak_note = f" 🔥{streak}d" if streak >= 2 else ""
        lines.append(f"{rank}. {display_name(uid, participants, config)} — {total}{streak_note}")
    winner_ids = challenge.get("winner_ids") or []
    if status in ("completed", "ended") and winner_ids:
        names = ", ".join(display_name(int(uid), participants, config) for uid in winner_ids)
        lines.append(f"🏆 Winner: {names}")
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


def runtime_for_instance(
    context: ContextTypes.DEFAULT_TYPE, instance_id: str
) -> Optional[Tuple[Ledger, BotConfig]]:
    """Resolve a ledger/config from an explicit instance id (used by inline
    buttons, whose callback data pins the instance they were created for)."""
    store: InstanceStore = context.application.bot_data["store"]
    base_config: BotConfig = context.application.bot_data["base_config"]
    if instance_id not in store.state.get("instances", {}):
        return None
    return store.ledger_for(instance_id), store.config_for(base_config, instance_id)


def user_state_key(config: BotConfig, user_id: int, name: str) -> str:
    return f"{name}:{config.instance_id}:{user_id}"


def user_from_id(user_id: int, config: BotConfig) -> Optional[UserConfig]:
    return next((u for u in config.users if u.id == user_id), None)


def resolve_participant(user: Any, config: BotConfig) -> Optional[UserConfig]:
    """Who is acting in a challenge/agreement flow.

    The default DM instance keeps its fixed roster. Group instances are open:
    anyone present in the chat can agree, accept, and score without touching
    the expense-splitting member list (/join stays opt-in for money).
    """
    member = user_from_id(user.id, config)
    if member:
        return member
    if config.instance_id == "default":
        return None
    name = getattr(user, "full_name", None) or getattr(user, "username", None) or str(user.id)
    return UserConfig(id=user.id, name=str(name))


def match_participant(token: str, item: dict, config: BotConfig) -> Optional[UserConfig]:
    """Match a typed name (or @name) against known participants of an item."""
    wanted = token.lstrip("@").strip().lower()
    if not wanted:
        return None
    candidates: Dict[int, str] = {user.id: user.name for user in config.users}
    for raw_uid, name in (item.get("participants", {}) or {}).items():
        try:
            candidates[int(raw_uid)] = str(name)
        except (TypeError, ValueError):
            continue
    for uid, name in candidates.items():
        if name.lower() == wanted:
            return UserConfig(id=uid, name=name)
    for uid, name in candidates.items():
        if name.lower().startswith(wanted):
            return UserConfig(id=uid, name=name)
    return None


async def announce_to_instance(
    context: ContextTypes.DEFAULT_TYPE,
    config: BotConfig,
    current_chat_id: Optional[int],
    text: str,
) -> None:
    """Mirror a noteworthy event to the instance's home group chat when the
    triggering action happened somewhere else (e.g. a DM via /use)."""
    store: InstanceStore = context.application.bot_data["store"]
    instance = store.state.get("instances", {}).get(config.instance_id) or {}
    chat_id = instance.get("chat_id")
    if chat_id and chat_id != current_chat_id:
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:
            logger.warning("Failed to announce to instance chat %s: %s", chat_id, exc)

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
            [
                InlineKeyboardButton("🏆 Challenges", callback_data=f"{CB_MENU_PREFIX}challenges"),
                InlineKeyboardButton("🤝 Agreements", callback_data=f"{CB_MENU_PREFIX}agreements"),
            ],
            [InlineKeyboardButton("💪 Push-ups", callback_data=f"{CB_MENU_PREFIX}pushups")],
            [InlineKeyboardButton("🏅 Standings", callback_data=f"{CB_MENU_PREFIX}pushups_standings")],
            [InlineKeyboardButton("ℹ️ Help", callback_data=f"{CB_MENU_PREFIX}help")],
        ]
    )


def agreement_keyboard(instance_id: str, agreement_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Accept", callback_data=f"agr|a|{instance_id}|{agreement_id}"
                ),
                InlineKeyboardButton(
                    "❌ Decline", callback_data=f"agr|d|{instance_id}|{agreement_id}"
                ),
            ]
        ]
    )


def challenge_keyboard(instance_id: str, challenge_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("+1", callback_data=f"chl|s|{instance_id}|{challenge_id}|1"),
                InlineKeyboardButton("+5", callback_data=f"chl|s|{instance_id}|{challenge_id}|5"),
                InlineKeyboardButton("+10", callback_data=f"chl|s|{instance_id}|{challenge_id}|10"),
            ],
            [
                InlineKeyboardButton("📊 Standings", callback_data=f"chl|v|{instance_id}|{challenge_id}"),
                InlineKeyboardButton("🏁 End", callback_data=f"chl|e|{instance_id}|{challenge_id}"),
            ],
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
# Agreement/challenge callbacks use '|' separators because instance ids
# contain ':' (e.g. "chat:-100123"). Shape: agr|<action>|<instance>|<id>
CB_AGREEMENT_PREFIX = "agr|"
CB_CHALLENGE_PREFIX = "chl|"


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
        "Shared group bot is online.\n"
        "Add expenses by sending: `<amount> [CUR] <description>`\n"
        f"If no currency is given, I'll ask (options: {', '.join(available_currencies(config))}).\n"
        "Push-ups: `/pushups <count>` or tap the push-up buttons.\n\n"
        "Money:\n"
        "- /add 23.50 dinner, /balance, /history\n"
        "- /settle [comment]  (records who owed whom, marks it paid, clears expenses)\n"
        "- /join / /leave  (opt in/out of expense splitting in this group)\n\n"
        "Agreements 🤝:\n"
        "- /agree <text>  (propose; active once two people accept)\n"
        "- /accept [id], /decline [id], /revoke <id>\n"
        "- /breach [id] [name] [note]  (record a broken agreement — reply to someone to blame them)\n"
        "- /agreements [all|active|pending]\n\n"
        "Challenges 🏆:\n"
        "- /challenge [target] <title> [for 7d | until YYYY-MM-DD]\n"
        "- /score [id] <amount>  (negative corrects mistakes)\n"
        "- /challenges [all|done], /endchallenge [id]\n\n"
        f"Instance: {config.instance_name} (`{config.instance_id}`)\n"
        f"Expense participants: {names}\n"
        "Expenses are split evenly between joined participants.\n\n"
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


PARTICIPANT_GATE_TEXT = "You're not on the participant list for this instance."


def build_agreement_list(ledger: Ledger, config: BotConfig, which: str) -> str:
    agreements = ledger.state.get("agreements", [])
    if which == "all":
        selected = agreements
    elif which in ("active", "pending"):
        selected = [a for a in agreements if a.get("status") == which]
    else:
        which = "open"
        selected = [a for a in agreements if a.get("status") in ("pending", "active")]
    selected = selected[-10:]
    if not selected:
        return f"No {which} agreements here. Propose one with /agree <text>."
    return f"Agreements ({which}):\n\n" + "\n\n".join(
        format_agreement(a, config) for a in selected
    )


def build_challenge_list(
    ledger: Ledger, config: BotConfig, which: str, today
) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    challenges = ledger.state.get("challenges", [])
    if which == "all":
        selected = challenges
    elif which in ("done", "finished", "completed", "ended"):
        which = "done"
        selected = [c for c in challenges if c.get("status") in ("completed", "ended")]
    else:
        which = "active"
        selected = [c for c in challenges if c.get("status") == "active"]
    selected = selected[-10:]
    if not selected:
        return (
            f"No {which} challenges here. Start one with {CHALLENGE_USAGE[5:]}",
            None,
        )
    text = f"Challenges ({which}):\n\n" + "\n\n".join(
        format_challenge(c, config, today) for c in selected
    )
    active = [c for c in selected if c.get("status") == "active"]
    keyboard = (
        challenge_keyboard(config.instance_id, str(active[-1]["id"])) if active else None
    )
    return text, keyboard


async def agree_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = resolve_participant(user, config)
    if not actor:
        await update.message.reply_text(PARTICIPANT_GATE_TEXT)
        return
    text = " ".join(context.args)
    try:
        agreement = ledger.create_agreement(actor, text)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    await update.message.reply_text(
        f"Agreement proposed by {actor.name}.\n\n"
        f"{format_agreement(agreement, config)}\n\n"
        "Tap a button or use /accept, /decline. It takes effect once two people are in.",
        reply_markup=agreement_keyboard(config.instance_id, str(agreement["id"])),
    )


async def respond_agreement_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE, accept: bool
) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = resolve_participant(user, config)
    if not actor:
        await update.message.reply_text(PARTICIPANT_GATE_TEXT)
        return
    agreement_id = context.args[0] if context.args else None
    agreement = (
        ledger.find_agreement(agreement_id) if agreement_id else ledger.latest_open_agreement()
    )
    if not agreement:
        await update.message.reply_text("No matching agreement found. See /agreements.")
        return
    was_active = agreement.get("status") == "active"
    try:
        agreement = ledger.respond_agreement(str(agreement["id"]), actor, accept=accept)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    text = format_agreement(agreement, config)
    if not was_active and agreement.get("status") == "active":
        text = "🤝 Agreement is now ACTIVE.\n\n" + text
        await announce_to_instance(
            context,
            config,
            update.effective_chat.id if update.effective_chat else None,
            text,
        )
    keyboard = (
        agreement_keyboard(config.instance_id, str(agreement["id"]))
        if agreement.get("status") in ("pending", "active")
        else None
    )
    await update.message.reply_text(text, reply_markup=keyboard)


async def accept_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await respond_agreement_command(update, context, accept=True)


async def decline_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await respond_agreement_command(update, context, accept=False)


async def revoke_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = resolve_participant(user, config)
    if not actor:
        await update.message.reply_text(PARTICIPANT_GATE_TEXT)
        return
    if not context.args:
        await update.message.reply_text("Use: /revoke <agreement_id>")
        return
    try:
        agreement = ledger.revoke_agreement(context.args[0], actor)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    text = f"Agreement revoked by {actor.name}.\n\n{format_agreement(agreement, config)}"
    await update.message.reply_text(text)
    await announce_to_instance(
        context, config, update.effective_chat.id if update.effective_chat else None, text
    )


async def breach_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = resolve_participant(user, config)
    if not actor:
        await update.message.reply_text(PARTICIPANT_GATE_TEXT)
        return
    args = list(context.args or [])
    agreement = None
    if args:
        agreement = ledger.find_agreement(args[0])
        if agreement:
            args.pop(0)
    if not agreement:
        agreement = ledger.latest_active_agreement()
    if not agreement:
        await update.message.reply_text(
            "No active agreement found. Use: /breach [agreement_id] [name] [note]"
        )
        return

    offender: Optional[UserConfig] = None
    reply = update.message.reply_to_message
    if reply and reply.from_user:
        offender = UserConfig(
            id=reply.from_user.id,
            name=reply.from_user.full_name or str(reply.from_user.id),
        )
    elif args:
        offender = match_participant(args[0], agreement, config)
        if offender:
            args.pop(0)
    if offender is None:
        offender = actor  # owning up to it yourself

    note = " ".join(args).strip()
    try:
        agreement = ledger.record_breach(str(agreement["id"]), offender, actor, note)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    strikes = sum(
        1 for b in agreement.get("breaches", []) if int(b.get("user_id", 0)) == offender.id
    )
    text = (
        f"⚠️ Breach recorded against {offender.name} (strike {strikes}).\n\n"
        f"{format_agreement(agreement, config)}"
    )
    await update.message.reply_text(text)
    await announce_to_instance(
        context, config, update.effective_chat.id if update.effective_chat else None, text
    )


async def agreements_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    which = context.args[0].lower() if context.args else "open"
    await update.message.reply_text(build_agreement_list(ledger, config, which))


async def challenge_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = resolve_participant(user, config)
    if not actor:
        await update.message.reply_text(PARTICIPANT_GATE_TEXT)
        return
    today = datetime.now(timezone.utc).date()
    try:
        target, title, deadline = parse_challenge_text(" ".join(context.args), today)
        challenge = ledger.create_challenge(actor, title, target, deadline)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    await update.message.reply_text(
        f"Challenge on! 🏁\n\n{format_challenge(challenge, config, today)}\n\n"
        f"Log progress with the buttons or /score {challenge['id']} <amount>.",
        reply_markup=challenge_keyboard(config.instance_id, str(challenge["id"])),
    )


async def score_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = resolve_participant(user, config)
    if not actor:
        await update.message.reply_text(PARTICIPANT_GATE_TEXT)
        return
    args = list(context.args or [])
    challenge_id: Optional[str] = None
    if len(args) == 1 and SCORE_AMOUNT_PATTERN.match(args[0]):
        raw_amount = args[0]
    elif len(args) >= 2:
        challenge_id, raw_amount = args[0], args[1]
    else:
        await update.message.reply_text(
            "Use: /score [challenge_id] <amount> — e.g. /score 20, or /score c2 -5 to correct."
        )
        return
    if challenge_id is None:
        challenge = ledger.latest_active_challenge()
        if not challenge:
            await update.message.reply_text("No active challenge. Start one with /challenge.")
            return
        challenge_id = str(challenge["id"])
    if not SCORE_AMOUNT_PATTERN.match(raw_amount):
        await update.message.reply_text(f"'{raw_amount}' is not a whole number.")
        return
    try:
        challenge, completed = ledger.add_challenge_score(challenge_id, actor, int(raw_amount))
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    today = datetime.now(timezone.utc).date()
    text = format_challenge(challenge, config, today)
    if completed:
        text = f"🎉 {actor.name} hit the target — challenge complete!\n\n" + text
        await update.message.reply_text(text)
        await announce_to_instance(
            context, config, update.effective_chat.id if update.effective_chat else None, text
        )
    else:
        await update.message.reply_text(
            text, reply_markup=challenge_keyboard(config.instance_id, str(challenge["id"]))
        )


async def end_challenge_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    user = update.effective_user
    if not user or not update.message:
        return
    actor = resolve_participant(user, config)
    if not actor:
        await update.message.reply_text(PARTICIPANT_GATE_TEXT)
        return
    challenge = (
        ledger.find_challenge(context.args[0])
        if context.args
        else ledger.latest_active_challenge()
    )
    if not challenge:
        await update.message.reply_text("No active challenge found.")
        return
    try:
        challenge = ledger.finish_challenge(str(challenge["id"]), actor)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    today = datetime.now(timezone.utc).date()
    text = "🏁 Challenge ended.\n\n" + format_challenge(challenge, config, today)
    await update.message.reply_text(text)
    await announce_to_instance(
        context, config, update.effective_chat.id if update.effective_chat else None, text
    )


async def challenges_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime_for_update(update, context)
    today = datetime.now(timezone.utc).date()
    ledger.expire_due_challenges(today)
    which = context.args[0].lower() if context.args else "active"
    text, keyboard = build_challenge_list(ledger, config, which, today)
    await update.message.reply_text(text, reply_markup=keyboard)


async def agreement_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    parts = (query.data or "").split("|")
    if len(parts) != 4:
        await query.answer()
        return
    _, action, instance_id, agreement_id = parts
    runtime = runtime_for_instance(context, instance_id)
    if not runtime:
        await query.answer("This instance no longer exists.", show_alert=True)
        return
    ledger, config = runtime
    actor = resolve_participant(query.from_user, config)
    if not actor:
        await query.answer(PARTICIPANT_GATE_TEXT, show_alert=True)
        return
    agreement = ledger.find_agreement(agreement_id)
    if not agreement:
        await query.answer("Agreement not found.", show_alert=True)
        return
    was_active = agreement.get("status") == "active"
    try:
        agreement = ledger.respond_agreement(agreement_id, actor, accept=(action == "a"))
    except ValueError as exc:
        await query.answer(str(exc), show_alert=True)
        return
    await query.answer("Accepted ✅" if action == "a" else "Declined ❌")
    text = format_agreement(agreement, config)
    if not was_active and agreement.get("status") == "active":
        text = "🤝 Agreement is now ACTIVE.\n\n" + text
        await announce_to_instance(
            context, config, query.message.chat.id if query.message else None, text
        )
    keyboard = (
        agreement_keyboard(instance_id, agreement_id)
        if agreement.get("status") in ("pending", "active")
        else None
    )
    try:
        await query.edit_message_text(text, reply_markup=keyboard)
    except Exception:
        pass  # message unchanged (double tap) or no longer editable


async def challenge_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    parts = (query.data or "").split("|")
    if len(parts) < 4:
        await query.answer()
        return
    action, instance_id, challenge_id = parts[1], parts[2], parts[3]
    runtime = runtime_for_instance(context, instance_id)
    if not runtime:
        await query.answer("This instance no longer exists.", show_alert=True)
        return
    ledger, config = runtime
    today = datetime.now(timezone.utc).date()

    if action == "v":
        challenge = ledger.find_challenge(challenge_id)
        if not challenge:
            await query.answer("Challenge not found.", show_alert=True)
            return
        await query.answer()
        if query.message:
            keyboard = (
                challenge_keyboard(instance_id, challenge_id)
                if challenge.get("status") == "active"
                else None
            )
            await query.message.reply_text(
                format_challenge(challenge, config, today), reply_markup=keyboard
            )
        return

    actor = resolve_participant(query.from_user, config)
    if not actor:
        await query.answer(PARTICIPANT_GATE_TEXT, show_alert=True)
        return

    if action == "e":
        try:
            challenge = ledger.finish_challenge(challenge_id, actor)
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        await query.answer("Challenge ended.")
        text = "🏁 Challenge ended.\n\n" + format_challenge(challenge, config, today)
        try:
            await query.edit_message_text(text)
        except Exception:
            pass
        await announce_to_instance(
            context, config, query.message.chat.id if query.message else None, text
        )
        return

    if action == "s" and len(parts) == 5:
        try:
            amount = int(parts[4])
        except ValueError:
            await query.answer()
            return
        try:
            challenge, completed = ledger.add_challenge_score(challenge_id, actor, amount)
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        await query.answer(f"+{amount} for {actor.name}")
        text = format_challenge(challenge, config, today)
        keyboard = None
        if completed:
            text = f"🎉 {actor.name} hit the target — challenge complete!\n\n" + text
            await announce_to_instance(
                context, config, query.message.chat.id if query.message else None, text
            )
        else:
            keyboard = challenge_keyboard(instance_id, challenge_id)
        try:
            await query.edit_message_text(text, reply_markup=keyboard)
        except Exception:
            pass


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

    action = data[len(CB_MENU_PREFIX) :]
    # Challenge/agreement views are open to everyone in the instance's chat;
    # the expense-roster gate below only applies to money actions.
    if action == "challenges":
        if query.message:
            today = datetime.now(timezone.utc).date()
            ledger.expire_due_challenges(today)
            text, keyboard = build_challenge_list(ledger, config, "active", today)
            await query.message.reply_text(text, reply_markup=keyboard)
        return
    if action == "agreements":
        if query.message:
            await query.message.reply_text(build_agreement_list(ledger, config, "open"))
        return

    actor = user_from_id(query.from_user.id, config)
    if not actor:
        if query.message:
            await query.message.reply_text("You're not on the traveler list for this bot.")
        return

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


async def challenge_deadline_sweep(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Close challenges whose deadline passed and announce final standings."""
    store: InstanceStore = context.application.bot_data["store"]
    base_config: BotConfig = context.application.bot_data["base_config"]
    today = datetime.now(timezone.utc).date()
    for instance_id in store.all_instance_ids():
        ledger = store.ledger_for(instance_id)
        config = store.config_for(base_config, instance_id)
        ended = ledger.expire_due_challenges(today)
        if not ended:
            continue
        instance = store.state.get("instances", {}).get(instance_id) or {}
        chat_id = instance.get("chat_id")
        recipients = [chat_id] if chat_id else [u.id for u in config.users]
        for challenge in ended:
            text = "⏰ Deadline reached.\n\n" + format_challenge(challenge, config, today)
            for recipient in recipients:
                try:
                    await context.bot.send_message(chat_id=recipient, text=text)
                except Exception as exc:
                    logger.warning(
                        "Failed to send challenge deadline report to %s: %s", recipient, exc
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

    if application.job_queue is None:
        logger.warning(
            "JobQueue unavailable (install python-telegram-bot[job-queue]); "
            "daily reports and deadline sweeps are disabled."
        )
    else:
        application.job_queue.run_daily(
            pushups_daily_report,
            time=time(hour=0, minute=0, tzinfo=timezone.utc),
            name="pushups-daily-report",
        )
        application.job_queue.run_daily(
            challenge_deadline_sweep,
            time=time(hour=0, minute=5, tzinfo=timezone.utc),
            name="challenge-deadline-sweep",
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
    application.add_handler(CommandHandler("decline", decline_handler))
    application.add_handler(CommandHandler("revoke", revoke_handler))
    application.add_handler(CommandHandler("breach", breach_handler))
    application.add_handler(CommandHandler("agreements", agreements_handler))
    application.add_handler(CommandHandler("challenge", challenge_handler))
    application.add_handler(CommandHandler("score", score_handler))
    application.add_handler(CommandHandler("challenges", challenges_handler))
    application.add_handler(CommandHandler("endchallenge", end_challenge_handler))
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
    application.add_handler(CallbackQueryHandler(agreement_callback, pattern=r"^agr\|"))
    application.add_handler(CallbackQueryHandler(challenge_callback, pattern=r"^chl\|"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, add_text_handler))
    return application


def main() -> None:
    config = load_config()
    store = InstanceStore(config.data_path, config.users)
    application = build_application(config, store)
    application.run_polling(stop_signals=None)


if __name__ == "__main__":
    main()
