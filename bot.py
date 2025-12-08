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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class Ledger:
    def __init__(self, path: Path, users: List[UserConfig]):
        self.path = path
        self.users = {user.id: user for user in users}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, List[dict]] = {"expenses": []}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.state = {"expenses": [], "settlements": []}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            if "expenses" not in self.state:
                self.state["expenses"] = []
            if "settlements" not in self.state:
                self.state["settlements"] = []
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load ledger from %s: %s", self.path, exc)
            self.state = {"expenses": [], "settlements": []}

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        tmp_path.replace(self.path)

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
        amount: float,
        currency: str,
        comment: str,
        amount_base: float,
        base_currency: str,
        balances_before: Dict[int, float],
    ) -> dict:
        settlement = {
            "payer_id": payer_id,
            "payer_name": self.users[payer_id].name,
            "receiver_id": receiver_id,
            "receiver_name": self.users[receiver_id].name,
            "amount": round(amount, 2),
            "currency": currency,
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


# ------------------------------ utilities ------------------------------- #


AMOUNT_PATTERN = re.compile(
    r"^\s*([-+]?\d+(?:[.,]\d{1,2})?)(?:\s+([A-Za-z]{3}))?\s+(.+)$"
)
SETTLEMENT_PATTERN = re.compile(
    r"^\s*([-+]?\d+(?:[.,]\d{1,2})?)(?:\s+([A-Za-z]{3}))?(?:\s+(.*))?$"
)


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


def parse_settlement_text(text: str) -> Tuple[float, Optional[str], str]:
    match = SETTLEMENT_PATTERN.match(text or "")
    if not match:
        raise ValueError("Use: <amount> [CUR] [comment] (e.g. 100 MAD payout)")
    amount_raw, currency_raw, comment_raw = match.groups()
    currency = normalize_currency(currency_raw) or None
    amount = float(amount_raw.replace(",", "."))
    if amount <= 0:
        raise ValueError("Amount must be positive.")
    comment = (comment_raw or "").strip() or "settlement"
    return round(amount, 2), currency, comment


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


def get_runtime(context: ContextTypes.DEFAULT_TYPE) -> Tuple[Ledger, BotConfig]:
    return (
        context.application.bot_data["ledger"],
        context.application.bot_data["config"],
    )


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


CB_CURRENCY_PREFIX = "currency:"


async def currency_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime(context)
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    if not data.startswith(CB_CURRENCY_PREFIX):
        return
    currency = data[len(CB_CURRENCY_PREFIX) :].strip().upper()
    pending = context.user_data.get("pending_currency")
    if not pending:
        await query.edit_message_text("No pending entry to apply this currency to.")
        return
    if pending.get("payer_id") != query.from_user.id:
        await query.edit_message_text("This choice does not belong to you.")
        return

    kind = pending.get("kind")
    amount = pending.get("amount")
    description = pending.get("description")
    comment = pending.get("comment")
    actor = user_from_id(query.from_user.id, config)
    if not actor:
        await query.edit_message_text("You're not on the traveler list for this bot.")
        context.user_data.pop("pending_currency", None)
        return

    if kind == "expense":
        if amount is None or description is None:
            await query.edit_message_text("Incomplete pending expense data.")
            context.user_data.pop("pending_currency", None)
            return
        await finalize_expense(query, context, actor, amount, currency, description)
    elif kind == "settlement":
        receiver_id = pending.get("receiver_id")
        if amount is None or comment is None or receiver_id is None:
            await query.edit_message_text("Incomplete pending settlement data.")
            context.user_data.pop("pending_currency", None)
            return
        receiver = user_from_id(receiver_id, config)
        if not receiver:
            await query.edit_message_text("Receiver not found.")
            context.user_data.pop("pending_currency", None)
            return
        await finalize_settlement(query, context, actor, receiver, amount, currency, comment)
    else:
        await query.edit_message_text("Unknown pending action.")

    context.user_data.pop("pending_currency", None)

# ------------------------------ handlers -------------------------------- #


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _, config = get_runtime(context)
    names = " & ".join([u.name for u in config.users])
    help_text = (
        "Shared travel bot is online.\n"
        "Add expenses by sending: `<amount> [CUR] <description>`\n"
        f"If no currency is given, I'll ask (options: {', '.join(available_currencies(config))}).\n"
        "or `/add <amount> [CUR] <description>`.\n\n"
        "Commands:\n"
        "- /add 23.50 dinner\n"
        "- /balance\n"
        "- /history\n\n"
        "- /settle 100 MAD payout note  (records settlement and clears expenses)\n\n"
        f"Participants: {names}\n"
        "Every entry is split evenly between both people."
    )
    await update.message.reply_text(help_text)


async def add_expense_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime(context)
    user = update.effective_user
    if not user:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("You're not on the traveler list for this bot.")
        return

    text = " ".join(context.args) if context.args else (update.message.text or "")
    try:
        amount, currency, description = parse_expense_text(text)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    if not currency:
        context.user_data["pending_currency"] = {
            "kind": "expense",
            "amount": amount,
            "description": description,
            "payer_id": actor.id,
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
    # Avoid responding to service messages in groups.
    if update.message.chat.type != "private":
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
    ledger, config = get_runtime(context)
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
        await update.message.reply_text(reply_text)
    elif hasattr(update, "edit_message_text"):
        await update.edit_message_text(reply_text)


async def balance_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime(context)
    net = ledger.balances()
    text = format_balance_lines(net, config.users, config.base_currency)
    await update.message.reply_text(text)


async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ledger, config = get_runtime(context)
    entries = ledger.last_entries(limit=10)
    if not entries:
        await update.message.reply_text("No expenses yet.")
        return
    lines = [format_entry_line(exp, config.base_currency) for exp in entries]
    await update.message.reply_text("Recent activity:\n" + "\n".join(lines))


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
    ledger, config = get_runtime(context)
    user = update.effective_user
    if not user:
        return
    actor = user_from_id(user.id, config)
    if not actor:
        await update.message.reply_text("You're not on the traveler list for this bot.")
        return

    # Determine receiver (the other configured user)
    receiver = next((u for u in config.users if u.id != actor.id), None)
    if not receiver:
        await update.message.reply_text("Need exactly two users configured for settlement.")
        return

    text = " ".join(context.args) if context.args else (update.message.text or "")
    try:
        amount, currency, comment = parse_settlement_text(text)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    if not currency:
        context.user_data["pending_currency"] = {
            "kind": "settlement",
            "amount": amount,
            "comment": comment,
            "payer_id": actor.id,
            "receiver_id": receiver.id,
        }
        await update.message.reply_text(
            "Choose a currency for the settlement:",
            reply_markup=build_currency_keyboard(config),
        )
        return

    await finalize_settlement(update, context, actor, receiver, amount, currency, comment)


async def finalize_settlement(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    actor: UserConfig,
    receiver: UserConfig,
    amount: float,
    currency: str,
    comment: str,
) -> None:
    ledger, config = get_runtime(context)
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
        amount=amount,
        currency=currency,
        comment=comment,
        amount_base=amount_base,
        base_currency=config.base_currency,
        balances_before=net_before,
    )
    reply = (
        f"Recorded settlement: {actor.name} paid {receiver.name} {amount:.2f} {currency} "
        f"('{comment}'). Expenses reset. Balances are now 0 for everyone."
    )
    if hasattr(update, "message") and update.message:
        await update.message.reply_text(reply)
    elif hasattr(update, "edit_message_text"):
        await update.edit_message_text(reply)


# ------------------------------ bootstrap -------------------------------- #


async def on_startup(app: Application) -> None:
    config: BotConfig = app.bot_data["config"]
    await healthcheck(app, config)
    logger.info("Bot started.")


def build_application(config: BotConfig, ledger: Ledger) -> Application:
    application = (
        Application.builder()
        .token(config.token)
        .post_init(on_startup)
        .build()
    )
    application.bot_data["config"] = config
    application.bot_data["ledger"] = ledger

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("add", add_expense_handler))
    application.add_handler(CommandHandler("balance", balance_handler))
    application.add_handler(CommandHandler("history", history_handler))
    application.add_handler(CommandHandler("settle", settle_handler))
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
    ledger = Ledger(config.data_path, config.users)
    application = build_application(config, ledger)
    application.run_polling(stop_signals=None)


if __name__ == "__main__":
    main()
