# Nasnas Bot

Lightweight Telegram bot for shared expenses, agreements, and challenges. The original two-person DM flow still works as the default instance, and the same bot can now host separate group instances with their own subscribed members and state.

## Setup
- Copy `.env.example` to `.env` and fill in `BOT_TOKEN`, both user IDs, and display names. Optional: `HEALTHCHECK_CHAT_ID` for a daily "alive" ping.
- Create a venv and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Run locally:
  ```bash
  python bot.py
  ```
- Data lives in `data/expenses.json` (can be changed via `DATA_PATH` in `.env`). Existing single-ledger JSON is migrated into the `default` instance when the bot saves again.
- Multi-currency: set `BASE_CURRENCY` (e.g. `MAD`) and rates like `RATE_EUR_TO_BASE`, `RATE_CHF_TO_BASE` (1 EUR * rate = MAD). If no rate is set, EUR/CHF entries will be rejected.

## Instances
- Private chats use the legacy `default` instance configured by `USER_ONE_ID` and `USER_TWO_ID`.
- Group chats automatically get their own instance keyed by the Telegram chat ID.
- In a group, each participant subscribes with `/join`; `/leave` removes them from that group instance.
- `/instance` shows the active instance and members.
- `/instances` lists the instances you belong to; in a DM, `/use <instance_id>` switches your active instance.

## Commands
- `/add 23.50 dinner` — log an expense paid by you, split evenly (if you omit currency, the bot asks with an inline picker).
- `/add 42.5 EUR taxi` — log with explicit currency; converted to base using the configured rate.
- `/balance` — show who owes whom.
- `/history` — show the last 10 entries (expenses + settlements).
- `/settle [comment]` — for two-member instances, record who owed whom at current balances (base currency), mark it paid, and reset expenses to zero.
- `/agree <text>` — propose an agreement for all subscribed members.
- `/accept [agreement_id]` — accept the latest pending agreement, or a specific one.
- `/agreements` — list recent agreements.
- `/challenge [target] <title>` — create a shared challenge, e.g. `/challenge 100 push-ups`.
- `/score [challenge_id] <amount>` — add progress to the latest active challenge or a specific one.
- `/challenges` — list recent challenge standings.
- `/pushups <count>` — built-in daily push-up challenge log.
- Plain text like `12.40 coffee` also logs an expense.

Note: This repo currently tracks small maintenance tweaks; feel free to remove this line after merging the YOLO test PR.

## Systemd service (server)
1) Copy the repo to your server (e.g. `/usr/bots/travel-bot`) and create a `.env` there.  
2) Create a venv and install deps on the server:
   ```bash
   cd /usr/bots/travel-bot
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3) Example unit file `/etc/systemd/system/travel-bot.service`:
   ```
   [Unit]
   Description=Travel Split Telegram Bot
   After=network.target

   [Service]
   WorkingDirectory=/usr/bots/travel-bot
   EnvironmentFile=/usr/bots/travel-bot/.env
   ExecStart=/usr/bots/travel-bot/.venv/bin/python /usr/bots/travel-bot/bot.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
4) Enable & start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now travel-bot.service
   sudo systemctl status travel-bot.service --no-pager
   ```

## Notes
- Only the two configured user IDs can interact with the bot.
- Amounts are always split 50/50; adjust code if you need custom splits.
- The bot keeps an append-only JSON ledger; back it up if you care about history.
