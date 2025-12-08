# Travel Split Bot

Lightweight Telegram bot that lets two people track shared travel expenses via DM. Every entry is split 50/50 and the bot replies with current standings.

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
- Data lives in `data/expenses.json` (can be changed via `DATA_PATH` in `.env`).
- Multi-currency: set `BASE_CURRENCY` (e.g. `MAD`) and rates like `RATE_EUR_TO_BASE`, `RATE_CHF_TO_BASE` (1 EUR * rate = MAD). If no rate is set, EUR/CHF entries will be rejected.

## Commands (DM the bot)
- `/add 23.50 dinner` — log an expense paid by you, split evenly (if you omit currency, the bot asks with an inline picker).
- `/add 42.5 EUR taxi` — log with explicit currency; converted to base using the configured rate.
- `/balance` — show who owes whom.
- `/history` — show the last 10 entries (expenses + settlements).
- `/settle 100 MAD payout note` — record a settlement payout from the caller to the other user (comment optional) and reset balances to zero. The settlement is stored in history.
- Plain text like `12.40 coffee` also logs an expense.

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
