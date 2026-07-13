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
- `/join` and `/leave` control **expense splitting** membership only. Agreements and challenges are open to everyone in a group chat — no subscription needed.
- `/instance` shows the active instance and members.
- `/instances` lists the instances you belong to; in a DM, `/use <instance_id>` switches your active instance. Inline buttons always act on the instance they were posted for, so accepting an agreement from a DM works even with another instance active.
- Actions taken from a DM (agreement activated, challenge won/ended) are announced in the instance's group chat.

## Expense commands
- `/add 23.50 dinner` — log an expense paid by you, split evenly (if you omit currency, the bot asks with an inline picker).
- `/add 42.5 EUR taxi` — log with explicit currency; converted to base using the configured rate.
- `/balance` — show who owes whom.
- `/history` — show the last 10 entries (expenses + settlements).
- `/settle [comment]` — for two-member instances, record who owed whom at current balances (base currency), mark it paid, and reset expenses to zero.
- `/pushups <count>` — built-in daily push-up challenge log.
- Plain text like `12.40 coffee` also logs an expense (DMs only).

## Agreements 🤝
An agreement binds the people who accept it, and takes effect once **two** people are in.
- `/agree <text>` — propose an agreement. The proposal message carries ✅ Accept / ❌ Decline buttons; the proposer counts as accepted.
- `/accept [id]` / `/decline [id]` — respond to the latest open agreement, or a specific one. Accepting an active agreement joins it; declining one you accepted leaves it (below two acceptors it goes back to pending).
- `/revoke <id>` — creator withdraws the agreement for everyone.
- `/breach [id] [name] [note]` — record a broken agreement. Reply to someone's message to blame them, name them (`/breach a2 Bob skipped again`), or omit the name to own up yourself. Strike counts show on the agreement.
- `/agreements [all|active|pending|done]` — list agreements (default: open ones).

### Daily check-ins (tracked agreements)
Add a schedule and/or period to `/agree` and the bot keeps score of compliance:
```
/agree meet 8.30 weekdays and 10 weekends every day for 1y
/agree gym before work weekdays
/agree no sugar for 30d          # a period alone implies daily
/agree meditate daily until 2026-12-31
```
- Schedules: `daily` / `every day`, `weekdays`, `weekends`. Periods: `for Nd/Nw/Nm/Ny` or `until YYYY-MM-DD` (m = 30 days, y = 365 days).
- On every scheduled day at 18:00 UTC the bot asks "Did you hold it today?" with **✅ Made it / ❌ Missed** buttons (sent to the group chat, or to both DMs for the default instance). Answers can be changed; `/checkin [id] yes|no` works too, and `/checkin` alone shows today's card.
- Tracking per person: ✅/❌ totals and 🔥 streaks (streaks skip non-scheduled days, so a weekday agreement survives the weekend).
- Checking in on an agreement you haven't accepted counts as accepting it.
- When the period ends, the agreement completes with a final report (also swept daily). `/breach` stays available for disputes beyond the daily question.

## Challenges 🏆
- `/challenge [target] <title> [for 7d | for 2w | for 1y | until YYYY-MM-DD]` — start a challenge, e.g. `/challenge 100 push-ups for 7d`.
- Every challenge message has `+1 / +5 / +10` buttons that update the leaderboard in place, plus 📊 Standings and 🏁 End (creator only).
- `/score [id] <amount>` — log progress; negative amounts correct mistakes (totals never go below zero).
- First to reach the target wins on the spot 🎉. Deadline challenges close automatically (checked daily at 00:05 UTC and on access) and the top scorer takes it.
- Leaderboards show progress bars toward the target and 🔥 day-streaks.
- `/challenges [all|done]` — list challenges (default: active).
- `/endchallenge [id]` — creator closes a challenge early; top scorer wins.

## Bündnerdeutsch 🏔
- `/dialekt` (or `/dialect`) toggles the bot's Bündnerdeutsch character per instance; `/dialekt on|off` sets it explicitly. The flag is persisted, so each group keeps its language.
- Covers menus, buttons, expense/settle/balance replies, agreements, challenges, check-ins, push-up standings, and the daily prompts ("Und, häsch es hüt gschafft?"). Command names and rare internal error messages stay English.

## Tests
```bash
python3 -m unittest discover -s tests -t .
```

## Deployment
- Pushes to `main` deploy automatically through GitHub Actions (`.github/workflows/deploy.yml`).
- The workflow uses these repository secrets: `NASNAS_DEPLOY_HOST`, `NASNAS_DEPLOY_USER`, and `NASNAS_DEPLOY_KEY`.
- Manual fallback from a trusted machine:
  ```bash
  ./deploy.sh deploy
  ```
- Server-only files are preserved during deploy: `.env` and `data/`.

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
- Expense commands are limited to configured/joined members; agreements and challenges are open to anyone in a group chat.
- Amounts are always split evenly between joined members; adjust code if you need custom splits.
- The bot keeps an append-only JSON ledger; back it up if you care about history.
