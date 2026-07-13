#!/usr/bin/env bash
# =============================================================================
# nasnas / travel-bot -- server reference for agents and humans
# =============================================================================
#
# SERVER
#   Host    : 82.165.45.100
#   User    : root, SSH key only
#   Bot dir : /usr/bots/travel-bot
#   Service : travel-bot
#   Venv    : /usr/bots/travel-bot/.venv
#
# Files NOT in git and preserved on the server:
#   .env                 -- bot token, Telegram IDs, currency rates
#   data/                -- live bot database and backups
#
# Deploy method:
#   ./deploy.sh deploy
#
# =============================================================================

set -euo pipefail

SERVER="root@82.165.45.100"
BOT_DIR="/usr/bots/travel-bot"
SERVICE="travel-bot"

redact_bot_tokens() {
  sed -E 's#bot[0-9]+:[A-Za-z0-9_-]+#bot<redacted>#g'
}

case "${1:-help}" in
  deploy)
    rsync -az --delete \
      --exclude ".git/" \
      --exclude ".DS_Store" \
      --exclude "__pycache__/" \
      --exclude ".venv/" \
      --exclude ".env" \
      --exclude "data/" \
      --exclude "deploy/*credentials*.txt" \
      --exclude "deploy/ssh_*.sh" \
      --exclude "travel-bot.service" \
      ./ "$SERVER:$BOT_DIR/"
    ssh "$SERVER" "
      set -euo pipefail
      cd '$BOT_DIR'
      python3 -m venv .venv
      .venv/bin/pip install -r requirements.txt
      systemctl restart '$SERVICE'
      systemctl is-active '$SERVICE'
    "
    ;;

  status)
    ssh "$SERVER" "
      systemctl is-active '$SERVICE'
      systemctl show '$SERVICE' --property=MainPID,ExecMainStatus,FragmentPath --no-pager
    "
    ;;

  logs)
    ssh "$SERVER" "journalctl -u '$SERVICE' -n 80 --no-pager" | redact_bot_tokens
    ;;

  follow)
    ssh "$SERVER" "journalctl -u '$SERVICE' -f" | redact_bot_tokens
    ;;

  restart)
    ssh "$SERVER" "systemctl restart '$SERVICE' && systemctl is-active '$SERVICE'"
    ;;

  data)
    scp "$SERVER:$BOT_DIR/data/expenses.json" /tmp/nasnas-expenses-live.json
    echo "Saved to /tmp/nasnas-expenses-live.json"
    ;;

  push-data)
    local_file="${2:?usage: deploy.sh push-data <expenses.json>}"
    scp "$local_file" "$SERVER:$BOT_DIR/data/expenses.json"
    ssh "$SERVER" "systemctl restart '$SERVICE' && systemctl is-active '$SERVICE'"
    ;;

  help|*)
    grep "^#" "$0" | sed 's/^# \{0,1\}//'
    echo ""
    echo "Usage: $0 {deploy|status|logs|follow|restart|data|push-data <expenses.json>}"
    ;;
esac
