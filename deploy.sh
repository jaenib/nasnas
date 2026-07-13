#!/usr/bin/env bash
# =============================================================================
# nasnas / travel-bot -- server reference for agents and humans
# =============================================================================
#
# SERVER
#   Host    : 82.165.45.100
#   User    : deploy for automation, root for emergency admin
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

SERVER="${NASNAS_SERVER:-root@82.165.45.100}"
BOT_DIR="${NASNAS_BOT_DIR:-/usr/bots/travel-bot}"
SERVICE="${NASNAS_SERVICE:-travel-bot}"
REMOTE_SUDO="${NASNAS_REMOTE_SUDO:-}"
SSH_ARGS=()
if [[ -n "${NASNAS_SSH_KEY:-}" ]]; then
  SSH_ARGS+=("-i" "$NASNAS_SSH_KEY" "-o" "IdentitiesOnly=yes")
fi
SSH_ARGS+=("-o" "StrictHostKeyChecking=yes")

ssh_run() {
  ssh "${SSH_ARGS[@]}" "$@"
}

rsync_ssh() {
  local args=()
  local arg
  for arg in "${SSH_ARGS[@]}"; do
    args+=("$arg")
  done
  printf 'ssh'
  for arg in "${args[@]}"; do
    printf ' %q' "$arg"
  done
}

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
      -e "$(rsync_ssh)" \
      ./ "$SERVER:$BOT_DIR/"
    ssh_run "$SERVER" "
      set -euo pipefail
      cd '$BOT_DIR'
      python3 -m venv .venv
      .venv/bin/pip install -r requirements.txt
      $REMOTE_SUDO systemctl restart '$SERVICE'
      $REMOTE_SUDO systemctl is-active '$SERVICE'
    "
    ;;

  status)
    ssh_run "$SERVER" "
      $REMOTE_SUDO systemctl is-active '$SERVICE'
      $REMOTE_SUDO systemctl show '$SERVICE' --property=MainPID,ExecMainStatus,FragmentPath --no-pager
    "
    ;;

  logs)
    ssh_run "$SERVER" "$REMOTE_SUDO journalctl -u '$SERVICE' -n 80 --no-pager" | redact_bot_tokens
    ;;

  follow)
    ssh_run "$SERVER" "$REMOTE_SUDO journalctl -u '$SERVICE' -f" | redact_bot_tokens
    ;;

  restart)
    ssh_run "$SERVER" "$REMOTE_SUDO systemctl restart '$SERVICE' && $REMOTE_SUDO systemctl is-active '$SERVICE'"
    ;;

  data)
    scp "${SSH_ARGS[@]}" "$SERVER:$BOT_DIR/data/expenses.json" /tmp/nasnas-expenses-live.json
    echo "Saved to /tmp/nasnas-expenses-live.json"
    ;;

  push-data)
    local_file="${2:?usage: deploy.sh push-data <expenses.json>}"
    scp "${SSH_ARGS[@]}" "$local_file" "$SERVER:$BOT_DIR/data/expenses.json"
    ssh_run "$SERVER" "$REMOTE_SUDO systemctl restart '$SERVICE' && $REMOTE_SUDO systemctl is-active '$SERVICE'"
    ;;

  help|*)
    grep "^#" "$0" | sed 's/^# \{0,1\}//'
    echo ""
    echo "Usage: $0 {deploy|status|logs|follow|restart|data|push-data <expenses.json>}"
    ;;
esac
