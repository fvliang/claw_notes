#!/bin/bash
# claw_notes daily update script
set -e

cd /root/.openclaw/workspace/claw_notes

# Run daily search
python3 scripts/daily_search.py --verbose 2>&1 | tee /tmp/clawbot_daily.log

# If there are changes, commit and push
if [[ -n $(git status --porcelain) ]]; then
    git add -A
    git commit -m "[claw-bot] daily update $(date +%Y-%m-%d)"
    git push origin main
    echo "Pushed updates"
else
    echo "No changes to push"
fi
