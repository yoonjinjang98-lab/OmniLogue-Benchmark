#!/usr/bin/env bash
# Cross-worktree sync check. Each worktree (main/baselines/data-metrics) is a
# self-contained plugin universe: a Claude Code session working in one has no
# automatic visibility into what's landed in the others -- `db list`/`db
# validate` only ever see what's checked out locally, and a new metric or
# method sitting in `main` produces no error, no warning, just silence, in a
# sibling worktree that hasn't merged it in yet. Run this at the start of a
# session (or before wrapping one up) as a substitute for that missing
# visibility.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== commits: main vs HEAD =="
behind=$(git rev-list --count HEAD..main)
ahead=$(git rev-list --count main..HEAD)
echo "behind main by $behind commit(s), ahead by $ahead"
if [ "$behind" -gt 0 ]; then
  echo "  --> git merge main --no-edit   (then: .venv/bin/python -m pytest -q)"
fi

echo
echo "== plugins that exist in main but not here =="
found=0
for kind in datasets methods metrics; do
  while IFS= read -r name; do
    [ -z "$name" ] && continue
    if [ ! -d "plugins/$kind/$name" ]; then
      echo "  plugins/$kind/$name"
      found=1
    fi
  done < <(git ls-tree -d --name-only "main:plugins/$kind" 2>/dev/null)
done
[ "$found" -eq 0 ] && echo "  (none)"

echo
echo "== pyproject.toml: differs from main? =="
if ! git diff --quiet main -- pyproject.toml 2>/dev/null; then
  echo "  yes -- diff:"
  git diff main -- pyproject.toml | grep '^[+-]' | grep -v '^[+-][+-][+-]'
  if [ "$behind" -gt 0 ]; then
    echo "  --> this may include new optional-dependencies extras from main --"
    echo "      after merging, re-run: .venv/bin/pip install -e '.[dev,<any-new-extra>]'"
  fi
  if [ "$ahead" -gt 0 ] && [ "$behind" -eq 0 ]; then
    echo "  --> this worktree is ahead of main, so these are extras/deps added HERE that"
    echo "      main (and worktrees still behind main) don't have yet -- not something to"
    echo "      install locally; flag it to the lead so it gets propagated up to main."
  fi
else
  echo "  (no diff)"
fi
