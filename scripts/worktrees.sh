#!/usr/bin/env bash
# Sets up one git worktree + isolated venv per collaborator branch, all sharing
# this same account/server. Worktrees live under worktrees/<branch>/ inside
# this same repo (gitignored, so `git status`/rg in the main checkout never
# see into them) rather than as sibling directories -- keeps everything under
# one dialogue_benchmarks/ instead of scattering sibling folders next to it.
#
# Two isolation layers, both needed:
# - git worktree: separate checkout/working directory per branch, so two
#   Claude Code sessions under the same login never touch the same
#   file at once.
# - its own .venv: `pip install -e .` registers an editable install in
#   whatever Python environment runs it. This machine's `python`/`pip` is the
#   shared base conda env, not a per-project venv -- if two worktrees both ran
#   `pip install -e .` there, whichever ran last would silently repoint `db`
#   and `import dialogue_benchmarks` for *everyone* sharing that env,
#   including the other two people. A `.venv` per worktree makes each
#   worktree's editable install affect only that worktree.
#
# Usage:
#   scripts/worktrees.sh create data-metrics    # -> worktrees/data-metrics (A)
#   scripts/worktrees.sh create baselines       # -> worktrees/baselines (B)
#   scripts/worktrees.sh list
#
# Always invoke the CLI as `.venv/bin/db` (not bare `db`) from inside a
# worktree -- shell state (like `source .venv/bin/activate`) does not persist
# between separate tool-call invocations in a Claude Code session, so
# activation-based workflows silently fall back to whatever `db` is on PATH.
#
# Data is never duplicated -- each worktree gets its own `data` symlink to the
# same shared HDD path (matching the main checkout's). DIALOGUE_BENCHMARKS_DATA
# is also set in the committed .claude/settings.json as a second, independent
# way to find it (used automatically inside a Claude Code session even without
# the symlink) -- but don't rely on that alone: it's only injected by Claude
# Code itself, not by a plain shell running `db`/pytest directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shared data dir; set DIALOGUE_BENCHMARKS_DATA to the lab's shared HDD path
DATA_ROOT="${DIALOGUE_BENCHMARKS_DATA:-$(readlink -f "$REPO_ROOT/data")}"

cmd="${1:-}"
branch="${2:-}"

case "$cmd" in
  create)
    [ -n "$branch" ] || { echo "usage: $0 create <branch-name>" >&2; exit 1; }
    dest="$REPO_ROOT/worktrees/$branch"
    if git -C "$REPO_ROOT" show-ref --quiet "refs/heads/$branch"; then
      git -C "$REPO_ROOT" worktree add "$dest" "$branch"
    else
      git -C "$REPO_ROOT" worktree add -b "$branch" "$dest"
    fi
    ln -s "$DATA_ROOT" "$dest/data"
    python3 -m venv "$dest/.venv"
    "$dest/.venv/bin/pip" install -q --upgrade pip
    "$dest/.venv/bin/pip" install -q -e "${dest}[dev]"
    echo "worktree ready at $dest (branch: $branch), venv installed"
    echo "next: cd $dest && .venv/bin/db list"
    ;;
  list)
    git -C "$REPO_ROOT" worktree list
    ;;
  *)
    echo "usage: $0 {create <branch-name>|list}" >&2
    exit 1
    ;;
esac
