#!/usr/bin/env bash
# Publish the project site (site/ plus the README figures) to the gh-pages branch,
# which GitHub Pages serves at https://mattbusel.github.io/srfm-lab/.
#
#   python scripts/make_figures.py   # refresh figures and site/data from the code
#   bash site/publish.sh             # copy into gh-pages, commit, push
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cd "$root"
tmp="$(mktemp -d)"
trap 'git worktree remove --force "$tmp" >/dev/null 2>&1 || true' EXIT
git fetch -q origin gh-pages
git worktree add -q "$tmp" origin/gh-pages
mkdir -p "$tmp/assets" "$tmp/data"
cp site/index.html "$tmp/index.html"
cp site/data/* "$tmp/data/"
cp assets/*.svg "$tmp/assets/"
cp .github/social-preview.png "$tmp/assets/social-preview.png"
touch "$tmp/.nojekyll"
cd "$tmp"
git add -A
if git diff --cached --quiet; then echo "gh-pages already up to date"; exit 0; fi
git commit -q -m "Publish site from $(git -C "$root" rev-parse --short HEAD)"
git push -q origin HEAD:gh-pages
echo "published to gh-pages"
