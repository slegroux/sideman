#!/usr/bin/env bash
# Render the public learn pages into site/learn/. The book's first chapter must
# be index.md, so learn.md is copied into place; the logo is bundled from
# assets/. Cells are never executed here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
cp learn.md index.md
mkdir -p img && cp ../assets/logo-light.svg img/
trap 'rm -f index.md' EXIT
quarto render . "$@"
# Quarto copies the directory's other markdown files as resources; they are
# the repository's reference docs, not part of the book.
rm -f ../site/learn/*.md
