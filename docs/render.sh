#!/usr/bin/env bash
# Render the lessons book into site/learn/. The book's first chapter must be
# index.md; CURRICULUM.md stays the source (the docs check asserts its counts)
# and is copied into place for the render. Cells are never executed here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
cp CURRICULUM.md index.md
trap 'rm -f index.md' EXIT
quarto render . "$@"
# Quarto copies the directory's other markdown files as resources; they are
# the repository's reference docs, not part of the book.
rm -f ../site/learn/*.md
