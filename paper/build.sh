#!/usr/bin/env bash
# Build marketrank.pdf: pdflatex, bibtex, pdflatex, pdflatex.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error marketrank.tex
bibtex marketrank
pdflatex -interaction=nonstopmode -halt-on-error marketrank.tex
pdflatex -interaction=nonstopmode -halt-on-error marketrank.tex
