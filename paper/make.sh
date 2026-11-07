#!/bin/bash
# paper.md -> paper_pandoc.md (YAML header, LaTeX-safe glyphs, table widths) is kept by hand;
# this renders paper_pandoc.md -> paper.pdf.  brew install pandoc tectonic
set -e
cd "$(dirname "$0")"
pandoc paper_pandoc.md -o paper.tex --standalone --from markdown+pipe_tables+raw_tex \
  -V documentclass=article --columns=72 --shift-heading-level-by=-1
tectonic paper.tex > tectonic.log 2>&1
echo "paper.pdf: $(mdls -name kMDItemNumberOfPages -raw paper.pdf) pages"
