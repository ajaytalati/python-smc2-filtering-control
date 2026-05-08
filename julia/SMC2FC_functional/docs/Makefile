# Build the SMC2FC_functional technical report as a PDF.
#
# Requires: pdflatex (TeX Live or MiKTeX). The .tex sources use only
# the listings, booktabs, longtable, hyperref, microtype, parskip,
# enumitem, and amsmath/amssymb packages -- all standard.
#
# Targets:
#   make           build the PDF (default)
#   make clean     remove intermediates
#   make distclean remove intermediates AND the PDF

DOC      = SMC2FC_functional_report
LATEXMK  = $(shell command -v latexmk 2>/dev/null)
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
ifeq ($(LATEXMK),)
	$(PDFLATEX) $(DOC).tex
	$(PDFLATEX) $(DOC).tex
else
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(DOC).tex
endif

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
	      $(DOC).fls $(DOC).fdb_latexmk $(DOC).synctex.gz

distclean: clean
	rm -f $(DOC).pdf
