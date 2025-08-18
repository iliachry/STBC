# Makefile for LaTeX document compilation

# Main document name (without extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
LATEXFLAGS = -interaction=nonstopmode

# Default target
all: $(MAIN).pdf

# Compile the PDF (run twice for references)
$(MAIN).pdf: $(MAIN).tex sections/*.tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Quick compile (single pass)
quick:
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).bbl $(MAIN).blg

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# View the PDF (Windows)
view: $(MAIN).pdf
	start $(MAIN).pdf

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile the PDF (default)"
	@echo "  quick    - Single pass compilation"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove all generated files including PDF"
	@echo "  view     - Open the PDF (Windows)"
	@echo "  help     - Show this help message"

.PHONY: all quick clean cleanall view help
