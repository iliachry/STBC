# Paper: Systematic Coding Gain Optimization for Tunable-Rate 4x4 STBCs

## Structure

The LaTeX paper has been organized into a modular structure for better maintainability:

```
paper/
├── main.tex                    # Main document with preamble and structure
├── sections/                   # Individual section files
│   ├── abstract.tex           # Abstract
│   ├── introduction.tex       # Section 1: Introduction
│   ├── related_work.tex       # Section 2: Related Work
│   ├── system_model.tex       # Section 3: System Model
│   ├── algebraic_preliminaries.tex  # Section 4: Algebraic Preliminaries
│   ├── stbc_construction.tex  # Section 5: Biquaternion STBC Construction Framework
│   ├── coding_gain_optimization.tex # Section 6: Coding Gain Optimization
│   ├── simulation_results.tex # Section 7: Simulation Results and Analysis
│   ├── conclusion.tex         # Section 8: Conclusion
│   └── bibliography.tex       # References
└── README.md                   # This file
```

## Compilation

To compile the paper, run from the `paper/` directory:

```bash
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

Or if using BibTeX (when you have a .bib file):
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Benefits of Modular Structure

1. **Easier Collaboration**: Multiple authors can work on different sections simultaneously
2. **Version Control**: Changes to individual sections are easier to track
3. **Reusability**: Sections can be reused in other documents
4. **Maintainability**: Easier to locate and modify specific content
5. **Compilation Speed**: Can comment out sections during drafting for faster compilation

## Editing Guidelines

- Each section file contains only the section content (starting with `\section{...}`)
- The main.tex file contains all document setup, packages, and theorem definitions
- To temporarily exclude a section, comment out the corresponding `\input{...}` line in main.tex
- Keep figures in a separate `figures/` directory (create if needed)

## Paper Summary

This paper presents a systematic framework for constructing and optimizing high-performance 4x4 Space-Time Block Codes (STBCs) from biquaternion division algebras. Key contributions include:

1. A flexible framework for tunable-rate 4x4 STBCs
2. Systematic optimization of the algebraic parameter γ for improved coding gain
3. Demonstration that optimization benefits depend critically on detection complexity:
   - ML detection: ~0.2 dB gain
   - MMSE detection: 1.0 dB gain  
   - ZF detection: 2.0 dB gain at 10^-5 BER

The work bridges theory and practice by explaining why optimization impact varies with system implementation choices.
