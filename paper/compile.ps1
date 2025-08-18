# PowerShell script for compiling the LaTeX document on Windows

param(
    [Parameter(Position=0)]
    [ValidateSet("all", "quick", "clean", "cleanall", "view", "help")]
    [string]$Target = "all"
)

$MainFile = "main"

function Compile-All {
    Write-Host "Compiling $MainFile.tex (full compilation)..." -ForegroundColor Green
    pdflatex -interaction=nonstopmode "$MainFile.tex"
    pdflatex -interaction=nonstopmode "$MainFile.tex"
    Write-Host "Compilation complete!" -ForegroundColor Green
}

function Compile-Quick {
    Write-Host "Quick compilation of $MainFile.tex..." -ForegroundColor Green
    pdflatex -interaction=nonstopmode "$MainFile.tex"
    Write-Host "Quick compilation complete!" -ForegroundColor Green
}

function Clean-Aux {
    Write-Host "Cleaning auxiliary files..." -ForegroundColor Yellow
    $auxFiles = @("*.aux", "*.log", "*.out", "*.toc", "*.lof", "*.lot", "*.bbl", "*.blg", "*.synctex.gz")
    foreach ($pattern in $auxFiles) {
        Remove-Item $pattern -ErrorAction SilentlyContinue
    }
    Write-Host "Auxiliary files cleaned!" -ForegroundColor Green
}

function Clean-All {
    Clean-Aux
    Write-Host "Removing PDF..." -ForegroundColor Yellow
    Remove-Item "$MainFile.pdf" -ErrorAction SilentlyContinue
    Write-Host "All generated files removed!" -ForegroundColor Green
}

function View-PDF {
    if (Test-Path "$MainFile.pdf") {
        Write-Host "Opening $MainFile.pdf..." -ForegroundColor Green
        Start-Process "$MainFile.pdf"
    } else {
        Write-Host "PDF file not found. Please compile first." -ForegroundColor Red
    }
}

function Show-Help {
    Write-Host ""
    Write-Host "LaTeX Compilation Script" -ForegroundColor Cyan
    Write-Host "========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\compile.ps1 [target]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available targets:" -ForegroundColor Green
    Write-Host "  all      - Full compilation (run pdflatex twice)" -ForegroundColor White
    Write-Host "  quick    - Quick compilation (single pdflatex pass)" -ForegroundColor White
    Write-Host "  clean    - Remove auxiliary files" -ForegroundColor White
    Write-Host "  cleanall - Remove all generated files including PDF" -ForegroundColor White
    Write-Host "  view     - Open the PDF file" -ForegroundColor White
    Write-Host "  help     - Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\compile.ps1        # Default: full compilation" -ForegroundColor White
    Write-Host "  .\compile.ps1 quick  # Quick single-pass compilation" -ForegroundColor White
    Write-Host "  .\compile.ps1 clean  # Clean auxiliary files" -ForegroundColor White
    Write-Host ""
}

# Execute based on target
switch ($Target) {
    "all"      { Compile-All }
    "quick"    { Compile-Quick }
    "clean"    { Clean-Aux }
    "cleanall" { Clean-All }
    "view"     { View-PDF }
    "help"     { Show-Help }
    default    { Compile-All }
}
