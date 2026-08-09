# compile.ps1 -- run from the paper/ directory
# Compiles manuscript_revised.tex -> manuscript_revised.pdf
# Requires MiKTeX (pdflatex + bibtex on PATH)

Set-Location $PSScriptRoot

$tex = "manuscript_revised.tex"
$base = "manuscript_revised"

Write-Host "=== Pass 1: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode $tex

Write-Host "=== bibtex ===" -ForegroundColor Cyan
bibtex $base

Write-Host "=== Pass 2: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode $tex

Write-Host "=== Pass 3: pdflatex (resolve all refs) ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode $tex

if (Test-Path "$base.pdf") {
    Write-Host "`nDone. PDF at: $(Resolve-Path "$base.pdf")" -ForegroundColor Green
    Start-Process "$base.pdf"   # open the PDF
} else {
    Write-Host "`nCompile failed -- check $base.log for errors." -ForegroundColor Red
}
