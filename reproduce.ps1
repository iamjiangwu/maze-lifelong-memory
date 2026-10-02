param(
    [string]$Figure = "all",
    [string]$OutputDir = "",
    [string]$Python = "python",
    [switch]$StatisticsOnly,
    [switch]$FiguresOnly,
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
$buildArguments = @("-B", (Join-Path $PSScriptRoot "reproduce.py"), "--figure", $Figure)
if ($OutputDir) { $buildArguments += @("--output-dir", $OutputDir) }
if ($StatisticsOnly) { $buildArguments += "--statistics-only" }
if ($FiguresOnly) { $buildArguments += "--figures-only" }
if ($CheckOnly) { $buildArguments += "--check-only" }
& $Python @buildArguments
if ($LASTEXITCODE -ne 0) { throw "Reproduction failed; inspect the output BUILD.log." }
