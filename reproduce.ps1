param(
    [string]$Figure = "all",
    [string]$OutputDir = "",
    [string]$Python = "python",
    [string]$NodePath = "",
    [string]$ArtifactToolModules = "",
    [switch]$StatisticsOnly,
    [switch]$FiguresOnly,
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
if ($ArtifactToolModules) { $env:ARTIFACT_TOOL_NODE_MODULES = $ArtifactToolModules }
$buildArguments = @("-B", (Join-Path $PSScriptRoot "reproduce.py"), "--figure", $Figure)
if ($OutputDir) { $buildArguments += @("--output-dir", $OutputDir) }
if ($NodePath) { $buildArguments += @("--node", $NodePath) }
if ($StatisticsOnly) { $buildArguments += "--statistics-only" }
if ($FiguresOnly) { $buildArguments += "--figures-only" }
if ($CheckOnly) { $buildArguments += "--check-only" }
& $Python @buildArguments
if ($LASTEXITCODE -ne 0) { throw "Reproduction failed; inspect the output BUILD.log." }
