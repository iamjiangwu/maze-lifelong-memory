param(
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Python="python",
    [string]$Artifact="all",
    [string]$NodePath="",
    [string]$ArtifactToolModules="",
    [switch]$NoCompare,
    [switch]$StatisticsOnly
)
$ErrorActionPreference="Stop"
$releaseNodeRoot=Join-Path $env:USERPROFILE ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
if (!$NodePath) {
    $releaseBundledNode=Join-Path $releaseNodeRoot "bin/node.exe"
    $NodePath=if (Test-Path -LiteralPath $releaseBundledNode) {$releaseBundledNode} else {"node"}
}
if (!$ArtifactToolModules) {
    $releaseBundledModules=Join-Path $releaseNodeRoot "node_modules"
    if (Test-Path -LiteralPath $releaseBundledModules) {$ArtifactToolModules=$releaseBundledModules}
}
if ($ArtifactToolModules) {$env:ARTIFACT_TOOL_NODE_MODULES=$ArtifactToolModules}
$releaseArguments=@("-B",(Join-Path $PSScriptRoot "reproduce.py"),"--artifact",$Artifact,"--output-dir",$OutputDir,"--node",$NodePath)
if ($NoCompare) {$releaseArguments+="--no-compare"}
if ($StatisticsOnly) {$releaseArguments+="--statistics-only"}
& $Python @releaseArguments
if ($LASTEXITCODE -ne 0) {throw "Source-data reproduction failed; inspect the output reports and BUILD.log."}
