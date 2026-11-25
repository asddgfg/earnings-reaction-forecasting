param(
    [Parameter(Position=0, ValueFromRemainingArguments=$true)][string[]]$Arguments,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Launcher-Common.ps1')
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Arguments) { $Arguments = @('doctor') }
exit (Invoke-MqfCommand -ProjectRoot $projectRoot -Python $Python -PythonArguments (@('-m','mqf_research') + $Arguments))