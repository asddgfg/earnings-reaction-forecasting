param([string]$Python)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Launcher-Common.ps1')
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
exit (Invoke-MqfCommand -ProjectRoot $projectRoot -Python $Python -PythonArguments @((Join-Path $projectRoot 'tools\run_checks.py')))