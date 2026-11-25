# Shared, side-effect-free path resolution for both PowerShell entry points.
function Resolve-LauncherPath {
    <#
    .SYNOPSIS
    Resolve a nonempty path relative to the configuration directory.
    #>
    param([object]$Value, [string]$Anchor)
    if ($Value -isnot [string] -or [string]::IsNullOrWhiteSpace($Value)) {
        throw 'research_root must be a nonempty filesystem path.'
    }
    Push-Location -LiteralPath $Anchor
    try { return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Value) }
    finally { Pop-Location }
}

function Get-LauncherResearchRoot {
    <#
    .SYNOPSIS
    Match Python configuration precedence without opening credentials.
    #>
    param([string]$ProjectRoot)
    $configRoot = Join-Path $ProjectRoot 'configs'
    $location = Join-Path $configRoot 'project_settings.json'
    $value = '..'
    if (Test-Path -LiteralPath $location -PathType Leaf) {
        $config = Get-Content -LiteralPath $location -Raw | ConvertFrom-Json
        if ($null -eq $config -or $config -isnot [pscustomobject]) {
            throw 'Project configuration must be a JSON object.'
        }
        $unknown = @(foreach ($property in $config.PSObject.Properties) {
            if ($property.Name -cnotin @('research_root','source_project','output_root')) { $property.Name }
        })
        if ($unknown.Count -gt 0) { throw ('Unknown project path settings: ' + ($unknown -join ', ')) }
        if ($null -ne $config.PSObject.Properties['research_root']) { $value = $config.research_root }
    }
    $override = [Environment]::GetEnvironmentVariable('MQF_RESEARCH_ROOT', 'Process')
    if ($null -ne $override) { $value = $override }
    return Resolve-LauncherPath -Value $value -Anchor $configRoot
}

function Invoke-MqfCommand {
    <#
    .SYNOPSIS
    Invoke Python with explicit package roots and restore the caller environment.
    .DESCRIPTION
    Arguments remain an array. No string-built shell command, installation, or
    network request is performed here. The invoked Python command owns its work.
    #>
    param([string]$ProjectRoot,[string]$Python,[string[]]$PythonArguments)
    $researchRoot = Get-LauncherResearchRoot -ProjectRoot $ProjectRoot
    if (-not $Python) {
        $localPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $localPython -PathType Leaf) { $Python = $localPython }
        else { $Python = Join-Path $researchRoot '.venv_codex_tags\Scripts\python.exe' }
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python not found at '$Python'. Pass -Python with a configured executable."
    }
    $Python = (Resolve-Path -LiteralPath $Python).Path
    $priorPythonPath = $env:PYTHONPATH
    $priorBytecode = $env:PYTHONDONTWRITEBYTECODE
    try {
        $paths = @((Join-Path $ProjectRoot 'src'))
        $developmentTools = Join-Path $ProjectRoot '.dev_dependencies'
        if (-not (Test-Path -LiteralPath $developmentTools -PathType Container)) {
            $developmentTools = Join-Path $researchRoot 'maintained\.dev_dependencies'
        }
        if (Test-Path -LiteralPath $developmentTools -PathType Container) { $paths += $developmentTools }
        $paths += $researchRoot
        if ($priorPythonPath) { $paths += $priorPythonPath }
        $env:PYTHONPATH = $paths -join [IO.Path]::PathSeparator
        $env:PYTHONDONTWRITEBYTECODE = '1'
        & $Python @PythonArguments | Out-Host
        $commandExit = $LASTEXITCODE
    } finally {
        $env:PYTHONPATH = $priorPythonPath
        $env:PYTHONDONTWRITEBYTECODE = $priorBytecode
    }
    return $commandExit
}