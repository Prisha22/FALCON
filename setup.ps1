<#
.SYNOPSIS
    First-clone bootstrap for FALCON: Python deps, SU2, XFOIL and MS-MPI.

.DESCRIPTION
    Downloads and installs everything FALCON needs, then puts the executables on
    PATH so the app finds them via shutil.which(). Safe to re-run -- every step
    detects what is already present and skips it.

    Solver binaries land in .\third_party\ by default (inside the repo, ignored
    by git) so uninstalling is just deleting that folder and running
    -RemoveFromPath.

.PARAMETER SystemPath
    Write PATH machine-wide (HKLM) instead of per-user. Requires Administrator.
    Per-user is the default and is sufficient for FALCON.

.PARAMETER ToolsDir
    Where to install SU2 and XFOIL. Default: <repo>\third_party

.PARAMETER SkipPython / SkipSu2 / SkipXfoil / SkipMpi
    Skip individual components.

.PARAMETER RemoveFromPath
    Undo: remove the entries this script added, then exit.

.EXAMPLE
    .\setup.ps1
.EXAMPLE
    .\setup.ps1 -SystemPath          # run from an elevated prompt
.EXAMPLE
    .\setup.ps1 -SkipMpi -SkipXfoil  # SU2 + Python deps only
#>
[CmdletBinding()]
param(
    [switch]$SystemPath,
    [string]$ToolsDir,
    [switch]$SkipPython,
    [switch]$SkipSu2,
    [switch]$SkipXfoil,
    [switch]$SkipMpi,
    [switch]$RemoveFromPath
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# --------------------------------------------------------------------------
# Configuration -- verified against the upstream release pages on 2026-08-02
# --------------------------------------------------------------------------
$SU2_VERSION = 'v8.5.0'
$SU2_URL     = "https://github.com/su2code/SU2/releases/download/$SU2_VERSION/SU2-$SU2_VERSION-win64-mpi.zip"
$XFOIL_URL   = 'https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip'
$MSMPI_URL   = 'https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisetup.exe'

$RepoRoot = $PSScriptRoot
if (-not $ToolsDir) { $ToolsDir = Join-Path $RepoRoot 'third_party' }
$DownloadDir = Join-Path $ToolsDir '_downloads'

$PathScope = 'User'
if ($SystemPath) { $PathScope = 'Machine' }

# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
function Write-Step   { param($m) Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Ok     { param($m) Write-Host "  [ok]   $m" -ForegroundColor Green }
function Write-Info   { param($m) Write-Host "  [..]   $m" -ForegroundColor Gray }
function Write-Warn   { param($m) Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Write-Fail   { param($m) Write-Host "  [FAIL] $m" -ForegroundColor Red }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pr = New-Object Security.Principal.WindowsPrincipal($id)
    return $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# --------------------------------------------------------------------------
# PATH handling
#
# CRITICAL: read the scope's *stored* value via [Environment]::GetEnvironment-
# Variable(...,'User'). Never write $env:PATH back to the User scope -- the
# process PATH is Machine+User merged, so doing that copies every machine entry
# into the user scope and corrupts it permanently.
# --------------------------------------------------------------------------
function Get-StoredPath {
    param([string]$Scope)
    $v = [Environment]::GetEnvironmentVariable('Path', $Scope)
    if ($null -eq $v) { return '' }
    return $v
}

function Add-ToPath {
    param([string]$Directory, [string]$Scope)

    if (-not (Test-Path $Directory)) {
        Write-Warn "not adding missing directory to PATH: $Directory"
        return $false
    }
    $Directory = (Resolve-Path $Directory).Path.TrimEnd('\')

    $stored = Get-StoredPath -Scope $Scope
    $entries = @($stored -split ';' | Where-Object { $_ -ne '' })

    foreach ($e in $entries) {
        if ($e.TrimEnd('\') -ieq $Directory) {
            Write-Ok "already on $Scope PATH: $Directory"
            # Still needs to be live in this session.
            if (($env:PATH -split ';' | Where-Object { $_.TrimEnd('\') -ieq $Directory }).Count -eq 0) {
                $env:PATH = "$env:PATH;$Directory"
            }
            return $true
        }
    }

    $newPath = (@($entries) + $Directory) -join ';'
    if ($newPath.Length -gt 2000) {
        Write-Warn "$Scope PATH is now $($newPath.Length) chars; Windows truncates some legacy readers past ~2048."
    }

    [Environment]::SetEnvironmentVariable('Path', $newPath, $Scope)
    $env:PATH = "$env:PATH;$Directory"      # live in this session too
    Write-Ok "added to $Scope PATH: $Directory"
    return $true
}

function Remove-FromPath {
    param([string]$Directory, [string]$Scope)
    $Directory = $Directory.TrimEnd('\')
    $stored = Get-StoredPath -Scope $Scope
    $entries = @($stored -split ';' | Where-Object { $_ -ne '' })
    $kept = @($entries | Where-Object { $_.TrimEnd('\') -ine $Directory })

    if ($kept.Count -eq $entries.Count) { return $false }
    [Environment]::SetEnvironmentVariable('Path', ($kept -join ';'), $Scope)
    Write-Ok "removed from $Scope PATH: $Directory"
    return $true
}

# --------------------------------------------------------------------------
# Download / extract
# --------------------------------------------------------------------------
function Get-RemoteFile {
    param([string]$Url, [string]$Destination)

    if (Test-Path $Destination) {
        $mb = [math]::Round((Get-Item $Destination).Length / 1MB, 1)
        Write-Ok "already downloaded ($mb MB): $(Split-Path $Destination -Leaf)"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $Destination) | Out-Null
    Write-Info "downloading $Url"

    $partial = "$Destination.partial"
    $progress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is ~10x faster without it
    try {
        Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
        Move-Item $partial $Destination -Force
    } finally {
        $ProgressPreference = $progress
        if (Test-Path $partial) { Remove-Item $partial -Force -ErrorAction SilentlyContinue }
    }

    # Downloads carry Mark-of-the-Web, which makes SmartScreen block execution.
    try { Unblock-File -Path $Destination -ErrorAction Stop } catch { }

    $mb = [math]::Round((Get-Item $Destination).Length / 1MB, 1)
    Write-Ok "downloaded ($mb MB): $(Split-Path $Destination -Leaf)"
}

function Expand-ToDirectory {
    param([string]$ZipPath, [string]$Destination)

    if (Test-Path $Destination) {
        Write-Info "removing previous extract: $Destination"
        Remove-Item $Destination -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Expand-Archive -Path $ZipPath -DestinationPath $Destination -Force

    # Clear MOTW from every extracted binary.
    Get-ChildItem $Destination -Recurse -Include *.exe, *.dll -ErrorAction SilentlyContinue |
        ForEach-Object { try { Unblock-File $_.FullName -ErrorAction Stop } catch { } }
}

function Find-Executable {
    param([string]$Root, [string]$Name)
    $hit = Get-ChildItem -Path $Root -Filter $Name -Recurse -File -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($hit) { return $hit.FullName }
    return $null
}

# Actually launch the binary. Catches Smart App Control / Code Integrity blocks,
# which a mere Test-Path cannot see.
function Test-Runnable {
    param([string]$ExePath, [string[]]$Arguments = @())

    try {
        $psi = New-Object Diagnostics.ProcessStartInfo
        $psi.FileName = $ExePath
        $psi.Arguments = ($Arguments -join ' ')
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        # XFOIL is interactive; hand it EOF immediately so it exits on its own
        # instead of blocking until the timeout.
        $psi.RedirectStandardInput = $true
        $psi.CreateNoWindow = $true

        $p = [Diagnostics.Process]::Start($psi)
        $p.StandardInput.Close()
        if (-not $p.WaitForExit(20000)) { try { $p.Kill() } catch { } }
        return @{ Ok = $true; Message = 'launched' }
    } catch {
        return @{ Ok = $false; Message = $_.Exception.Message }
    }
}

# --------------------------------------------------------------------------
# Uninstall path entries and exit
# --------------------------------------------------------------------------
if ($RemoveFromPath) {
    Write-Step "Removing FALCON tool directories from $PathScope PATH"
    if ($SystemPath -and -not (Test-Admin)) {
        Write-Fail 'Machine PATH requires an elevated prompt.'
        exit 1
    }
    $removed = 0
    foreach ($d in @(
        (Join-Path $ToolsDir 'SU2'),
        (Join-Path $ToolsDir 'xfoil')
    )) {
        Get-ChildItem -Path $d -Directory -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object { if (Remove-FromPath -Directory $_.FullName -Scope $PathScope) { $removed++ } }
        if (Remove-FromPath -Directory $d -Scope $PathScope) { $removed++ }
    }
    Write-Host "`nRemoved $removed entr(ies). Delete $ToolsDir to reclaim the disk space." -ForegroundColor Cyan
    exit 0
}

# ==========================================================================
Write-Host @"

  FALCON setup
  repo    : $RepoRoot
  tools   : $ToolsDir
  PATH    : $PathScope scope
"@ -ForegroundColor White

if ($SystemPath -and -not (Test-Admin)) {
    Write-Fail 'PATH requires Administrator. Re-run from an elevated prompt, or drop -SystemPath.'
    exit 1
}

New-Item -ItemType Directory -Force -Path $ToolsDir, $DownloadDir | Out-Null
$results = [ordered]@{}

# --------------------------------------------------------------------------
# 1. Python environment
# --------------------------------------------------------------------------
if (-not $SkipPython) {
    Write-Step '1/4  Python environment'

    $venv = Join-Path $RepoRoot '.venv'
    $venvPy = Join-Path $venv 'Scripts\python.exe'

    if (Test-Path $venvPy) {
        Write-Ok "virtualenv exists: $venv"
    } else {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) { $bootstrap = 'py' } else {
            $sys = Get-Command python -ErrorAction SilentlyContinue
            if (-not $sys) {
                Write-Fail 'No Python found. Install Python 3.12+ from https://python.org and re-run.'
                exit 1
            }
            $bootstrap = 'python'
        }
        Write-Info "creating virtualenv with $bootstrap"
        & $bootstrap -m venv $venv
        if (-not (Test-Path $venvPy)) { Write-Fail 'virtualenv creation failed'; exit 1 }
        Write-Ok "created $venv"
    }

    Write-Info 'installing requirements (this takes a few minutes)'
    & $venvPy -m pip install --upgrade pip --quiet --disable-pip-version-check
    & $venvPy -m pip install -r (Join-Path $RepoRoot 'requirements.txt') --quiet --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        Write-Fail 'pip install failed -- see output above'
        $results['Python packages'] = 'FAILED'
    } else {
        Write-Ok 'python packages installed'
        $results['Python packages'] = 'ok'
    }
} else {
    Write-Info 'skipping Python'
}

# --------------------------------------------------------------------------
# 2. SU2
# --------------------------------------------------------------------------
if (-not $SkipSu2) {
    Write-Step "2/4  SU2 $SU2_VERSION"

    $existing = Get-Command SU2_CFD -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Ok "SU2_CFD already on PATH: $($existing.Source)"
        $results['SU2'] = "already installed: $($existing.Source)"
    } else {
        $zip = Join-Path $DownloadDir "SU2-$SU2_VERSION-win64-mpi.zip"
        $dir = Join-Path $ToolsDir 'SU2'
        Get-RemoteFile -Url $SU2_URL -Destination $zip
        Write-Info 'extracting SU2'
        Expand-ToDirectory -ZipPath $zip -Destination $dir

        $exe = Find-Executable -Root $dir -Name 'SU2_CFD.exe'
        if (-not $exe) {
            Write-Fail "SU2_CFD.exe not found under $dir"
            $results['SU2'] = 'FAILED (binary not found in archive)'
        } else {
            $binDir = Split-Path $exe -Parent
            Write-Ok "SU2_CFD.exe at $exe"
            Add-ToPath -Directory $binDir -Scope $PathScope | Out-Null
            # SU2's own python helpers read SU2_RUN.
            [Environment]::SetEnvironmentVariable('SU2_RUN', $binDir, $PathScope)
            $env:SU2_RUN = $binDir
            Write-Ok "SU2_RUN = $binDir"
            $results['SU2'] = $binDir
        }
    }
} else {
    Write-Info 'skipping SU2'
}

# --------------------------------------------------------------------------
# 3. XFOIL
# --------------------------------------------------------------------------
if (-not $SkipXfoil) {
    Write-Step '3/4  XFOIL 6.99'

    $existing = Get-Command xfoil -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Ok "xfoil already on PATH: $($existing.Source)"
        $results['XFOIL'] = "already installed: $($existing.Source)"
    } else {
        $zip = Join-Path $DownloadDir 'XFOIL6.99.zip'
        $dir = Join-Path $ToolsDir 'xfoil'
        Get-RemoteFile -Url $XFOIL_URL -Destination $zip
        Write-Info 'extracting XFOIL'
        Expand-ToDirectory -ZipPath $zip -Destination $dir

        $exe = Find-Executable -Root $dir -Name 'xfoil.exe'
        if (-not $exe) {
            Write-Fail "xfoil.exe not found under $dir"
            $results['XFOIL'] = 'FAILED (binary not found in archive)'
        } else {
            $binDir = Split-Path $exe -Parent
            Write-Ok "xfoil.exe at $exe"
            Add-ToPath -Directory $binDir -Scope $PathScope | Out-Null
            $results['XFOIL'] = $binDir
        }
    }
} else {
    Write-Info 'skipping XFOIL'
}

# --------------------------------------------------------------------------
# 4. Microsoft MPI  (installer needs elevation)
# --------------------------------------------------------------------------
if (-not $SkipMpi) {
    Write-Step '4/4  Microsoft MPI'

    $existing = Get-Command mpiexec -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Ok "mpiexec already on PATH: $($existing.Source)"
        $results['MS-MPI'] = "already installed: $($existing.Source)"
    } elseif (-not (Test-Admin)) {
        Write-Warn 'MS-MPI needs Administrator to install. Skipping.'
        Write-Warn 'Re-run this script from an elevated prompt, or install manually:'
        Write-Warn "  $MSMPI_URL"
        $results['MS-MPI'] = 'SKIPPED (needs admin)'
    } else {
        $setup = Join-Path $DownloadDir 'msmpisetup.exe'
        Get-RemoteFile -Url $MSMPI_URL -Destination $setup
        Write-Info 'running MS-MPI installer (unattended)'
        $p = Start-Process -FilePath $setup -ArgumentList '-unattend', '-force' -Wait -PassThru
        if ($p.ExitCode -ne 0) {
            Write-Fail "MS-MPI installer exited with code $($p.ExitCode)"
            $results['MS-MPI'] = "FAILED (exit $($p.ExitCode))"
        } else {
            # The installer writes MSMPI_BIN to machine env; pick it up now.
            $mpiBin = [Environment]::GetEnvironmentVariable('MSMPI_BIN', 'Machine')
            if (-not $mpiBin) { $mpiBin = Join-Path $env:ProgramFiles 'Microsoft MPI\Bin' }
            if (Test-Path $mpiBin) {
                $env:PATH = "$env:PATH;$mpiBin"
                Write-Ok "MS-MPI installed: $mpiBin"
                $results['MS-MPI'] = $mpiBin
            } else {
                Write-Warn "installer finished but $mpiBin not found; open a new shell and check mpiexec"
                $results['MS-MPI'] = 'installed (path unconfirmed)'
            }
        }
    }
} else {
    Write-Info 'skipping MS-MPI'
}

# --------------------------------------------------------------------------
# Verification -- resolve AND launch each binary
# --------------------------------------------------------------------------
Write-Step 'Verifying'

$checks = @(
    @{ Name = 'SU2_CFD'; Args = @() },
    @{ Name = 'mpiexec'; Args = @('-help') },
    @{ Name = 'xfoil';   Args = @() }
)

$blocked = @()
foreach ($c in $checks) {
    $cmd = Get-Command $c.Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Fail "$($c.Name) does not resolve on PATH"
        continue
    }
    $r = Test-Runnable -ExePath $cmd.Source -Arguments $c.Args
    if ($r.Ok) {
        Write-Ok "$($c.Name) runs -> $($cmd.Source)"
    } else {
        Write-Fail "$($c.Name) found but WILL NOT RUN: $($r.Message)"
        $blocked += $c.Name
    }
}

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
Write-Host "`n================ Summary ================" -ForegroundColor White
foreach ($k in $results.Keys) {
    Write-Host ("  {0,-16} {1}" -f $k, $results[$k])
}

if ($blocked.Count -gt 0) {
    Write-Host @"

  !! $($blocked -join ', ') installed but blocked from running.

  The usual cause on Windows 11 is Smart App Control / WDAC refusing unsigned
  binaries. SU2 and XFOIL ship unsigned, so a machine with it enforced will
  block them. Check with:

      Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' |
          Select VerifiedAndReputablePolicyState
      # 0 = off, 1 = enforced, 2 = evaluation

  Also check the block record:
      Get-WinEvent -LogName Microsoft-Windows-CodeIntegrity/Operational -MaxEvents 20

  Turning Smart App Control off is IRREVERSIBLE without reinstalling Windows,
  so decide deliberately: Windows Security > App & browser control >
  Smart App Control.
"@ -ForegroundColor Yellow
}

Write-Host @"

  Open a NEW terminal for the PATH change to take effect, then:

      .\.venv\Scripts\python.exe main.py

"@ -ForegroundColor Cyan
