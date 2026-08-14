param(
    [int]$Camera0 = 0,
    [int]$Camera1 = 1,
    [int]$Width = 1280,
    [int]$Height = 720,
    [double]$TargetFps = 24,
    [double]$RecalibrateSeconds = 5,
    [double]$DurationSeconds = 0,
    [ValidateSet("fast", "feather")][string]$BlendMode = "feather",
    [double]$FeatherRadius = 48,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $python) { throw "python was not found on PATH. Install Python 3 and reopen PowerShell." }
& $python.Source -c "import cv2, numpy" 2>$null
if ($LASTEXITCODE -ne 0) { throw "Install dependencies with: python -m pip install numpy opencv-python" }

$arguments = @(
    "run_realtime.py",
    "--camera0", $Camera0,
    "--camera1", $Camera1,
    "--width", $Width,
    "--height", $Height,
    "--target-fps", $TargetFps,
    "--recalibrate-seconds", $RecalibrateSeconds,
    "--blend-mode", $BlendMode,
    "--feather-radius", $FeatherRadius
)
if ($DurationSeconds -gt 0) { $arguments += @("--duration-seconds", $DurationSeconds) }
if ($Rebuild) { $arguments += @("--rebuild-cuda", "--rebuild-renderer") }
& $python.Source @arguments
exit $LASTEXITCODE
