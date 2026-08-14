param(
    [string]$Output = ".\output\build\panocall_renderer.dll",
    [string]$Architecture = "native"
)

$ErrorActionPreference = "Stop"
$source = Join-Path $PSScriptRoot "src\cuda_panorama_renderer.cu"
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($null -eq $nvcc) { throw "nvcc was not found. Install the CUDA Toolkit and reopen PowerShell." }
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Output))
New-Item -ItemType Directory -Force ([System.IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$arguments = @($source, "-O3", "--use_fast_math", "-shared", "-Xcompiler", "/MD", "-o", $outputPath)
if ($Architecture -ne "native") { $arguments += @("-arch", $Architecture) }
& $nvcc.Source @arguments
if ($LASTEXITCODE -ne 0) { throw "CUDA renderer compilation failed with exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) { throw "Renderer DLL was not created: $outputPath" }
Write-Host "Built CUDA renderer: $outputPath"
