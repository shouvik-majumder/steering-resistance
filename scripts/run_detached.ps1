<#
Run any project script as a detached background process with logs, so it survives closing the
terminal or the Claude session. Usage (PowerShell, from D:\dev\ESR):

  .\scripts\run_detached.ps1 03_run_esr.py --model gemma-2b --n-latents 20 --trials-per-latent 10
  Get-Content data\results\logs\03_run_esr_*.log -Wait -Tail 30     # follow the log

Stop it with:  Get-Process python | Where-Object { $_.Path -like '*envs\esr*' } | Stop-Process
#>
param(
    [Parameter(Mandatory = $true, Position = 0)] [string] $Script,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]] $ScriptArgs
)
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = "$env:USERPROFILE\.conda\envs\esr\python.exe"
if (-not (Test-Path $python)) { throw "conda env 'esr' not found at $python" }
$logDir = Join-Path $root 'data\results\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stem = [IO.Path]::GetFileNameWithoutExtension($Script)
$log = Join-Path $logDir "${stem}_$stamp.log"
$env:PYTHONIOENCODING = 'utf-8'
$argList = @('-u', (Join-Path $root "scripts\$Script")) + $ScriptArgs
$proc = Start-Process -FilePath $python -ArgumentList $argList -WorkingDirectory $root `
    -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$', '.err.log') `
    -WindowStyle Hidden -PassThru
Write-Output "PID $($proc.Id)  log: $log"
