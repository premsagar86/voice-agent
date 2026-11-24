# Start the voice agent. Usage:  .\start-voice.ps1            (console)
#                                .\start-voice.ps1 -Tray      (system tray icon)
# Run it from the project folder you want Claude to work in, or pass -Project.
param([switch]$Tray, [string]$Project = (Get-Location).Path)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Project
$extra = @()
if ($Tray) { $extra += "--tray" }
python (Join-Path $here "run.py") @extra
