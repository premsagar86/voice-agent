# Optional: build a single tray executable with PyInstaller (dist\voice-agent.exe).
# Whisper models are downloaded on first run, they are not bundled.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python -m pip install pyinstaller
python -m PyInstaller --onefile --noconsole --name voice-agent `
    --collect-all faster_whisper --collect-all ctranslate2 `
    --hidden-import pystray._win32 run.py
Write-Host "Built: $here\dist\voice-agent.exe (start it with --tray)"
