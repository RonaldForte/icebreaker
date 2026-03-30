$ErrorActionPreference = "Stop"

# Resolve all paths from the script location so the launcher works from any terminal cwd.
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot "venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$pipExe = Join-Path $venvPath "Scripts\pip.exe"
$uvicornExe = Join-Path $venvPath "Scripts\uvicorn.exe"
$streamlitExe = Join-Path $venvPath "Scripts\streamlit.exe"

Set-Location $projectRoot

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment..."
    py -m venv venv
}

Write-Host "Installing Python dependencies..."
& $pythonExe -m pip install --upgrade pip
& $pipExe install -r (Join-Path $projectRoot "requirements.txt")

# Pulling the Ollama models here makes first-run setup predictable for demos.
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ensuring Ollama models are available..."
    & ollama pull llama3.2:3b
    & ollama pull nomic-embed-text
} else {
    Write-Warning "Ollama CLI was not found. Install Ollama and pull llama3.2:3b plus nomic-embed-text before using AI features."
}

# Launch FastAPI and Streamlit in separate PowerShell windows so the logs stay visible.
$backendCommand = "Set-Location '$projectRoot'; & '$uvicornExe' app.main:app --reload"
$streamlitCommand = "Set-Location '$projectRoot'; & '$streamlitExe' run app/streamlit.py"

Write-Host "Starting FastAPI backend..."
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $backendCommand) | Out-Null

Start-Sleep -Seconds 4

Write-Host "Starting Streamlit UI..."
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $streamlitCommand) | Out-Null

Start-Sleep -Seconds 5

Write-Host "Opening Streamlit in your browser..."
Start-Process "http://127.0.0.1:8501"