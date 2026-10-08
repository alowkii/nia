@echo off
rem Starts Bonsai 2 27B on llama-server with the settings from the HUD (settings.py / settings.json).
rem Needs ~6.3 GB VRAM - unload Ollama models first (ollama ps). Ctrl+C stops it.
cd /d "%~dp0"
".venv\Scripts\python.exe" -c "import subprocess, settings, agent.server as c; s = settings.load(); print('Bonsai is already running on port', s['port']) if c.llm_up(s['port']) else subprocess.run(c.llm_server_command(s))"
