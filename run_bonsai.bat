@echo off
rem Starts Ternary Bonsai 2 27B on PrismML's llama.cpp fork: OpenAI-compatible API at http://127.0.0.1:8081/v1
rem Needs ~6.3 GB VRAM - unload Ollama models first (ollama ps). Ctrl+C stops it.
set BONSAI=D:\bonsai

curl -s http://127.0.0.1:8081/health | find "ok" >nul && (echo Bonsai is already running on http://127.0.0.1:8081 & exit /b 0)

"%BONSAI%\llama-prism\llama-server.exe" -m "%BONSAI%\Ternary-Bonsai-2-27B-PTQ1_0.gguf" ^
  -ngl 99 -fa on -c 8192 -np 1 --reasoning off ^
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 ^
  --host 127.0.0.1 --port 8081
