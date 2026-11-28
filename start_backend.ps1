$env:PYTHONPATH = "src;.;$PSScriptRoot\.venv\Lib\site-packages"
& "C:\Users\suraj\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe" -m uvicorn sql_grpo.serving.api:app --host 0.0.0.0 --port 8000
