from huggingface_hub import HfApi
import os
import tempfile
import subprocess
import torch
import time
import atexit
import sys
import signal
import requests
from pathlib import Path
import json

def start_vllm_server(model_id):
    api = HfApi()
    tokenizer_config = api.hf_hub_download(
        repo_id=model_id,
        filename="tokenizer_config.json",
        revision="main",
        use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
    )
    with Path(tokenizer_config).open("r") as f:
        tokenizer_config = json.load(f)

    chat_template: str = tokenizer_config.get("chat_template", None)

    if chat_template is None:
        raise ValueError("chat_template is None. Please provide a valid chat_template in the configuration.")
    
    available_gpus = torch.cuda.device_count()

    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        # chat_templateをファイルに書き込んでパスを取得
        temp_file.write(chat_template)
        chat_template_path = temp_file.name

        # サーバーを起動するためのコマンド
        command = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--served-model-name", model_id,
            "--dtype", "float16",
            "--chat-template", chat_template_path,
            "--max-model-len", "3000",
            "--max-num-seqs", "40",
            "--tensor-parallel-size", str(available_gpus),
            "--device", "auto",
            "--seed", "42",
            "--uvicorn-log-level", "warning",
            "--disable-log-stats",
            "--disable-log-requests",
            "--revision", "main",
        ]

        process = subprocess.Popen(command)

    with open('vllm_server.pid', 'w') as pid_file:
        pid_file.write(str(process.pid))
    time.sleep(10)

    print("vLLM server is starting...")

    # スクリプト終了時にサーバーを終了する
    def cleanup():
        print("Terminating vLLM server...")
        process.terminate()
        process.wait()

    atexit.register(cleanup)
    
    # SIGTERMシグナルをキャッチしてサーバーを終了する
    def handle_sigterm(signal, frame):
        print("SIGTERM received. Shutting down vLLM server gracefully...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    # サーバーが完全に起動するのを待つ
    
    url = "http://localhost:8000/health"
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Health check passed!")
                break
            else:
                print(f"Health check failed with status code: {response.status_code}")
        except requests.ConnectionError:
            print("Failed to connect to the server. Retrying...")
        time.sleep(10)  # 待機してから再試行
