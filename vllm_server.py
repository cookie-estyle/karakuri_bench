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
    """
    指定されたモデルIDでvLLMサーバーを起動します
    
    Args:
        model_id (str): Hugging Face モデルID
        
    Note:
        サーバーは起動後、ヘルスチェックが成功するまで待機します。
        プロセスIDは vllm_server.pid ファイルに保存されます。
    """
    chat_template: str = get_chat_template(model_id)

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
            "--trust-remote-code",
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

def stop_vllm_server():
    """
    実行中のvLLMサーバーを停止します
    
    vllm_server.pid ファイルからプロセスIDを読み取り、
    SIGTERMシグナルを送信してサーバーを停止します。
    """
    pid_file = Path('vllm_server.pid')
    if pid_file.exists():
        with pid_file.open('r') as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to vLLM server process {pid}")
            # ファイルを削除
            pid_file.unlink()
        except ProcessLookupError:
            print(f"Process {pid} not found")
        except Exception as e:
            print(f"Error stopping vLLM server: {e}")

def get_chat_template(model_id):
    """
    モデルのchat_templateを取得する関数
    
    Args:
        model_id (str): Hugging Face モデルID
        
    Returns:
        str: 処理済みのchat_template
        
    Raises:
        ValueError: chat_templateが見つからない、または無効な形式の場合
    """
    api = HfApi()
    
    tokenizer_config = api.hf_hub_download(
        repo_id=model_id,
        filename="tokenizer_config.json",
        revision="main",
        use_auth_token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
    )
    
    with Path(tokenizer_config).open("r") as f:
        tokenizer_config = json.load(f)
    
    chat_template = tokenizer_config.get("chat_template", None)
    
    if chat_template is None:
        raise ValueError(f"No chat_template found for model {model_id}")
    
    # chat_templateの処理
    if isinstance(chat_template, list):
        # リストの場合、"default"という名前の要素を探す
        default_found = False
        for template_obj in chat_template:
            if isinstance(template_obj, dict):
                if template_obj.get("name") == "default" and "template" in template_obj:
                    chat_template = template_obj.get("template")
                    default_found = True
                    break
        
        # defaultが見つからなければ最初の要素を使用
        if not default_found and len(chat_template) > 0:
            if isinstance(chat_template[0], dict) and "template" in chat_template[0]:
                chat_template = chat_template[0].get("template")
            else:
                raise ValueError(f"Invalid chat_template format for model {model_id}: first item in list is not a valid template object")
    
    # 最終的にchat_templateが文字列でない場合はエラー
    if not isinstance(chat_template, str):
        raise ValueError(f"Invalid chat_template format for model {model_id}: not a string")
    
    return chat_template