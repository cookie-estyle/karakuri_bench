import weave
import re
from weave import Model, Evaluation
from openai import OpenAI
import asyncio
from pydantic import BaseModel
from predictor import ModelTemplate
from pydantic import BaseModel, PrivateAttr
from dotenv import load_dotenv
import os
from vllm_server import start_vllm_server, stop_vllm_server
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import random
from typing import Dict
import json
import os
import boto3
from botocore.config import Config
from pydantic import BaseModel
from models import MODELS_TO_EVALUATE

load_dotenv(override=True)

# weave
PROJECT = 'karakuri-bench/karakuri-bench'
DATASET_REF = 'karakuri-bench-dataset:latest'
EVALUATE_PROMPT_REF = 'evaluate_prompt:latest'

# evaluation
EVALUATE_MODEL_NAME = 'gpt-4o-2024-08-06'

# execution
MAX_RETRY_COUNT = 5
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 30

class EvaluationResult(BaseModel):
    response_text: str
    category: str
    score: int | None
    retries: int = 0

@weave.op()
async def evaluate(
    output,
    index=None, 
    category=None, 
    question=None,
    example_answer=None,
    marking_scheme=None,
) -> EvaluationResult:

    if output is None:
        return {
            "response_text": "Prediction failed: No output received",
            "category": category,
            "score": {
                "overall": None,
                f"{category}": None
            },
            "retries": 0
        }

    if not isinstance(output, dict) or 'answer' not in output:
        return {
            "response_text": f"Prediction failed: Invalid output format - {str(output)}",
            "category": category,
            "score": {
                "overall": None,
                f"{category}": None
            },
            "retries": 0
        }

    retries = 0
    last_error = None
    client = OpenAI()

    user_prompt = USER_PROMPT.format(
        question=question,
        example_answer=example_answer,
        marking_scheme=marking_scheme,
        answer=output['answer'],
    ).strip()

    for attempt in range(MAX_RETRY_COUNT):
        try:
            try:
                response = client.chat.completions.create(
                    model=EVALUATE_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT.strip()},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
            except Exception as e:
                print(f"Evaluation error: {str(e)}")
                last_error = e
            
            response_text = response.choices[0].message.content
            try:
                score_text = response_text.split("【評点】")[1].strip()
                score_match = re.search(r'[1-5]', score_text)
                if score_match:
                    score = int(score_match.group())
                else:
                    raise ValueError("No valid score found")
                
                return {
                    "response_text": response_text,
                    "category": category,
                    "score": {
                        "overall": score,
                        f"{category}": score
                    },
                    "retries": retries
                }
            except Exception as e:
                print(f"Score parsing error: {str(e)}")
                last_error = e

        except Exception as e:
            print(f"Evaluation error (attempt {attempt + 1}/{MAX_RETRY_COUNT}): {str(e)}")
            last_error = e

        retries += 1
        if attempt < MAX_RETRY_COUNT - 1:
            delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** attempt))
            await asyncio.sleep(delay)

    return {
        "response_text": f"Evaluation failed after {retries} attempts. Last error: {str(last_error)}",
        "category": category,
        "score": {
            "overall": None,
            f"{category}": None
        },
        "retries": retries
    }

async def evaluate_all_models():
    weave.init(PROJECT)
    
    dataset = weave.ref(DATASET_REF).get()
    dataset = [{key: value for key, value in row.items()} for row in dataset.rows]
    
    prompt_dataset = weave.ref(EVALUATE_PROMPT_REF).get()
    global SYSTEM_PROMPT, USER_PROMPT
    SYSTEM_PROMPT = prompt_dataset.messages[0].get('content')
    USER_PROMPT = prompt_dataset.messages[1].get('content')
    
    results = {}
    
    for model_config in MODELS_TO_EVALUATE:
        api_type = model_config['api_type']
        model_name = model_config['model_name']
        
        print(f"Evaluating model: {model_name} with API type: {api_type}")
        
        # vLLM サーバーの起動（必要な場合）
        if api_type == 'vllm':
            start_vllm_server(model_name)
        
        # モデルクラスの作成
        class_name = model_name.replace('-', '_').replace('.', '_').replace(':','_').replace('/','_')
        if re.match(r'^\d', class_name):
            class_name = f"_{class_name}"
        model_template = ModelTemplate.get_template(api_type, model_name, class_name)
        exec(model_template, globals())
        
        # モデルのインスタンス化
        model = eval(f"{class_name}")(predict_model_name=model_name)
        
        # 評価の実行
        evaluation = Evaluation(dataset=dataset, scorers=[evaluate])
        result = await evaluation.evaluate(model)
        
        # 結果の保存
        results[f"{api_type}_{model_name}"] = result
        
        # vLLM サーバーの停止（必要な場合）
        if api_type == 'vllm':
            stop_vllm_server()
        
    return results

if __name__ == "__main__":
    asyncio.run(evaluate_all_models())