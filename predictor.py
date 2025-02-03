from typing import Dict
import json
import os
import boto3
from botocore.config import Config
from pydantic import BaseModel

class ModelTemplate:
    @classmethod
    def get_template(cls, api_type: str, model_name: str, class_name: str) -> str:
        templates = {
            "openai": {
                "standard": cls._get_openai_standard_template,
                "o_series": cls._get_openai_o_series_template
            },
            "bedrock": {
                "standard": cls._get_bedrock_template
            }
        }
        
        if api_type == "bedrock":
            template_type = "standard"
        else:
            template_type = "o_series" if model_name.startswith("o") else "standard"
            
        return templates[api_type][template_type](class_name)

    @staticmethod
    def _get_openai_standard_template(class_name: str) -> str:
        return f'''
class {class_name}(Model):
    predict_model_name: str

    @weave.op()
    def predict(self, question: str) -> dict:
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.predict_model_name,
                messages=[
                    {{"role": "user", "content": question}}
                ],
                temperature=0.0,
                response_format={{"type": "text"}},
            )
            answer = response.choices[0].message.content
            return {{'answer': answer, 'question': question}}
        except Exception as e:
            print(f"Prediction error: {{str(e)}}")
            raise
'''

    @staticmethod
    def _get_openai_o_series_template(class_name: str) -> str:
        return f'''
class {class_name}(Model):
    predict_model_name: str

    @weave.op()
    def predict(self, question: str) -> dict:
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.predict_model_name,
                messages=[
                    {{"role": "user", "content": question}}
                ],
                response_format={{"type": "text"}},
            )
            answer = response.choices[0].message.content
            return {{'answer': answer, 'question': question}}
        except Exception as e:
            print(f"Prediction error: {{str(e)}}")
            raise
'''

    @staticmethod
    def _get_bedrock_template(class_name: str) -> str:
        return f'''
import json
import os
import boto3
import time
import random
from botocore.config import Config
from pydantic import BaseModel, PrivateAttr
from tenacity import retry, stop_after_attempt, wait_exponential

class {class_name}(Model):
    predict_model_name: str
    _bedrock_runtime: object = PrivateAttr(default=None)
    _generator_config: dict = PrivateAttr(default={{"temperature": 0.7, "top_p": 0.9}})
    _last_request_time: float = PrivateAttr(default=0)
    _min_request_interval: float = PrivateAttr(default=1.0)  # 最小リクエスト間隔（秒）
    
    def __init__(self, **data):
        super().__init__(**data)
        self._bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            config=Config(read_timeout=1000),
        )

    def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            # ジッターを追加してサーバーへの負荷を分散
            sleep_time += random.uniform(0, 0.1)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _format_llama_prompt(self, messages):
        formatted_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted_prompt += f"[INST] <<SYS>>\\n{{msg['content']}}\\n<</SYS>>\\n[/INST]\\n"
            elif msg["role"] == "user":
                formatted_prompt += f"[INST] {{msg['content']}} [/INST]\\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"{{msg['content']}}\\n"
        return formatted_prompt

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True
    )
    def _invoke_model(self, body_dict):
        self._wait_for_rate_limit()
        return self._bedrock_runtime.invoke_model(
            body=json.dumps(body_dict),
            modelId=self.predict_model_name
        )

    @weave.op()
    def predict(self, question: str) -> dict:
        try:
            messages = [{{"role": "user", "content": question}}]
            is_claude = "anthropic" in self.predict_model_name.lower()
            is_llama = "llama" in self.predict_model_name.lower()

            if is_claude:
                body_dict = {{
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": messages,
                    **self._generator_config,
                }}
            elif is_llama:
                prompt = self._format_llama_prompt(messages)
                body_dict = {{
                    "prompt": prompt,
                    "max_gen_len": 1024,
                    **self._generator_config,
                }}
            else:
                raise ValueError(f"Unsupported model: {{self.predict_model_name}}")

            try:
                response = self._invoke_model(body_dict)
                response_body = json.loads(response.get("body").read())
                
                if is_claude:
                    answer = response_body.get("content")[0].get("text", "")
                elif is_llama:
                    answer = response_body.get("generation", "")
                
                return {{"answer": answer, "question": question}}
                
            except Exception as e:
                if "ThrottlingException" in str(e):
                    print(f"Rate limit exceeded, retrying with backoff: {{str(e)}}")
                    raise  # リトライロジックによって処理される
                else:
                    raise  # その他のエラーは上位で処理
                
        except Exception as e:
            print(f"Prediction error: {{str(e)}}")
            return {{"answer": f"Error: {{str(e)}}", "question": question}}
'''