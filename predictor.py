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
            },
            "google": {
                "standard": cls._get_google_template
            },
            "vllm": {
                "standard": cls._get_vllm_template
            }
        }
        
        if api_type == "bedrock":
            template_type = "standard"
        elif api_type == "google":
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
    def _get_google_template(class_name: str) -> str:
        return f'''
class {class_name}(Model):
    predict_model_name: str
    _last_request_time: float = PrivateAttr(default=0)
    _min_request_interval: float = PrivateAttr(default=5.0)  # 最小リクエスト間隔（秒）

    def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            sleep_time += random.uniform(0, 0.5)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    @weave.op()
    def predict(self, question: str) -> dict:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        import os

        max_retries = 5
        base_delay = 4
        max_delay = 60
        
        for attempt in range(max_retries):
            try:
                # レート制限のための待機
                self._wait_for_rate_limit()
                
                # 安全性設定の構成
                categories = [
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    HarmCategory.HARM_CATEGORY_HARASSMENT,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                ]
                safety_settings = {{cat: HarmBlockThreshold.BLOCK_NONE for cat in categories}}

                # モデルの初期化
                llm = ChatGoogleGenerativeAI(
                    model=self.predict_model_name,
                    api_key=os.environ["GOOGLE_API_KEY"],
                    safety_settings=safety_settings,
                    temperature=0.0,
                )

                # 予測の実行
                response = llm.invoke(question)
                answer = response.content
                
                return {{'answer': answer, 'question': question}}
                
            except Exception as e:
                print(f"Attempt {{attempt+1}}/{{max_retries}} failed: {{str(e)}}")
                if attempt < max_retries - 1:  # 最後の試行でなければリトライ
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    print(f"Retrying in {{delay}} seconds...")
                    time.sleep(delay)
                else:
                    print("All retry attempts failed")
                    return {{'answer': f"Error after {{max_retries}} attempts: {{str(e)}}", 'question': question}}
'''

    @staticmethod
    def _get_bedrock_template(class_name: str) -> str:
        return f'''
class {class_name}(Model):
    predict_model_name: str
    _bedrock_runtime: object = PrivateAttr(default=None)
    _generator_config: dict = PrivateAttr(default={{"temperature": 0.0, "top_p": 1.0}})
    _last_request_time: float = PrivateAttr(default=0)
    _min_request_interval: float = PrivateAttr(default=1.0)  # 最小リクエスト間隔（秒）
    
    def __init__(self, **data):
        super().__init__(**data)
        self._bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            config=Config(read_timeout=1000),
        )

    def _format_llama_prompt(self, messages):
        formatted_prompt = "<|begin_of_text|>"
        
        # システムメッセージがあれば最初に追加
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        if system_message:
            formatted_prompt += f"<|start_header_id|>system<|end_header_id|>\\n{{system_message['content']}}<|eot_id|>"
        
        # ユーザーとアシスタントのメッセージを追加
        for message in messages:
            if message["role"] == "system":
                continue  # システムメッセージは既に処理済み
            
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\\n{{content}}<|eot_id|>"
            elif role == "assistant":
                formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\\n{{content}}<|eot_id|>"
        
        # 最後にアシスタントのヘッダーを追加して応答を促す
        if not messages or messages[-1]["role"] != "assistant":
            formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>"
            
        return formatted_prompt

    def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            sleep_time += random.uniform(0, 0.1)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

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
            # モデルタイプの判定
            model_id = self.predict_model_name.lower()
            
            if "amazon.nova" in model_id:
                body_dict = {{
                    "messages": [
                        {{
                            "role": "user",
                            "content": [
                                {{
                                    "text": question
                                }}
                            ]
                        }}
                    ]
                }}
            elif "anthropic" in model_id:
                body_dict = {{
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [
                        {{
                            "role": "user",
                            "content": [
                                {{
                                    "type": "text",
                                    "text": question
                                }}
                            ]
                        }}
                    ],
                    **self._generator_config,
                }}
            elif "llama" in model_id:
                prompt = self._format_llama_prompt([{{"role": "user", "content": question}}])
                body_dict = {{
                    "prompt": prompt,
                    "max_gen_len": 1024,
                    **self._generator_config,
                }}
            elif "amazon.titan" in model_id:
                body_dict = {{
                    "inputText": question,
                    "textGenerationConfig": {{
                        "maxTokenCount": 1024,
                        **self._generator_config,
                    }}
                }}
            elif "ai21" in model_id:
                body_dict = {{
                    "prompt": question,
                    "maxTokens": 1024,
                    **self._generator_config,
                }}
            else:
                raise ValueError(f"Unsupported model: {{self.predict_model_name}}")

            try:
                response = self._invoke_model(body_dict)
                response_body = json.loads(response.get("body").read())
                
                if "amazon.nova" in model_id:
                    answer = response_body.get("output", {{}}).get("message", {{}}).get("content", [{{}}])[0].get("text", "")
                elif "anthropic" in model_id:
                    answer = response_body.get("content")[0].get("text", "")
                elif "llama" in model_id:
                    answer = response_body.get("generation", "")
                elif "amazon.titan" in model_id:
                    answer = response_body.get("results", [{{}}])[0].get("outputText", "")
                elif "ai21" in model_id:
                    answer = response_body.get("completions", [{{}}])[0].get("data", {{}}).get("text", "")
                else:
                    answer = "Unsupported model response format"
                
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

    @staticmethod
    def _get_vllm_template(class_name: str) -> str:
        return f'''
class {class_name}(Model):
    predict_model_name: str

    @weave.op()
    def predict(self, question: str) -> dict:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )
        try:
            response = client.chat.completions.create(
                model=self.predict_model_name,
                messages=[{{"role": "user", "content": question}}],
                temperature=0.0,
                max_tokens=1024,
                response_format={{"type": "text"}},
            )
            answer = response.choices[0].message.content
            return {{'answer': answer, 'question': question}}
        except Exception as e:
            print(f"Prediction error: {{str(e)}}")
            raise
'''