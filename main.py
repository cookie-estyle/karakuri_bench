import weave
import re
from weave import Model
from openai import OpenAI
import asyncio
from pydantic import BaseModel

PREDICT_MODEL_NAME='gpt-4o-mini-2024-07-18'
EVALUATE_MODEL_NAME = 'gpt-4o-2024-11-20'
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 30

SYSTEM_PROMPT = """
あなたは採点者です。【問題】【正解例】【採点基準】【回答】が与えられるので、以下のフォーマットに従って回答を評価してください。

# 評価フォーマット
```
【採点基準に沿った回答の評価】（自由記述）
【評点】（1以上5以下の整数）
```
"""

USER_PROMPT = """
# 問題
{question}

# 正解例
{example_answer}

# 採点基準
{marking_scheme}

# 回答
{answer}
"""

class EvaluationResult(BaseModel):
    response_text: str
    category: str
    score: int | None
    retries: int = 0

class GPTModel(Model):
    predict_model_name: str

    @weave.op()
    def predict(self, question: str) -> dict:
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.predict_model_name,
                messages=[
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                response_format={"type": "text"},
            )
            answer = response.choices[0].message.content
            return {'answer': answer, 'question': question}
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

@weave.op()
async def evaluate(
    output,
    index=None, 
    category=None, 
    question=None,
    example_answer=None,
    marking_scheme=None,
) -> EvaluationResult:

    retries = 0
    last_error = None
    client = OpenAI()

    user_prompt = USER_PROMPT.format(
        question = question,
        example_answer = example_answer,
        marking_scheme = marking_scheme,
        answer = output["answer"],
    ).strip()

    for attempt in range(MAX_RETRIES):
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
            print(f"Evaluation error (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            last_error = e

        retries += 1
        if attempt < MAX_RETRIES - 1:
            delay = min(
                MAX_RETRY_DELAY,
                INITIAL_RETRY_DELAY * (2 ** attempt)
            )
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

weave.init("karakuri-bench/weave-test")

dataset = weave.ref('karakuri-bench-dataset:latest').get()
dataset = [{key: value for key, value in row.items()} for row in dataset.rows]
model = GPTModel(predict_model_name=PREDICT_MODEL_NAME)
evaluation = weave.Evaluation(dataset=dataset, scorers=[evaluate])
asyncio.run(evaluation.evaluate(model))