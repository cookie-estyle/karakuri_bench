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

load_dotenv(override=True)

# weave
PROJECT = 'karakuri-bench/weave-test2'
DATASET_REF = 'karakuri-bench-dataset:latest'
EVALUATE_PROMPT_REF = 'evaluate_prompt:latest'

# model
API_TYPE = 'openai' # 'openai', 'bedrock', 'google'
PREDICT_MODEL_NAME = 'gpt-4o-mini-2024-07-18'
EVALUATE_MODEL_NAME = 'gpt-4o-2024-11-20'

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
    print(f"Evaluating {index}/{len(dataset)}: {category}")

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

weave.init(PROJECT)

class_name = PREDICT_MODEL_NAME.replace('-', '_').replace('.', '_').replace(':','_')
model_template = ModelTemplate.get_template(API_TYPE, PREDICT_MODEL_NAME, class_name)
exec(model_template)

dataset = weave.ref(DATASET_REF).get()
dataset = [{key: value for key, value in row.items()} for row in dataset.rows]

prompt_dataset = weave.ref(EVALUATE_PROMPT_REF).get()
SYSTEM_PROMPT = prompt_dataset.messages[0].get('content')
USER_PROMPT = prompt_dataset.messages[1].get('content')

model = eval(f"{class_name}")(predict_model_name=PREDICT_MODEL_NAME)
evaluation = Evaluation(dataset=dataset, scorers=[evaluate])
asyncio.run(evaluation.evaluate(model))