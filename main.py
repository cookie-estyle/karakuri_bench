import weave
import re
import tomli
from weave import Model, Evaluation
from openai import OpenAI
import asyncio
from pydantic import BaseModel
from predictor import ModelTemplate
from pydantic import BaseModel, PrivateAttr
from dotenv import load_dotenv
import os

load_dotenv(override=True)
with open("config.toml", "rb") as f:
    config = tomli.load(f)

class EvaluationResult(BaseModel):
    response_text: str
    category: str
    score: int | None
    retries: int = 0

class_name = config['predict_model_name'].replace('-', '_').replace('.', '_').replace(':','_')
model_template = ModelTemplate.get_template(
    config['api_type'],
    config['predict_model_name'],
    class_name
)
exec(model_template)

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

    for attempt in range(config['max_retries']):
        try:
            try:
                response = client.chat.completions.create(
                    model=config['evaluate_model_name'],
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
            print(f"Evaluation error (attempt {attempt + 1}/{config['max_retries']}): {str(e)}")
            last_error = e

        retries += 1
        if attempt < config['max_retries'] - 1:
            delay = min(
                config['max_retry_delay'],
                config['initial_retry_delay'] * (2 ** attempt)
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

weave.init(config['project'])

dataset = weave.ref(config['dataset_ref']).get()

prompt_dataset = weave.ref(config['evaluate_prompt_ref']).get()
SYSTEM_PROMPT = prompt_dataset.messages[0].get('content')
USER_PROMPT = prompt_dataset.messages[1].get('content')

model = eval(f"{class_name}")(predict_model_name=config['predict_model_name'])
evaluation = Evaluation(dataset=dataset, scorers=[evaluate])
asyncio.run(evaluation.evaluate(model))