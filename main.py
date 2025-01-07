import weave
from weave import Dataset
import re
import polars as pl
from openai import AsyncOpenAI
from weave import Model
from pydantic import BaseModel
import asyncio
from typing import List
from weave.flow import leaderboard
from weave.trace.weave_client import get_ref
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

@weave.op()
def prepare_dataset(raw_dataset):
    """WeaveのDataset形式にデータを変換する"""
    return weave.Dataset(
        name="evaluation-dataset",
        rows=[{
            "question": row["question"],
            "example_answer": row["example_answer"],
            "marking_scheme": row["marking_scheme"],
            "category": row["category"]
        } for row in raw_dataset.rows]
    )

@weave.op()
async def check_score(question: str, example_answer: str, marking_scheme: str, output: dict):
    """回答のスコアを評価する関数"""
    try:
        result = await model.evaluate_with_retry(
            answer=output["answer"],
            category=output["category"],
            question=question,
            example_answer=example_answer,
            marking_scheme=marking_scheme
        )
        
        original_category = output.get("original_category", output["category"])
        
        return {
            "score": result.score,
            "category": original_category,
            "response_text": result.response_text,
            "category_score": {
                original_category: result.score
            }
        }
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {
            "score": None,
            "category": None,
            "response_text": str(e),
            "category_score": None
        }

@weave.op()
async def gpt_model(question: str, category: str):
    """質問に対する回答を生成する関数"""
    try:
        response = await model.predict(question)
        return {
            "answer": response.choices[0].message.content,
            "category": category
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            "answer": str(e),
            "category": "error"
        }

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 30
SEMAPHORE_LIMIT = 10

class EvaluationResult(BaseModel):
    response_text: str
    category: str
    score: int | None
    retries: int = 0

class GPTModel(Model):
    predict_model_name: str
    evaluate_model_name: str
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @weave.op()
    async def predict(self, question: str):
        client = AsyncOpenAI()
        try:
            response = await client.chat.completions.create(
                model=self.predict_model_name,
                messages=[{"role": "user", "content": question}],
                temperature=0,
            )
            return response
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    async def evaluate_with_retry(
        self, 
        answer: str, 
        category: str, 
        question: str, 
        example_answer: str, 
        marking_scheme: str
    ) -> EvaluationResult:
        retries = 0
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                client = AsyncOpenAI()
                system_prompt = f"""
あなたは採点者です。【問題】【正解例】【採点基準】【回答】が与えられるので、以下のフォーマットに従って回答を評価してください。

# 評価フォーマット
```
【採点基準に沿った回答の評価】（自由記述）
【評点】（1以上5以下の整数）
```
                """.strip()

                prompt = f"""
# 問題
{question}

# 正解例
{example_answer}

# 採点基準
{marking_scheme}

# 回答
{answer}
                """.strip()

                response = await client.chat.completions.create(
                    model=self.evaluate_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                
                response_text = response.choices[0].message.content
                try:
                    score_text = response_text.split("【評点】")[1].strip()
                    score_match = re.search(r'[1-5]', score_text)
                    if score_match:
                        score = int(score_match.group())
                    else:
                        raise ValueError("No valid score found")
                    
                    return EvaluationResult(
                        response_text=response_text,
                        category=category,
                        score=score,
                        retries=retries
                    )
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
        
        return EvaluationResult(
            response_text=f"Evaluation failed after {retries} attempts. Last error: {str(last_error)}",
            category=category,
            score=None,
            retries=retries
        )

    @weave.op()
    async def evaluate(self, answer: str, category: str, question: str, example_answer: str, marking_scheme: str):
        return await self.evaluate_with_retry(answer, category, question, example_answer, marking_scheme)

    async def predict_all(self, questions: List[str]):
        tasks = [self.predict(question) for question in questions]
        return await asyncio.gather(*tasks)

    async def evaluate_all(self, answers: List[str], rows: List[dict]):
        semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
        
        async def evaluate_with_semaphore(answer, row):
            async with semaphore:
                return await self.evaluate(
                    answer=answer,
                    category=row["category"],
                    question=row["question"],
                    example_answer=row["example_answer"],
                    marking_scheme=row["marking_scheme"]
                )
        
        tasks = [
            evaluate_with_semaphore(answer, row)
            for answer, row in zip(answers, rows)
        ]
        return await asyncio.gather(*tasks)

    async def execute_async(self, dataset):
        answers = await self.predict_all(dataset["question"].to_list())
        results = await self.evaluate_all(answers, dataset.to_dicts())
        return results

    def execute(self, dataset):
        data = [{key: value for key, value in row.items()} for row in dataset.rows]
        df = pl.DataFrame(data)
        return asyncio.run(self.execute_async(df))

if __name__ == "__main__":
    async def main():
        weave.init("karakuri-bench/weave-test")
        raw_dataset = weave.ref('karakuri-bench-dataset:latest').get()
        
        dataset = prepare_dataset(raw_dataset)
        
        global model
        model = GPTModel(
            predict_model_name='gpt-4o-mini-2024-07-18',
            evaluate_model_name='gpt-4o-mini-2024-07-18',
        )
        
        evaluation = weave.Evaluation(
            name="GPT Model Evaluation",
            dataset=dataset,
            scorers=[check_score]
        )
        
        await evaluation.evaluate(gpt_model)
        
        spec = leaderboard.Leaderboard(
            name="GPT Model Performance",
            description="""
            This leaderboard shows the performance of GPT models on various tasks.
            
            ### Metrics
            1. Overall Score: Average score across all evaluations
            2. Category Scores: Average scores by category
            """,
            columns=[
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(evaluation).uri(),
                    scorer_name="check_score",
                    summary_metric_path="score.mean",
                    name="Overall Score"
                )
            ]
        )
        
        ref = weave.publish(spec)

    asyncio.run(main())