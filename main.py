import weave
import polars as pl
from openai import AsyncOpenAI
from weave import Model
from pydantic import BaseModel
import asyncio
from typing import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class EvaluationResult(BaseModel):
    response_text: str
    score: int | None
    retries: int = 0

class GPTModel(Model):
    predict_model_name: str
    evaluate_model_name: str
    max_retries: int = 5
    initial_retry_delay: float = 1
    max_retry_delay: float = 30
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @weave.op()
    async def predict(self, question: str):
        client = AsyncOpenAI()
        prompt = question
        try:
            response = await client.chat.completions.create(
                model=self.predict_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    async def evaluate_with_retry(
        self, 
        answer: str, 
        question: str, 
        example_answer: str, 
        marking_scheme: str
    ) -> EvaluationResult:
        retries = 0
        last_error = None
        
        for attempt in range(self.max_retries):
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
                    score = min(5, max(1, int(response_text.split("【評点】")[1].lstrip()[0])))
                    return EvaluationResult(
                        response_text=response_text,
                        score=score,
                        retries=retries
                    )
                except Exception as e:
                    print(f"Score parsing error: {str(e)}")
                    last_error = e
                    
            except Exception as e:
                print(f"Evaluation error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                last_error = e
            
            retries += 1
            if attempt < self.max_retries - 1:
                delay = min(
                    self.max_retry_delay,
                    self.initial_retry_delay * (2 ** attempt)
                )
                await asyncio.sleep(delay)
        
        # すべてのリトライが失敗した場合
        return EvaluationResult(
            response_text=f"Evaluation failed after {retries} attempts. Last error: {str(last_error)}",
            score=None,
            retries=retries
        )

    @weave.op()
    async def evaluate(self, answer: str, question: str, example_answer: str, marking_scheme: str):
        return await self.evaluate_with_retry(answer, question, example_answer, marking_scheme)

    async def predict_all(self, questions: List[str]):
        tasks = [self.predict(question) for question in questions]
        return await asyncio.gather(*tasks)

    async def evaluate_all(self, answers: List[str], rows: List[dict]):
        semaphore = asyncio.Semaphore(10)
        
        async def evaluate_with_semaphore(answer, row):
            async with semaphore:
                return await self.evaluate(
                    answer=answer,
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
    weave.init("karakuri-bench/weave-test")

    dataset = weave.ref('karakuri-bench-dataset:latest').get()

    model = GPTModel(
        predict_model_name='gpt-4o-mini-2024-07-18',
        evaluate_model_name='gpt-4o-mini-2024-07-18',
    )

    results = model.execute(dataset)