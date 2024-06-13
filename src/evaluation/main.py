import argparse

import openai
import polars as pl
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pydantic import BaseModel


# https://beta.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


class EvaluationResult(BaseModel):
    response_text: str
    score: int | None


def evaluate_answer(
    question: str,
    example_answer: str,
    marking_scheme: str,
    answer: str
) -> EvaluationResult:
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
    response = completion_with_backoff(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response_text = response.choices[0].message.content
    try:
        score = min(5, max(1, int(response_text.split("【評点】")[1].lstrip()[0])))
    except Exception:
        score = None

    return EvaluationResult(
        response_text=response_text,
        score=score
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("QUESTIONS_CSV")
    parser.add_argument("ANSWER_CSV")
    parser.add_argument("EVALUATION_CSV")
    args = parser.parse_args()

    df_questions = pl.read_csv(args.QUESTIONS_CSV)
    assert {*df_questions.columns} >= {"index", "question", "example_answer", "marking_scheme"}

    df_answers = pl.read_csv(args.ANSWER_CSV)
    assert {*df_answers.columns} >= {"index", "answer"}

    df = df_questions.join(df_answers, on="index", how="inner", validate="1:1")

    out_rows = []
    for row in df.iter_rows(named=True):
        evaluation_result = evaluate_answer(
            row["question"],
            row["example_answer"],
            row["marking_scheme"],
            row["answer"]
        )
        out_rows.append({
            "index": row["index"],
            "response": evaluation_result.response_text,
            "score": evaluation_result.score,
        })
    
    out_df = pl.DataFrame(out_rows)
    out_df.write_csv(args.EVALUATION_CSV)
