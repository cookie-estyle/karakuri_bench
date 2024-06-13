import argparse

import openai
import polars as pl
from tenacity import retry, stop_after_attempt, wait_random_exponential


# https://beta.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("QUESTIONS_CSV")
    parser.add_argument("ANSWER_CSV")
    args = parser.parse_args()

    df = pl.read_csv(args.QUESTIONS_CSV)
    assert {*df.columns} >= {"index", "question"}

    out_rows = []
    for row in df.iter_rows(named=True):
        res = completion_with_backoff(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": row["question"]}]
        )
        out_rows.append({
            "index": row["index"],
            "answer": res.choices[0].message.content
        })

    out_df = pl.DataFrame(out_rows)
    out_df.write_csv(args.ANSWER_CSV)
