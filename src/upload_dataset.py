import weave
from weave import Dataset
import polars as pl

weave.init("karakuri-bench/weave-test")

df = pl.read_csv('data/tasks/questions.csv')
rows = df.to_dicts()
dataset = Dataset(
    name='karakuri-bench-dataset',
    rows=rows
)
weave.publish(dataset)

prompts = [
    {
        "system_prompt": """
あなたは採点者です。【問題】【正解例】【採点基準】【回答】が与えられるので、以下のフォーマットに従って回答を評価してください。

# 評価フォーマット
```
【採点基準に沿った回答の評価】（自由記述）
【評点】（1以上5以下の整数）
```
""",
        "user_prompt": """
# 問題
{question}

# 正解例
{example_answer}

# 採点基準
{marking_scheme}

# 回答
{answer}
""",
    },
]

prompt_dataset = Dataset(
    name="evaluate_prompt",
    rows=prompts,
)

weave.publish(prompt_dataset)
