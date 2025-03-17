import weave
from weave import Dataset
import polars as pl
from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT = 'karakuri-bench/karakuri-bench'
DATASET_REF = 'karakuri-bench-dataset:latest'
EVALUATE_PROMPT_REF = 'evaluate_prompt:latest'

def upload_dataset():
    """
    CSVファイルからデータセットを読み込み、weaveプラットフォームに公開します
    
    data/questions.csv ファイルを読み込み、データセットとして公開します。
    また、評価用のプロンプトも公開します。
    """
    df = pl.read_csv('data/questions.csv')
    rows = df.to_dicts()
    dataset = Dataset(rows=rows)

    evaluation_prompt = weave.MessagesPrompt([
        {
            "role": "system",
            "content": """あなたは採点者です。【問題】【正解例】【採点基準】【回答】が与えられるので、以下のフォーマットに従って回答を評価してください。

# 評価フォーマット
```
【採点基準に沿った回答の評価】（自由記述）
【評点】（1以上5以下の整数）
```
"""
        },
        {
            "role": "user",
            "content": """
# 問題
{question}

# 正解例
{example_answer}

# 採点基準
{marking_scheme}

# 回答
{answer}
"""
        }
    ])

    weave.publish(dataset, name=DATASET_REF.split(':')[0])
    weave.publish(evaluation_prompt, name=EVALUATE_PROMPT_REF.split(':')[0])

if __name__ == "__main__":
    weave.init(PROJECT)
    upload_dataset()