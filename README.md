---
license: apache-2.0
configs:
- config_name: default
  data_files: "/data/tasks/questions.csv"
---

# karakuri-bench-v0.1

## Usage

### 準備

```sh
export OPENAI_API_KEY="<YOUR_API_KEY>"
```

### 回答生成

```sh
python src/generation/gpt4.py data/tasks/questions.csv data/answers/gpt4.csv
# python src/generation/gpt35.py data/tasks/questions.csv data/answers/gpt35.csv
```

### 生成された回答の評価

```sh
python src/evaluation/main.py data/tasks/questions.csv data/answers/gpt4.csv data/evaluation/gpt4.csv
# python src/evaluation/main.py data/tasks/questions.csv data/answers/gpt35.csv data/evaluation/gpt35.csv
```

