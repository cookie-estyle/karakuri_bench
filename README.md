# karakuri-bench

karakuri-benchは、大規模言語モデル（LLM）の評価フレームワークです。様々なLLMの性能を評価し、比較するためのツールセットを提供します。

## 機能

- 複数のAPIタイプ（OpenAI、Bedrock、Google、DeepSeek、vLLM）に対応したモデル評価
- 質問応答、日本語処理、ビジネス文書作成など多様なカテゴリでの評価
- 評価結果の集計と分析

## 前提条件

- Python 3.8以上
- CUDA対応のGPU（vLLMモデル評価時）
- 各種APIキー（使用するモデルに応じて）

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-username/karakuri-bench.git
cd karakuri-bench

# 仮想環境の作成と有効化（推奨）
python -m venv .venv
source .venv/bin/activate  # Linuxの場合
# または
.\.venv\Scripts\activate  # Windowsの場合

# 依存パッケージのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .envファイルを編集して必要なAPIキーを設定
```

## 環境変数の設定

`.env`ファイルに以下の環境変数を設定します：

- `WANDB_API_KEY`: WandB APIキー（評価結果の記録用）
- `GOOGLE_API_KEY`: Google APIキー（Google系モデル評価用）
- `OPENAI_API_KEY`: OpenAI APIキー（OpenAI系モデル評価用）
- `HUGGING_FACE_HUB_TOKEN`: Hugging Face Hub トークン（モデルダウンロード用）
- `DEEPSEEK_API_KEY`: DeepSeek APIキー（DeepSeek系モデル評価用）
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`: AWS認証情報（Bedrock系モデル評価用）
- `WEAVE_PARALLELISM`: 並列処理数（デフォルト: 10）

## 詳細な使用方法

### 1. データセットのアップロード

まず、評価用のデータセットをWandBプラットフォームにアップロードします：

```bash
python upload_dataset.py
```

このスクリプトは`data/questions.csv`ファイルを読み込み、WandBプラットフォームにデータセットとして公開します。また、評価用のプロンプトも公開します。

### 2. 評価対象モデルの設定

`models.py`ファイルを編集して、評価対象のモデルを設定します。各モデルはAPIタイプとモデル名のペアで指定します：

```python
MODELS_TO_EVALUATE = [
    {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06'},
    {'api_type': 'vllm', 'model_name': 'microsoft/Phi-4-mini-instruct'},
    {'api_type': 'bedrock', 'model_name': 'anthropic.claude-3-sonnet-20240229-v1:0'},
    # 他のモデルを追加...
]
```

評価したいモデルのコメントアウトを外すか、新しいモデルを追加します。

### 3. モデルの評価実行

以下のコマンドを実行して評価を開始します：

```bash
python main.py
```

このスクリプトは以下の処理を行います：

1. WandBプラットフォームに接続
2. データセットと評価用プロンプトを取得
3. 設定されたモデルを順番に評価
4. 評価結果をWandBに記録

vLLMモデルの場合は、評価前にローカルでvLLMサーバーを起動し、評価後にサーバーを停止します。

### 4. 特定のモデルのみ評価

特定のモデルのみを評価したい場合は、`models.py`ファイルで該当するモデル以外をコメントアウトします：

```python
MODELS_TO_EVALUATE = [
    # {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06'},  # コメントアウト
    {'api_type': 'vllm', 'model_name': 'microsoft/Phi-4-mini-instruct'},  # 評価対象
    # {'api_type': 'bedrock', 'model_name': 'anthropic.claude-3-sonnet-20240229-v1:0'},  # コメントアウト
]
```

### 5. 評価結果の確認

評価結果はWandBプラットフォーム上で確認できます。ブラウザでWandBにログインし、プロジェクト「karakuri-bench/karakuri-bench」を開きます。

各モデルの評価結果は以下の情報を含みます：
- 全体スコア
- カテゴリ別スコア
- 質問ごとの回答と評価

### 6. カスタムデータセットの使用

独自の評価データセットを使用する場合は、`data/questions.csv`と同じ形式のCSVファイルを作成し、`data`ディレクトリに配置します。CSVファイルは以下の列を含む必要があります：

- `index`: 質問のインデックス
- `category`: 質問のカテゴリ（reader, japanese, writing, business）
- `question`: 質問文
- `example_answer`: 模範回答（N/Aでも可）
- `marking_scheme`: 採点基準

### 7. vLLMモデルの評価

vLLMモデルを評価する場合、以下の点に注意してください：

- 十分なGPUメモリが必要です
- モデルのダウンロードに時間がかかる場合があります
- サーバーの起動と停止は自動的に行われます

vLLMサーバーの設定を変更する場合は、`vllm_server.py`の`start_vllm_server`関数内のパラメータを調整します：

```python
command = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_id,
    "--served-model-name", model_id,
    "--dtype", "float16",  # 精度を変更（float16, bfloat16, float32）
    "--chat-template", chat_template_path,
    "--max-model-len", "3000",  # コンテキスト長を変更
    "--max-num-seqs", "40",  # 並列シーケンス数を変更
    "--tensor-parallel-size", str(available_gpus),  # テンソル並列度を変更
    # その他のパラメータ...
]
```

### 8. トラブルシューティング

#### APIキーエラー
環境変数が正しく設定されているか確認してください。`.env`ファイルが正しく読み込まれていることを確認します。

#### GPUメモリ不足
vLLMモデルを評価する際にGPUメモリ不足エラーが発生した場合は、以下の対策を試してください：
- 小さいモデルを使用する
- `--dtype`を`float16`に設定する
- `--max-model-len`を小さくする
- `--tensor-parallel-size`を増やす（複数GPUがある場合）

#### 評価タイムアウト
評価に時間がかかりすぎる場合は、`main.py`の`MAX_RETRY_COUNT`や`MAX_RETRY_DELAY`の値を調整してください。

## プロジェクト構成の詳細

```
├── .env.example       # 環境変数のテンプレート
├── .gitignore         # Gitの無視ファイル設定
├── README.md          # プロジェクト説明書
├── data               # データセットディレクトリ
│   └── questions.csv  # 評価用質問データ
├── main.py            # メイン実行スクリプト（評価ロジック）
├── models.py          # 評価対象モデル定義
├── predictor.py       # モデル予測用テンプレート（各APIタイプ対応）
├── requirements.txt   # 依存パッケージリスト
├── upload_dataset.py  # データセットアップロードスクリプト
└── vllm_server.py     # vLLMサーバー管理スクリプト
```

## 評価カテゴリと質問例

データセットには以下のカテゴリの質問が含まれています：

### reader（文書読解・質問応答）
例: 「以下の【知識源】の内容をもとに、以下の【問い合わせ】の内容に回答してください。」

### japanese（日本語処理）
例: 「以下の文章の誤字脱字を修正してください」、「与えられた文章中の漢字に読み仮名を括弧書きで振ってください」

### writing（文書作成）
例: 「電子レンジの製品マニュアルの「基本の使い方」のセクションを書いてください」

### business（ビジネス知識）
例: 「カスタマーサクセスとはどのような概念ですか？カスタマーサポートとの違いを含めて答えてください。」

## ライセンス

Apache License 2.0
