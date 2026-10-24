# 実際のFAISS検索の評価

既存の `src.retriever.Retriever` を呼び出し、農業QAの正解資料と照合します。

```bash
python scripts/evaluate_live_retrieval.py \
  --documents data/sample_documents \
  --questions evaluation/agri_questions.example.jsonl \
  --k 5 \
  --output data/index/retrieval_metrics.json
```

`OPENAI_API_KEY` が必要です。索引作成および質問Embedding時にAPI料金が発生します。回答生成用のLLMは呼びません。

`relevant_ids` は `data/sample_documents` からの**相対ファイル名**で記述します。検索されたチャンクの `chunk.source` を重複除去してファイル名で比較します。

**重要な制約**: FAISSが返す上位Kチャンクを取得した後に重複ファイル名をまとめるため、ユニーク文書の上位K件を検索する評価とは異なります。正解データは説明用の少数例であり、研究上の正解付け・精度保証はしていません。精度を比較する際は十分な質問数、同一データセットと同一Kを使ってください。テストはモック検索のみでAPIを呼びません。
