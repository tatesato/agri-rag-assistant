# 検索精度評価（オフライン）

順位付きの検索結果と正解文書IDを JSONL で照合し、**Recall@K / MRR@K / Hit@K** を計算します。

```bash
python scripts/evaluate_retrieval.py --input evaluation/example.jsonl --k 3
```

各行には `question`、`relevant_ids`（正解ID一覧）、`retrieved_ids`（検索順位順のID一覧）を設定します。

- **Recall@K**: 正解IDのうち上位K件に入った割合
- **MRR@K**: 最初の正解の順位の逆数を質問間で平均
- **Hit@K**: 上位K件に正解が1件でも含まれる質問の割合

重要：このスクリプトは**保存済みの検索順位を集計するツール**です。現時点では既存FAISS Retrieverを自動実行しません。付属の `example.jsonl` は架空の順位を含む動作確認用データで、実際のRAGの精度を示すものではありません。

実データで評価するには、農業QAの質問と正解文書を用意し、Retrieverの検索結果を `retrieved_ids` として保存してください。正解・検索結果のID粒度は文書単位またはチャンク単位で揃えます。比較時は同じ評価セットとKを使用します。
