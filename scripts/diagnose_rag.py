"""Run the real pipeline stage by stage; never print credentials."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np

from src.diagnostics import safe_traceback
from src.ingest import load_documents
from src.llm import create_client, OpenAIEmbedder, OpenAIAnswerer
from src.retriever import Retriever, _vectors
from src.rag import RAGAssistant


def main():
    stage = ".env / client"
    try:
        print(stage, ROOT / '.env', flush=True)
        client = create_client()
        print('API key configured:', bool(client.api_key), flush=True)
        stage = "UTF-8 documents / chunks"
        directory = ROOT / 'data/sample_documents'
        print(stage, directory, flush=True)
        chunks = load_documents(directory)
        print([(c.source, c.chunk_id, len(c.text)) for c in chunks], flush=True)
        stage = "OpenAI embeddings"
        print(stage, flush=True)
        embedder = OpenAIEmbedder(client, os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'))
        values = embedder.embed([c.text for c in chunks])
        stage = "NumPy / normalization"
        print(stage, flush=True)
        matrix = _vectors(values, len(chunks))
        print('shape:', matrix.shape, 'dtype:', matrix.dtype,
              'contiguous:', matrix.flags.c_contiguous, 'finite:', bool(np.isfinite(matrix).all()), flush=True)
        stage = "FAISS creation"
        print(stage, flush=True)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        retriever = Retriever(embedder, chunks, index)
        stage = "FAISS save / reload"
        target = ROOT / 'data/index'
        print(stage, target, flush=True)
        retriever.save(target)
        restored = Retriever.load(target, embedder)
        print('dimension:', restored.index.d, 'count:', restored.index.ntotal, flush=True)
        stage = "RAG"
        print(stage, flush=True)
        question = 'あきさかりの出穂予測では、気温と日長をどのように利用しますか？'
        response = RAGAssistant(restored, OpenAIAnswerer(client, os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'))).ask(question)
        report = {'question': question, 'answer': response.answer, 'sources': response.sources,
                  'results': [{'source': r.chunk.source, 'chunk_id': r.chunk.chunk_id,
                               'score': r.score, 'text': r.chunk.text} for r in response.results]}
        output = json.dumps(report, ensure_ascii=False, indent=2)
        (target / 'rag_verification.json').write_text(output, encoding='utf-8')
        print(output)
    except Exception as exc:
        print('FAILED STAGE:', stage)
        print(safe_traceback(exc))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
