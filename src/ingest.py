"""Read UTF-8 technical documents and create overlapping character chunks."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A document excerpt with its relative source path and ordinal."""

    text: str
    source: str
    chunk_id: int


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into bounded windows without losing non-whitespace content."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def load_documents(directory: Path, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Recursively load .txt/.md files; reject missing or empty corpora."""
    chunk_text("", size, overlap)
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Document directory does not exist.")
    chunks = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read UTF-8 document: {path.name}") from exc
        chunks.extend(
            Chunk(part, path.relative_to(directory).as_posix(), ordinal)
            for ordinal, part in enumerate(chunk_text(text, size, overlap))
        )
    if not chunks:
        raise ValueError("No non-empty .txt or .md documents found.")
    return chunks
