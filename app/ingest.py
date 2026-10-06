"""Chargement des documents et découpage en chunks."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    source: str  # nom du fichier d'origine
    text: str


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Découpe un texte en morceaux d'environ `size` caractères avec chevauchement.

    Le chevauchement évite de couper une information importante entre deux chunks.
    On coupe sur un espace pour ne pas casser les mots.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Il faut size > 0 et 0 <= overlap < size")
    text = " ".join(text.split())  # normalise les espaces et retours à la ligne
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # garantit qu'on avance toujours
    return chunks


def load_chunks(docs_dir: Path, size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Lit tous les fichiers .md et .txt d'un dossier et les découpe."""
    docs_dir = Path(docs_dir)
    files = sorted([*docs_dir.glob("*.md"), *docs_dir.glob("*.txt")])
    if not files:
        raise FileNotFoundError(f"Aucun document .md/.txt trouvé dans {docs_dir}")
    chunks: list[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks += [Chunk(path.name, piece) for piece in chunk_text(text, size, overlap)]
    return chunks
