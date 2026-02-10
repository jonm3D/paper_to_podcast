"""Zotero SQLite database integration."""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paper:
    """Represents a paper from Zotero with its PDF path."""
    item_id: int
    title: str
    pdf_path: Path | None


def get_zotero_dir() -> Path:
    """Get the Zotero data directory from env var or default location."""
    if zotero_dir := os.environ.get("ZOTERO_DATA_DIR"):
        return Path(zotero_dir)
    return Path.home() / "Zotero"


def get_db_path(zotero_dir: Path | None = None) -> Path:
    """Get the path to the Zotero SQLite database."""
    if zotero_dir is None:
        zotero_dir = get_zotero_dir()
    return zotero_dir / "zotero.sqlite"


def get_storage_dir(zotero_dir: Path | None = None) -> Path:
    """Get the path to Zotero's storage directory for attachments."""
    if zotero_dir is None:
        zotero_dir = get_zotero_dir()
    return zotero_dir / "storage"


def get_collections(zotero_dir: Path | None = None) -> list[str]:
    """Get all collection names from the Zotero library.

    Args:
        zotero_dir: Path to Zotero data directory. Uses default if not specified.

    Returns:
        List of collection names.
    """
    db_path = get_db_path(zotero_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Zotero database not found at {db_path}")

    # Use immutable mode to avoid locking conflicts when Zotero is running
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    try:
        cursor = conn.execute("SELECT collectionName FROM collections ORDER BY collectionName")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_papers_in_collection(collection_name: str, zotero_dir: Path | None = None) -> list[Paper]:
    """Get all papers with their PDF paths from a collection.

    Args:
        collection_name: Name of the Zotero collection.
        zotero_dir: Path to Zotero data directory. Uses default if not specified.

    Returns:
        List of Paper objects with titles and PDF paths.
    """
    db_path = get_db_path(zotero_dir)
    storage_dir = get_storage_dir(zotero_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Zotero database not found at {db_path}")

    # Use immutable mode to avoid locking conflicts when Zotero is running
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    try:
        # Get collection ID
        cursor = conn.execute(
            "SELECT collectionID FROM collections WHERE collectionName = ?",
            (collection_name,)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Collection '{collection_name}' not found")
        collection_id = row[0]

        # Get items in collection with their titles
        # itemTypeID 2 = 'attachment', we want parent items
        # fieldID 1 = 'title' field
        query = """
            SELECT DISTINCT i.itemID, idv.value as title
            FROM collectionItems ci
            JOIN items i ON ci.itemID = i.itemID
            LEFT JOIN itemData id ON i.itemID = id.itemID AND id.fieldID = 1
            LEFT JOIN itemDataValues idv ON id.valueID = idv.valueID
            WHERE ci.collectionID = ?
              AND i.itemTypeID != 2
            ORDER BY title
        """
        cursor = conn.execute(query, (collection_id,))
        items = cursor.fetchall()

        papers = []
        for item_id, title in items:
            # Find PDF attachment for this item
            pdf_path = _find_pdf_for_item(conn, item_id, storage_dir)
            papers.append(Paper(
                item_id=item_id,
                title=title or "(No title)",
                pdf_path=pdf_path
            ))

        return papers
    finally:
        conn.close()


def _find_pdf_for_item(conn: sqlite3.Connection, item_id: int, storage_dir: Path) -> Path | None:
    """Find the PDF attachment path for an item.

    Args:
        conn: SQLite connection.
        item_id: The parent item ID.
        storage_dir: Path to Zotero storage directory.

    Returns:
        Path to PDF file, or None if not found.
    """
    # Get attachments for this item
    # The 'key' column is in the items table, not itemAttachments
    # contentType = 'application/pdf' for PDF files
    query = """
        SELECT ia.path, i.key
        FROM itemAttachments ia
        JOIN items i ON ia.itemID = i.itemID
        WHERE ia.parentItemID = ?
          AND ia.contentType = 'application/pdf'
        LIMIT 1
    """
    cursor = conn.execute(query, (item_id,))
    row = cursor.fetchone()

    if not row:
        return None

    path, key = row

    # Zotero stores PDFs in storage/<key>/<filename>
    # The path column has format "storage:filename.pdf"
    if key and path and path.startswith("storage:"):
        filename = path[8:]  # Remove 'storage:' prefix
        pdf_path = storage_dir / key / filename
        if pdf_path.exists():
            return pdf_path

    # Fallback: check the directory for any PDF
    if key:
        attachment_dir = storage_dir / key
        if attachment_dir.exists():
            pdfs = list(attachment_dir.glob("*.pdf"))
            if pdfs:
                return pdfs[0]

    # Fallback: linked files (path without storage: prefix)
    if path and not path.startswith("storage:"):
        pdf_path = Path(path)
        if pdf_path.exists():
            return pdf_path

    return None


def search_papers(query: str, zotero_dir: Path | None = None) -> list[Paper]:
    """Search for papers by title across the entire library.

    Args:
        query: Search string to match against titles (case-insensitive).
        zotero_dir: Path to Zotero data directory. Uses default if not specified.

    Returns:
        List of Paper objects matching the query.
    """
    db_path = get_db_path(zotero_dir)
    storage_dir = get_storage_dir(zotero_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Zotero database not found at {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    try:
        # Search all items by title
        # fieldID 1 = 'title' field
        sql = """
            SELECT DISTINCT i.itemID, idv.value as title
            FROM items i
            LEFT JOIN itemData id ON i.itemID = id.itemID AND id.fieldID = 1
            LEFT JOIN itemDataValues idv ON id.valueID = idv.valueID
            WHERE i.itemTypeID != 2
              AND idv.value LIKE ?
            ORDER BY title
        """
        cursor = conn.execute(sql, (f"%{query}%",))
        items = cursor.fetchall()

        papers = []
        for item_id, title in items:
            pdf_path = _find_pdf_for_item(conn, item_id, storage_dir)
            papers.append(Paper(
                item_id=item_id,
                title=title or "(No title)",
                pdf_path=pdf_path
            ))

        return papers
    finally:
        conn.close()


# Quick test when run directly
if __name__ == "__main__":
    print("Zotero data directory:", get_zotero_dir())
    print("Database path:", get_db_path())
    print()

    try:
        collections = get_collections()
        print(f"Found {len(collections)} collections:")
        for c in collections:
            print(f"  - {c}")

        if collections:
            print()
            first_collection = collections[0]
            print(f"Papers in '{first_collection}':")
            papers = get_papers_in_collection(first_collection)
            for p in papers:
                pdf_status = "PDF" if p.pdf_path else "No PDF"
                print(f"  - [{pdf_status}] {p.title}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
