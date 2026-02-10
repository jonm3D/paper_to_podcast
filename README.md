# Hearsay

Convert academic papers from your Zotero library into audio reviews for PhD qualifying exam preparation.

## Overview

Hearsay extracts text and figures from PDFs in your Zotero library, uses Claude to generate a narrated audio summary, then converts it to speech using Kokoro TTS (local, free). The result is a ~10 minute audio review designed to help you deeply understand papers before your qualifying exam.

## Features

- **Zotero Integration**: Directly reads your Zotero SQLite database to browse collections and select papers
- **Intelligent PDF Extraction**: Extracts text and figures using PyMuPDF with extensive cleaning
- **Claude-Powered Processing**:
  - Cleans and structures extracted text into markdown
  - Describes figures using vision capabilities
  - Filters out ads and artifacts from extracted images
  - Generates single-narrator audio summaries written for the ear
- **Narrated Summary**: Direct-to-listener format with natural pacing cues, no fake dialogue
- **PhD Qual Focus**: Prompts are tailored for qualifying exam preparation with:
  - Faithful paper walkthrough (methodology, equations, figures)
  - Critical analysis and limitations
  - Connections to your research
  - Likely committee questions with suggested approaches
- **MP3 Output**: Generates audio files with proper ID3 metadata for music players

## Installation

```bash
# System dependencies (macOS)
brew install espeak-ng ffmpeg

# Clone the repository
git clone git@github.com:jonm3D/hearsay.git
cd hearsay

# Install in development mode
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Optionally set your Zotero data directory (defaults to `~/Zotero`):

```
ZOTERO_DATA_DIR=/path/to/Zotero
```

## Usage

### Command Line

```bash
# Process a paper from a specific collection
hearsay --collection "Qualifying Exam"

# Search across your entire library
hearsay --search "coastal erosion"

# Specify output directory
hearsay --collection "Papers" --output-dir ./podcasts
```

### Interactive Selection

When you run the command, you'll be presented with papers in the collection and can select which one to process.

### Python API

```python
from hearsay.zotero import get_collections, get_papers_in_collection, search_papers
from hearsay.pdf import extract_text_raw, extract_figures
from hearsay.review import process_paper
from hearsay.tts import create_podcast

# List collections
collections = get_collections()

# Get papers from a collection
papers = get_papers_in_collection("My Collection")

# Search library
results = search_papers("machine learning")

# Process a PDF
markdown = process_paper(pdf_path, title)

# Generate podcast
result = create_podcast(markdown, title, output_dir)
```

## Output Structure

```
output/
└── paper_title/
    ├── script.txt            # Generated narration script
    └── Paper_Title.mp3       # Audio file with ID3 metadata
```

The MP3 includes metadata:
- **Title**: Paper title
- **Artist**: "Hearsay"
- **Album**: "PhD Qual Prep"
- **Year**: Generation date
- **Comment**: Generation timestamp

## Project Structure

```
hearsay/
├── pyproject.toml
├── src/
│   └── hearsay/
│       ├── __init__.py
│       ├── cli.py        # CLI entry point
│       ├── zotero.py     # Zotero SQLite integration
│       ├── pdf.py        # PDF text/figure extraction
│       ├── review.py     # Claude API for cleaning/figures
│       └── tts.py        # Script generation & Kokoro TTS
└── output/               # Default output directory
```

## How It Works

1. **Zotero Query**: Connects to your local Zotero SQLite database (in immutable mode to avoid conflicts) and retrieves papers with their PDF paths

2. **PDF Extraction**: Uses PyMuPDF to extract raw text and images, with preprocessing to handle:
   - Multi-column layouts
   - Broken lines and hyphenation
   - Headers, footers, and page numbers
   - Cover pages and institutional headers

3. **Claude Cleaning**: Sends extracted content to Claude API to:
   - Structure text as clean markdown
   - Describe figures using vision capabilities
   - Filter out advertisements and artifacts

4. **Script Generation**: Claude generates a ~10 minute single-narrator summary with:
   - Part 1: Faithful walkthrough of the paper
   - Part 2: Critical analysis and committee prep

5. **Audio Synthesis**: Kokoro TTS converts the script to audio locally (no API key needed):
   - Single narrator voice ("af_heart")
   - Paragraphs synthesized with natural pauses and exported as MP3

## Dependencies

- `click` - CLI framework
- `python-dotenv` - Environment variable loading
- `pymupdf` - PDF text and image extraction
- `anthropic` - Claude API client
- `kokoro` - Kokoro TTS (local text-to-speech, ~82M params)
- `soundfile` - WAV audio I/O
- `pydub` - WAV-to-MP3 conversion (requires ffmpeg)
- `mutagen` - MP3 ID3 metadata

## Requirements

- Python 3.10+
- Zotero with local storage (not cloud-only)
- Anthropic API key
- `espeak-ng` (phonemizer for Kokoro)
- `ffmpeg` (MP3 encoding via pydub)

## Notes

- The tool reads your Zotero database in read-only mode and never modifies it
- Processing a single paper uses approximately 10-20k tokens (Claude); TTS is free and local
- The Kokoro model (~300MB) is auto-downloaded from HuggingFace on first run
- Audio generation runs locally on CPU (Apple Silicon works well) and takes a few minutes for ~10 min episodes
- The podcast prompt is tailored for a geosciences PhD candidate but can be customized in `tts.py`

## License

MIT
