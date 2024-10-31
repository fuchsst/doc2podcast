# Doc2Podcast

Convert documents into engaging podcasts using AI.

## Features

- Convert PDF and DOCX documents into natural-sounding podcasts
- AI-powered content analysis and script generation
- High-quality text-to-speech using F5-TTS and GLM-4-Voice
- Interactive Streamlit interface
- Efficient document processing and caching

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/doc2podcast.git
cd doc2podcast
```

2. Install dependencies:
```bash
pip install -e .
```

3. Copy the example environment file and fill in your API keys:
```bash
cp .env.example .env
```

## Usage

Run the application using either:

```bash
# Using the installed command
doc2podcast

# Or using the run script
python src/run.py
```

The application will open in your default web browser. Then:

1. Upload your document (PDF or DOCX)
2. Configure voice and style settings
3. Generate your podcast

## Project Structure

```
src/doc2podcast/
├── config/           # Configuration management
├── generators/       # Script and voice generation
├── models/          # Data models
├── processors/      # Document processing
├── ui/              # Streamlit interface
│   ├── components/  # Reusable UI components
│   └── pages/       # Application pages
└── utils/           # Utility functions
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black src/ tests/
isort src/ tests/
```

## License

MIT License - see LICENSE file for details.
