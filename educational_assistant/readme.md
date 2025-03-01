# Educational Assistant

An AI-powered application for educators to generate Individualized Education Programs (IEPs) and lesson plans using RAG (Retrieval-Augmented Generation) technology.

## Features

- **Document Processing**: Upload and process various educational document formats (PDF, DOCX, TXT, etc.)
- **IEP Generation**: Generate comprehensive IEPs based on uploaded documents
- **Lesson Planning**: Create detailed lesson plans that incorporate IEP accommodations
- **Document Chat**: Ask questions about your educational documents
- **Analytics**: Visualize educational data and track progress

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/educational-assistant.git
   cd educational-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file to add your OpenAI API key
   ```

### Usage

1. Run the application:
   ```bash
   python main.py
   ```
   
   Or use the installed command:
   ```bash
   educational_assistant
   ```

2. Access the web interface at `http://localhost:8501`

3. Upload educational documents

4. Generate IEPs, lesson plans, or chat with your documents

## Development

### Project Structure

The project follows a modular architecture:

- `config/`: Configuration management
- `core/`: Core functionality
  - `document_processing/`: Document handling
  - `embeddings/`: Vector embeddings and storage
  - `llm/`: LLM client integration
  - `rag/`: Retrieval augmented generation
  - `pipelines/`: Processing pipelines
- `ui/`: Streamlit UI components
- `utils/`: Utility functions
- `tests/`: Test suite

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m document_processing
pytest -m embedding
pytest -m vector_store
pytest -m llm
pytest -m rag
pytest -m pipeline
pytest -m ui
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
