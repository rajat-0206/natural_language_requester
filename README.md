# API Requester & Semantic API Search

This project provides a smart, interactive command-line tool for matching user queries to API endpoints using semantic search and LLM-based intent extraction. It is designed to help developers and API consumers quickly find the right API and generate request payloads from natural language instructions.

## Features

- **Semantic API Search:**
  - Uses sentence-transformer embeddings to match user queries to the most relevant API endpoint.
  - Supports both t-SNE and PCA visualizations of API and query embeddings.
- **LLM-based Query Parsing:**
  - Uses Mistral (via Ollama) to extract the main action, data/object, and details from user queries.
  - Improves API matching by focusing the search on intent and object.
- **Automated Request Body Generation:**
  - Uses Mistral to generate the correct API request payload, filling in as many details as possible and prompting for missing required fields.
- **Interactive CLI:**
  - Guides the user through query, API selection, and parameter completion.
- **Caching:**
  - Embedding and TF-IDF indices are cached for fast repeated use.

## Setup

1. **Clone the repository** and install dependencies:
   ```bash
   git clone <repo-url>
   cd scripts
   pip install -r requirements.txt
   # or install manually: torch, faiss, numpy, scikit-learn, transformers, matplotlib, requests, pytz
   ```

2. **Start Ollama with Mistral model** (for LLM-based extraction and payload generation):
   - [Ollama installation instructions](https://ollama.com/)
   - Download and run the Mistral model:
     ```bash
     ollama run mistral
     ```

3. **Prepare your API schema:**
   - You **do not need to manually create `schema.json`**.
   - You can **download `schema.yaml` from your API's Swagger/OpenAPI documentation** and use the provided script to convert it:
     ```bash
     python api_requester/convert_schema.py schema.yaml schema.json
     ```
   - This will generate the required `schema.json` in the correct format for this project.

## Usage

### 1. Interactive API Requester
Run:
```bash
python api_requester/api_requester.py
```
- Type your natural language query (e.g., "Get all attendees of the event with id 123").
- The tool will:
  1. Use Mistral to extract action/data/details.
  2. Use semantic search to find the best API endpoint.
  3. Use Mistral to generate the request payload.
  4. Prompt you for any missing required fields.

### 2. Visualize API Embeddings
Run:
```bash
python api_requester/get_top_apis.py --visualize [--pca]
```
- Shows a 2D plot of all API endpoints and your query.
- Top search results are highlighted in red, the query in green.
- Use `--pca` to use PCA instead of t-SNE for dimensionality reduction.

### 3. Batch/Scripted Use
- You can import the core functions from `get_top_apis.py` or `api_requester.py` in your own scripts.

## File Overview
- `api_requester/api_requester.py` — Main interactive CLI tool.
- `api_requester/get_top_apis.py` — Standalone script for API search and embedding visualization.
- `schema.json` — Your API schema (OpenAPI-like format, can be generated from schema.yaml).
- `.cache/` — Stores cached embedding and TF-IDF indices for fast search.
- `api_requester/convert_schema.py` — Script to convert Swagger/OpenAPI YAML to schema.json.

## Requirements
- Python 3.8+
- torch, faiss, numpy, scikit-learn, transformers, matplotlib, requests, pytz
- Ollama running Mistral model (for LLM-based features)

## Example
```
You: Get all attendees of the event with id 123
Improved user query is {'action': 'get', 'data': 'attendees', 'details': 'of the event with id 123'}
Matched API:  { ... }
(Search text used for embedding:) get attendees
➡️ Action: Get all attendees
📦 API: /event/{id}/attendees
📦 Method: GET
📦 Payload: { ... }
```

## License
MIT 