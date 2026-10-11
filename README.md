Note: this is a proof of concept 

# Recipe RAG

A simple recipe question-answering API built with Python and Retrieval-Augmented Generation (RAG) for my favourite cookbook. It uses semantic search to find relevant recipes and a local language model to generate answers based on them.


## Getting Started

### Prerequisites

- Docker and Docker Compose
- A compatible Linux GPU setup for the supplied `llama.cpp` configuration (see models currently used below). This was deigned on a 16GB card.

### Run the application

Clone the repository and start the services:

```bash
git clone https://github.com/VelcroGecko/recipe-RAG.git
cd recipe-RAG

docker compose up --build
```

The first startup downloads the required models, so it will take a while

The API is available at `http://localhost:8000`. See docs at `http://localhost:8000/docs`.

To stop the application:

```bash
docker compose down
```

## Example Usage

The API provides two main endpoints.

### Search recipes

Find recipes that are semantically similar to a query without generating an AI response.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quick chicken dinner",
    "top_k": 3
  }'
```

Returns matching recipes and their similarity scores.

### Ask a question

Retrieve relevant recipes and use the language model to generate an answer.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What can I cook with chicken and peanuts?"
  }'
```

The response includes the generated answer and the recipes used to produce it.

## How It Works

The application uses Retrieval-Augmented Generation (RAG) to answer questions using a recipe collection.

1. **Load recipes:** Recipe data is loaded from `data/recipes.json`. The knowledge base (currently only 4 recipes)
2. **Create embeddings:** A sentence-transformer model converts recipes into embeddings. (to capture the meaning of the words)
3. **Retrieve recipes:** FAISS efficiently searches these embeddings to find recipes relevant to the question
4. **Generate an answer:** The retrieved recipes are added to the prompt and sent to a language model running through `llama.cpp`. (for the query endpoint)
5. **Return results:** The API returns the generated answer and the source recipes.

## Technologies

- **Python and FastAPI** — API and request handling
- **Ray Serve** — application serving and deployment
- **Sentence Transformers** — text embeddings
- **FAISS** — vector similarity search
- **llama.cpp** — local language model inference
- **Docker Compose** — running the application and model server together

## Current Models used

## Models Used

| Model | Purpose |
|---|---|
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | Generates text embeddings for semantic recipe search. |
| [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | Generates answers using the recipes retrieved by the RAG pipeline. Served through `llama.cpp` using the `UD-Q4_K_XL` quantisation. |
