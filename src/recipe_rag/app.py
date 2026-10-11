import json

import faiss
import httpx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve

from recipe_rag.settings import settings

app = FastAPI(title="recipe-RAG")

PROMPT_TEMPLATE = """
You are a cooking assistant.
Answer the question using ONLY the recipes below, citing their ids in square brackets.
If the recipes don't cover it, say so.

Recipes:
{context}

Question: {question}"""


class SearchRequest(BaseModel):
    """Retrieve the top-k most similar recipes for a query (no LLM involved)."""

    query: str = Field(
        description="Natural-language search text, e.g. 'spicy noodles with peanuts'"
    )
    top_k: int | None = Field(
        default=None,
        description="Number of recipes to return; falls back to TOP_K in settings",
    )


class QueryRequest(BaseModel):
    """Full RAG: retrieve recipes, then generate a grounded answer with citations."""

    question: str = Field(description="Natural-language question about your recipes")
    top_k: int | None = Field(
        default=None,
        description="Number of recipes to retrieve; falls back to TOP_K in settings",
    )


def recipe_to_text(recipe: dict) -> str:
    ingredients = "; ".join(
        f"{ing['quantity']} {ing['item']}".strip() for ing in recipe["ingredients"]
    )
    return (
        f"{recipe['name']}. Cuisine: {recipe['cuisine']}. "
        f"Tags: {', '.join(recipe['tags'])}. "
        f"Ingredients: {ingredients}. Steps: {' '.join(recipe['steps'])}"
    )


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


@serve.deployment
@serve.ingress(app)
class RecipeRAG:
    async def __init__(self, embedder):
        self._embedder = embedder
        with open(settings.recipes_path) as f:
            self._recipes = json.load(f)["recipes"]
        texts = [recipe_to_text(r) for r in self._recipes]
        vectors = np.array(
            await self._embedder.embed.remote(texts), dtype="float32"
        )
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    async def _retrieve(self, question: str, k: int | None = None) -> list[dict]:
        query_vec = np.array(
            await self._embedder.embed.remote([question]), dtype="float32"
        )
        scores, ids = self._index.search(query_vec, k or settings.top_k)
        return [
            {"score": round(float(score), 4), "recipe": self._recipes[idx]}
            for score, idx in zip(scores[0], ids[0])
        ]

    @app.get("/health")
    async def health(self):
        return {"status": "ok", "recipes_indexed": len(self._recipes)}

    @app.post("/search")
    async def search(self, body: SearchRequest):
        return {"results": await self._retrieve(body.query, body.top_k)}

    @app.post("/query")
    async def query(self, body: QueryRequest):
        hits = await self._retrieve(body.question, body.top_k)
        context = "\n\n".join(
            f"[{h['recipe']['id']}] {recipe_to_text(h['recipe'])}" for h in hits
        )
        payload = {
            "model": settings.generator_model,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        context=context, question=body.question
                    ),
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 400,
        }
        async with httpx.AsyncClient(
            base_url=settings.generator_base_url, timeout=60
        ) as client:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()
        return {
            "question": body.question,
            "answer": answer,
            "sources": [
                {"id": h["recipe"]["id"], "name": h["recipe"]["name"], "score": h["score"]}
                for h in hits
            ],
        }


entrypoint = RecipeRAG.bind(Embedder.bind(settings.embed_model))
