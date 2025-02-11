# microsearch

High-level (and possibly very slow) search over PostgreSQL + pgvector. Good for PoCs and prototypes.

# Installation

```sh
uv add git+https://github.com/nthe/microsearch
```

# Usage

Define model

```py
from pgvector.sqlalchemy import Vector
from sqlmodel import Field, SQLModel, Column, Relationship

class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    vector: list[float] = Field(default=None, sa_column=Column(Vector(384)))
```

Create engine and tables

```py
from sqlmodel import create_engine

engine = create_engine("postgresql://admin:admin@localhost:5432/db")

SQLModel.metadata.create_all(engine)
```

Index data

```py
import httpx
from itertools import batched

def embed(text: str) -> list[float]:
    import ollama
    return ollama.embed(model="all-minilm:33m", input=text).embeddings[0]

book = httpx.get("https://www.gutenberg.org/cache/epub/2591/pg2591.txt").text

docs = []
for no, words in enumerate(batched(book.split(), n=100)):
    chunk = " ".join(words)
    doc = Document(text=chunk, vector=embed(chunk))
    docs.append(doc)

with Session(engine) as session:
    session.add_all(docs)
    session.commit()
```

Search (full-text example)

```py
with ms.MicroSession(engine) as s:
    for hit in s.fulltext(query="Grimm Tales", schema=Document, multimatch=True):
        print(hit.score, "\n", ms.wrapped(hit.item.text), "\n")
```

Hybrid search + RRF reranking

```py
from src import microsearch as ms

query = "Grimm Tales"

def ident(doc: ms.Result[Document]) -> str:
    """Return unique (hashable) property of the object."""
    return doc.item.id


with ms.MicroSession(engine) as sx:
    docs, scores = ms.weighted_reciprocal_rank(
        arrays=[
            sx.fulltext(query=query, schema=Document, multimatch=True),
            sx.semantic(query=embed(query), schema=Document),
            sx.trigram(query=query, schema=Document),
        ],
        ident_fn=ident,
    )

    for doc in docs:
        print(doc.kind, doc.score, "\n", ms.wrapped(doc.item.text), "\n")

```
