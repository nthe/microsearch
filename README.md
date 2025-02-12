# microsearch

High-level (and possibly very slow) search over PostgreSQL + pgvector. Good for PoCs and prototypes.

# Installation

```sh
uv add git+https://github.com/nthe/microsearch
```

# Usage

Define tables using `SQLModel` / `Pydantic` syntax.

```py
from pgvector.sqlalchemy import Vector
from sqlmodel import Field, SQLModel, Column, Relationship

from microsearch import FullTextColumn

class Meta(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    documents: list["Document"] | None = Relationship(back_populates="meta")

class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    text_tsvector: str = Field(default=None, sa_column=FullTextColumn(column="text"))
    vector: list[float] = Field(default=None, sa_column=Column(Vector(dim=384)))

    meta_id: int | None = Field(default=None, foreign_key="meta.id")
    meta: Meta | None = Relationship(back_populates="documents")
```

Create `SQLAlchemy` engine instance and create tables

```py
from sqlmodel import create_engine

engine = create_engine("postgresql://admin:admin@localhost:5432/db")

SQLModel.metadata.create_all(engine)
```

Embedding function for data indexing

```py
import ollama

def embed(text: str) -> list[float]:
    return ollama.embed(model="all-minilm:33m", input=text).embeddings[0]
```

Load data for indexing

```py
import httpx
book = httpx.get("https://www.gutenberg.org/cache/epub/2591/pg2591.txt").text
```

Prepare index data in batches

```py
from itertools import batched

docs = []
for no, words in enumerate(batched(book.split(), n=100)):
    chunk = " ".join(words)
    meta = Meta(title=f"Document number {no}")
    doc = Document(text=chunk, vector=embed(chunk), meta=meta)
    docs.append(doc)
```

Push them to database

```py
from sqlmodel import Session

with Session(engine) as session:
    session.add_all(docs)
    session.commit()
```

Perform simple full-text search

```py
from microsearch import Result, MicroSession, wrapped

query = "Grimm Tales"

def ident(doc: Result[Document]) -> str:
    """Return unique (hashable) property of the object."""
    return doc.item.id

with MicroSession(engine) as sx:
    docs = sx.fulltext(
        query=query,
        schema=Document,
        multimatch=True,
        column="text_tsvector",
        is_tsvector=True,
        limit=5
    )

    for i, doc in enumerate(docs, start=1):
        print(
            f"{i:>5} | {doc.kind:^8} | {doc.score:>5.4f} | ",
            wrapped(doc.item.text[:60] + "..."),
            sep="",
        )

```

Perform advanced hybrid search

```py
from microsearch import Result, MicroSession, weighted_reciprocal_rank, wrapped

query = "Grimm Tales"

def ident(doc: Result[Document]) -> str:
    """Return unique (hashable) property of the object."""
    return doc.item.id

with MicroSession(engine) as sx:
    docs, scores = weighted_reciprocal_rank(
        arrays=[
            sx.fulltext(query=query, schema=Document, multimatch=True),
            sx.semantic(query=embed(query), schema=Document),
            sx.trigram(query=query, schema=Document),
        ],
        ident_fn=ident,
    )

    for i, doc in enumerate(docs[:20], start=1):
        print(
            f"{i:>5} | {doc.kind:^8} | {doc.score:>5.4f} | ",
            wrapped(doc.item.text[:60] + "..."),
            sep="",
        )
```
