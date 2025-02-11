# microsearch

High-level (and possibly very slow) search over PostgreSQL + pgvector. Good for PoCs and prototypes.

# Usage

Define models

```py
from pgvector.sqlalchemy import Vector
from sqlmodel import Field, SQLModel, Session, Column, Relationship

import microsearch as ms

engine = ms.create_engine("postgresql://admin:admin@localhost:5432/db")

ms.set_engine(engine)

class Meta(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    chunk: int
    document: list["Document"] | None = Relationship(back_populates="meta")


class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    meta_id: int | None = Field(default=None, foreign_key="meta.id")
    meta: Meta | None = Relationship(back_populates="document")
    vector: list[float] = Field(default=None, sa_column=Column(Vector(384)))


SQLModel.metadata.create_all(engine)
```

Index data

```py
import httpx
from itertools import batched

import microsearch as ms

book = httpx.get("https://www.gutenberg.org/cache/epub/2591/pg2591.txt").text


docs = []
for no, words in enumerate(batched(book.split(), n=100)):
    chunk = " ".join(words)
    meta = Meta(chunk=no)
    doc = Document(text=chunk, vector=embed(chunk), meta=meta)
    docs.append(doc)


with Session(ms.engine) as session:
    session.add_all(docs)
    session.commit()
```

Search

```py
import microsearch as ms

docs, scores = ms.weighted_reciprocal_rank(
    arrays=[
        ms.fulltext(query=query, schema=Document, multimatch=True),
        ms.semantic(query=embed(query), schema=Document),
        ms.trigram(query=query, schema=Document),
    ],
    ident_fn=ident,
)

print(len(docs))
if docs:
    print(ms.wrapped(next(iter(docs)).item.text))
```
