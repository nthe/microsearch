import ollama
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pgvector.sqlalchemy import Vector
from sqlmodel import create_engine, Field, SQLModel, Column, Relationship

from microsearch import TrigramIndex, FullTextIndex
from microsearch import Result, microsearch, weighted_reciprocal_rank


app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Meta(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    documents: list["Document"] | None = Relationship(back_populates="meta")


class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    vector: list[float] = Field(default=None, sa_column=Column(Vector(dim=384)))
    meta_id: int | None = Field(default=None, foreign_key="meta.id")
    meta: Meta | None = Relationship(back_populates="documents")

    __table_args__ = (
        TrigramIndex(table="document", column="text"),
        FullTextIndex(table="document", column="text"),
    )


engine = create_engine("postgresql://admin:admin@localhost:5432/db", echo=False)

SQLModel.metadata.create_all(engine)


def embed(text: str) -> list[float]:
    return ollama.embed(model="all-minilm:33m", input=text).embeddings[0]


def ident(doc: Result[Document]) -> str:
    """Return unique (hashable) property of the object."""
    return doc.item.id


def find(query: str, fulltext: bool, trigram: bool, semantic: bool):
    with microsearch(engine) as use:
        docs, _ = weighted_reciprocal_rank(
            arrays=[
                use.fulltext(table=Document, query=query) if fulltext else [],
                use.semantic(table=Document, query=embed(query)) if semantic else [],
                use.trigram(table=Document, query=query, strict=False) if trigram else [],
            ],
            ident_fn=ident,
        )

        yield from docs


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search")
async def search(q: str, t: bool, s: bool, f: bool):
    out = [
        d.model_dump(exclude={"item": {"vector"}})
        for d in find(
            query=q,
            fulltext=f,
            trigram=t,
            semantic=s,
        )
    ]
    return {"data": out}
