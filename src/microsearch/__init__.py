from collections import defaultdict
from textwrap import wrap
from typing import  Callable, Generator,  Hashable, Iterable, Literal, Sequence

from sqlmodel import SQLModel, Session, select, text, func
from sqlalchemy import Engine
from pydantic import BaseModel


engine: Engine = ...


def set_engine(engine_: Engine) -> None:
    global engine
    engine = engine_
    with Session(engine) as session:
        session.exec(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        session.exec(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))


def oneof(text: str) -> str:
    """Convert phrase or query to multi-match statement."""
    return " OR ".join(text.split())



def wrapped(text: str, width: int = 70) -> str:
    """Return text with words wrapped to width."""
    return "\n".join(wrap(text, width=width))



class Result[T: SQLModel](BaseModel):
    kind: Literal["trigram", "fulltext", "semantic"]
    score: float
    item: T


def trigram[T: SQLModel](
    query: str,
    schema: T,
    column: str = "text",
    limit: int = 20,
) -> Generator[Result[T], None, None]:
    """Perform trigram (fuzzy) search over column. Return similar results."""
    with Session(engine) as session:
        scorer = func.similarity(query, text(column))
        statement = select(scorer, schema).order_by(scorer.desc()).limit(limit)
        rows = session.exec(statement)
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="trigram")
        


def semantic[T: SQLModel](
    query: list[float],
    schema: T,
    column: str = "vector",
    limit: int = 20,
) -> Generator[Result[T], None, None]:
    """Perform semantic (vector) search over column. Return nearby results."""
    with Session(engine) as session:
        scorer = 1 - schema.model_fields[column].sa_column.cosine_distance(query)
        statement = select(scorer, schema).order_by(scorer.desc()).limit(limit)
        rows = session.exec(statement)
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="semantic")




def fulltext[T: SQLModel](
    query: str,
    schema: T,
    column: str = "text",
    limit: int = 20,
    multimatch: bool = True,
) -> Generator[Result[T], None, None]:
    """Perform full-text search over column. Return matched results."""
    with Session(engine) as session:
        scorer = text(
            f"ts_rank_cd(to_tsvector({column}), websearch_to_tsquery(:query)) AS score"
        )
        statement = select(scorer, schema).order_by(text("score DESC")).limit(limit)
        if multimatch:
            query = oneof(query)
        rows = session.exec(statement, params={"query": query})
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="fulltext")




def weighted_reciprocal_rank[T](
    arrays: Sequence[Iterable[T]],
    ident_fn: Callable[[T], Hashable] = lambda item: id(item),
    weights: Sequence[float] | None = None,
    c: int | None = None,
) -> list[T]:
    """Rerank and merge multiple sets using reciprocal rank fusion."""
    if weights is None:
        weights = [1.0] * len(arrays)

    if len(arrays) != len(weights):
        raise ValueError("Number of rank lists must be equal to the number of weights.")

    if c is None or c < 1:
        c = len(arrays)

    unique_docs = {}
    rrf_score: dict[str, float] = defaultdict(float)
    for array, weight in zip(arrays, weights):
        for rank, doc in enumerate(array, start=1):
            identity = ident_fn(doc)
            rrf_score[identity] += weight / (rank + c)
            if identity not in unique_docs:
                unique_docs[identity] = doc

    return sorted(
        unique_docs.values(),
        reverse=True,
        key=lambda doc: rrf_score[ident_fn(doc)],
    ), rrf_score