from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from textwrap import wrap
from typing import Callable, Generator, Hashable, Iterable, Literal, Sequence

from sqlmodel import SQLModel, Session, select, text, func
from sqlalchemy import Engine
from pydantic import BaseModel


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


@dataclass
class MicroSearch:
    session: Session

    def check_extensions(self) -> None:
        self.session.exec(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        self.session.exec(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

    def trigram[T: SQLModel](
        self,
        query: str,
        schema: T,
        column: str = "text",
        limit: int = 20,
    ) -> Generator[Result[T], None, None]:
        """Perform trigram (fuzzy) search over column. Return similar results."""
        scorer = func.similarity(query, text(column))
        statement = select(scorer, schema).order_by(scorer.desc()).limit(limit)
        rows = self.session.exec(statement)
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="trigram")

    def semantic[T: SQLModel](
        self,
        query: list[float],
        schema: T,
        column: str = "vector",
        limit: int = 20,
    ) -> Generator[Result[T], None, None]:
        """Perform semantic (vector) search over column. Return nearby results."""
        scorer = 1 - schema.model_fields[column].sa_column.cosine_distance(query)
        statement = select(scorer, schema).order_by(scorer.desc()).limit(limit)
        rows = self.session.exec(statement)
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="semantic")

    def fulltext[T: SQLModel](
        self,
        query: str,
        schema: T,
        column: str = "text",
        limit: int = 20,
        multimatch: bool = True,
    ) -> Generator[Result[T], None, None]:
        """Perform full-text search over column. Return matched results."""
        scorer = text(f"ts_rank_cd(to_tsvector({column}), websearch_to_tsquery(:query)) AS score")
        statement = select(scorer, schema).order_by(text("score DESC")).limit(limit)
        if multimatch:
            query = oneof(query)
        rows = self.session.exec(statement, params={"query": query})
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="fulltext")


@contextmanager
def MicroSession(engine: Engine):
    with Session(engine) as session:
        yield MicroSearch(session=session)


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
