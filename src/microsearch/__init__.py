from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from textwrap import wrap
from typing import Callable, Generator, Hashable, Iterable, Literal, Sequence

from sqlmodel import Column, Computed, Index, SQLModel, Session, select, text, func
from sqlalchemy import cast, Engine, literal
from sqlalchemy.dialects.postgresql import TSVECTOR, REGCONFIG
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ColumnExpressionArgument
from pydantic import BaseModel


class MicroSearchError(BaseException):
    """Common microsearch error."""


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


def TrigramIndex(table: str, column: str) -> Index:
    """Create trigram index on given column of the table."""
    return Index(
        f"{table}_{column}_trgm_gist",
        column,
        postgresql_using="gist",
        postgresql_ops={column: "gist_trgm_ops(siglen=256)"},
    )


def FullTextIndex(table: str, column: str) -> Index:
    """Create full-tect index on given column of the table."""
    return Index(
        f"{table}_{column}_fts_gin",
        func.to_tsvector(
            cast(literal("english"), type_=REGCONFIG),
            text(column),
        ),
        postgresql_using="gin",
    )


def FullTextColumn(column: str) -> Column:
    """Create always autogenerated column for full-text search."""
    return Column(
        TSVECTOR,
        Computed(
            func.to_tsvector(
                cast(literal("english"), type_=REGCONFIG),
                func.coalesce(text(column), ""),
            )
        ),
    )


@dataclass
class MicroSearch:
    session: Session

    def check_extensions(self) -> None:
        self.session.exec(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        self.session.exec(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

    def trigram[T: SQLModel](
        self,
        schema: T,
        query: str,
        where: ColumnExpressionArgument | None = None,
        column: str | InstrumentedAttribute = "text",
        limit: int = 20,
        is_strict: bool = False,
    ) -> Generator[Result[T], None, None]:
        """Perform trigram (fuzzy) search over column. Return similar results."""
        if isinstance(column, InstrumentedAttribute):
            column = column.key

        try:
            if is_strict:
                target_column = getattr(schema, column)
        except AttributeError as exc:
            raise MicroSearchError(f"No such column - {column}") from exc

        scorer = func.word_similarity(query, text(column))
        statement = select(scorer, schema)

        if is_strict:
            trigram_index_filter = target_column.op("%>")(query)
            statement = statement.where(trigram_index_filter)
        if where is not None:
            statement = statement.where(where)

        statement = statement.order_by(scorer.desc()).limit(limit)

        rows = self.session.exec(statement)
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="trigram")

    def semantic[T: SQLModel](
        self,
        schema: T,
        query: list[float],
        where: ColumnExpressionArgument | None = None,
        column: str | InstrumentedAttribute = "vector",
        limit: int = 20,
    ) -> Generator[Result[T], None, None]:
        """Perform semantic (vector) search over column. Return nearby results."""
        if isinstance(column, InstrumentedAttribute):
            column = column.key

        try:
            target_column = getattr(schema, column)
        except AttributeError as exc:
            raise MicroSearchError(f"No such column - {column}") from exc

        scorer = 1 - target_column.cosine_distance(query)
        statement = select(scorer, schema)
        if where is not None:
            statement = statement.where(where)

        statement = statement.order_by(scorer.desc()).limit(limit)

        rows = self.session.exec(statement)
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="semantic")

    def fulltext[T: SQLModel](
        self,
        schema: T,
        query: str,
        where: ColumnExpressionArgument | None = None,
        column: str | InstrumentedAttribute = "text",
        limit: int = 20,
        multimatch: bool = True,
    ) -> Generator[Result[T], None, None]:
        """Perform full-text search over column. Return matched results."""
        if isinstance(column, InstrumentedAttribute):
            column = column.key

        try:
            getattr(schema, column)
        except AttributeError as exc:
            raise MicroSearchError(f"No such column - {column}") from exc

        if multimatch:
            query = oneof(query)

        scorer = func.ts_rank(
            func.to_tsvector(text(column)),
            func.websearch_to_tsquery(query),
        )
        statement = select(scorer, schema)

        fulltext_index_filter = func.to_tsvector(
            cast(literal("english"), type_=REGCONFIG),
            text(column),
        ).op("@@")(func.websearch_to_tsquery(query))
        statement = statement.where(fulltext_index_filter)

        if where is not None:
            statement = statement.where(where)

        statement = statement.order_by(scorer.desc()).limit(limit)

        rows = self.session.exec(statement, params={"query": query})
        for score, item in rows.all():
            yield Result(score=score, item=item, kind="fulltext")


@contextmanager
def MicroSession(engine: Engine):
    with Session(engine) as session:
        yield MicroSearch(session=session)


def weighted_reciprocal_rank[T](
    arrays: Sequence[Iterable[T]],
    ident_fn: Callable[[T], Hashable] = id,
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
