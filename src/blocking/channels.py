"""
Candidate retrieval channels.

Every channel works on one country partition:

    channel.fit(targets)            # targets: DataFrame, positional rows
    for batch in channel.retrieve(queries):
        ...                         # CandidateBatch over queries[q_start:q_end]

Batches carry positional indices (query row, target row), never IDs, and
each batch holds the *complete* candidate list for a contiguous query
range. Consumers can therefore evaluate or write candidates chunk by chunk
without ever materializing the full candidate set.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from src.preprocessing.transliterate import transliterate_text


@dataclass
class CandidateBatch:
    q_start: int
    q_end: int
    q_idx: np.ndarray  # int32, absolute query row
    t_idx: np.ndarray  # int32, target row
    score: np.ndarray  # float32
    rank: np.ndarray  # int16, 0 = best within the query


class CandidateChannel:

    name = "base"

    def fit(self, targets: pd.DataFrame) -> None:
        raise NotImplementedError

    def retrieve(self, queries: pd.DataFrame) -> Iterator[CandidateBatch]:
        raise NotImplementedError


# ----------------------------------------------------------------------
# A. Exact normalized name
# ----------------------------------------------------------------------


def hash_strings(values: pd.Series) -> np.ndarray:

    return pd.util.hash_array(values.to_numpy(dtype=object))


class ExactNameChannel(CandidateChannel):
    """
    All targets whose normalized name equals the query's.

    Targets are stored as a sorted uint64 hash array; lookups are
    vectorized ``searchsorted`` ranges, so there is no Python dict.
    """

    name = "exact_name"

    def __init__(self, column: str = "norm_name", chunk: int = 200_000):
        self.column = column
        self.chunk = chunk

    def fit(self, targets: pd.DataFrame) -> None:

        names = targets[self.column].fillna("")
        hashes = hash_strings(names)

        # Empty names never match; give them a hash no query produces.
        valid = (names != "").to_numpy()

        order = np.argsort(hashes[valid], kind="stable")
        self.sorted_hashes = hashes[valid][order]
        self.sorted_rows = np.flatnonzero(valid)[order].astype(np.int32)

    def retrieve(self, queries: pd.DataFrame) -> Iterator[CandidateBatch]:

        names = queries[self.column].fillna("")

        for start in range(0, len(queries), self.chunk):
            end = min(start + self.chunk, len(queries))

            chunk_names = names.iloc[start:end]
            hashes = hash_strings(chunk_names)

            left = np.searchsorted(self.sorted_hashes, hashes, "left")
            right = np.searchsorted(self.sorted_hashes, hashes, "right")

            counts = right - left
            counts[(chunk_names == "").to_numpy()] = 0

            q_local = np.repeat(np.arange(end - start), counts)
            offsets = np.arange(counts.sum()) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            positions = np.repeat(left, counts) + offsets

            yield CandidateBatch(
                q_start=start,
                q_end=end,
                q_idx=(q_local + start).astype(np.int32),
                t_idx=self.sorted_rows[positions],
                score=np.ones(len(positions), dtype=np.float32),
                rank=offsets.astype(np.int16),
            )


# ----------------------------------------------------------------------
# B. Character n-gram TF-IDF top-K
# ----------------------------------------------------------------------

# Forked workers read these module globals copy-on-write, so the target
# index is shared rather than pickled into every process.
_SHARED: dict = {}


RERANK_SLICE = 250_000


def _rerank_full_cosine(
    q_abs: np.ndarray,
    t_idx: np.ndarray,
    partial: np.ndarray,
    k: int,
) -> tuple:
    """
    Re-score shortlisted pairs by cosine over ALL n-grams and keep the
    top-k per query. Only the shortlisted pairs are touched, in slices
    so the gathered row matrices stay small.
    """

    queries = _SHARED["full_queries"]
    full_targets = _SHARED["full_targets"]

    full = np.empty(len(q_abs), dtype=np.float32)

    for lo in range(0, len(q_abs), RERANK_SLICE):
        hi = lo + RERANK_SLICE
        left = queries[q_abs[lo:hi]]
        right = full_targets[t_idx[lo:hi]]
        full[lo:hi] = np.asarray(left.multiply(right).sum(axis=1)).ravel()

    # Group by query; best full cosine first, pruned score breaks ties.
    order = np.lexsort((-partial, -full, q_abs))
    q_sorted = q_abs[order]

    group_start = np.r_[0, np.flatnonzero(np.diff(q_sorted)) + 1]
    sizes = np.diff(np.r_[group_start, len(q_sorted)])
    rank = np.arange(len(q_sorted)) - np.repeat(group_start, sizes)

    keep = order[rank < k]

    return (
        q_abs[keep],
        t_idx[keep],
        full[keep],
        rank[rank < k].astype(np.int16),
    )


def _topk_rows(bounds: tuple[int, int]) -> tuple:

    start, end = bounds
    k = _SHARED["k"]
    pool_size = _SHARED["pool_size"]

    product = (_SHARED["queries"][start:end] @ _SHARED["index_t"]).tocsr()

    q_parts, t_parts, s_parts, r_parts = [], [], [], []

    indptr = product.indptr
    indices = product.indices
    data = product.data

    for row in range(end - start):
        lo, hi = indptr[row], indptr[row + 1]

        if lo == hi:
            continue

        scores = data[lo:hi]
        cols = indices[lo:hi]

        if hi - lo > pool_size:
            top = np.argpartition(-scores, pool_size - 1)[:pool_size]
            scores = scores[top]
            cols = cols[top]

        order = np.argsort(-scores, kind="stable")

        q_parts.append(np.full(len(order), start + row, dtype=np.int32))
        t_parts.append(cols[order])
        s_parts.append(scores[order])
        r_parts.append(np.arange(len(order), dtype=np.int16))

    if not q_parts:
        empty = np.empty(0)
        return (
            start,
            end,
            empty.astype(np.int32),
            empty.astype(np.int32),
            empty.astype(np.float32),
            empty.astype(np.int16),
        )

    q_abs = np.concatenate(q_parts)
    t_idx = np.concatenate(t_parts).astype(np.int32)
    scores = np.concatenate(s_parts).astype(np.float32)

    if _SHARED["full_targets"] is not None:
        return (
            start,
            end,
            *_rerank_full_cosine(q_abs, t_idx, scores, k),
        )

    return (
        start,
        end,
        q_abs,
        t_idx,
        scores,
        np.concatenate(r_parts),
    )


class CharNgramChannel(CandidateChannel):
    """
    Top-K targets by cosine similarity of character n-gram TF-IDF vectors.

    Memory-safety choices:
    - ``HashingVectorizer``: no vocabulary dict over millions of n-grams.
    - IDF is computed from column counts of the target matrix.
    - N-grams with target document frequency above ``max_df`` are left
      out of the retrieval index (their postings dominate the cost).
      Vectors are L2-normalized *before* pruning, so scores stay partial
      cosines on the original scale.
    - Queries are processed in chunks sized by an estimate of postings
      touched, in forked worker processes sharing the index.

    ``rerank_pool``: when set, shortlist that many targets per query by
    the pruned-index score, re-score only the shortlist by FULL cosine
    (all n-grams, including those left out of the index), and return
    the top ``k`` by full cosine. ``None`` keeps the pruned top-k.

    ``renormalize_pruned``: when True, query and target vectors are
    re-normalized to unit L2 norm over the RETAINED n-grams only, so the
    retrieval score is a true cosine in the pruned space. When False
    (default), vectors keep their full-space norm and the pruned score
    is a partial cosine.
    """

    def __init__(
        self,
        *,
        column: str = "norm_name",
        ngram_range: tuple[int, int] = (3, 3),
        n_features: int = 2**22,
        max_df: int = 30_000,
        k: int = 100,
        workers: int = 8,
        chunk_work: int = 20_000_000,
        rerank_pool: int | None = None,
        renormalize_pruned: bool = False,
    ):
        if rerank_pool is not None and rerank_pool < k:
            raise ValueError("rerank_pool must be >= k")

        self.column = column
        self.rerank_pool = rerank_pool
        self.renormalize_pruned = renormalize_pruned
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.k = k
        self.workers = workers
        self.chunk_work = chunk_work

        self.name = (
            f"char{ngram_range[0]}{ngram_range[1]}_{column}_df{max_df // 1000}k"
            + ("_renorm" if renormalize_pruned else "")
            + (f"_rr{rerank_pool}" if rerank_pool else "")
        )

        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            binary=True,
            dtype=np.float32,
        )

    def _vectorize(self, texts: pd.Series) -> sp.csr_matrix:

        counts = self.vectorizer.transform(texts.fillna("").tolist())
        weighted = counts.multiply(self.idf).tocsr()

        return l2_normalize(weighted, copy=False)

    def _prune(
        self,
        matrix: sp.csr_matrix,
        copy: bool = True,
    ) -> sp.csr_matrix:
        """Drop over-common n-grams; optionally re-normalize what is left."""

        if copy:
            matrix = matrix.copy()

        keep = self.df[matrix.indices] <= self.max_df
        matrix.data[~keep] = 0
        matrix.eliminate_zeros()

        if self.renormalize_pruned:
            matrix = l2_normalize(matrix, copy=False)

        return matrix

    def retrieval_vectors(self, texts: pd.Series) -> sp.csr_matrix:
        """Vectors as used for the pruned-index retrieval score."""

        return self._prune(self._vectorize(texts))

    def fit(self, targets: pd.DataFrame) -> None:

        texts = targets[self.column].fillna("")
        counts = self.vectorizer.transform(texts.tolist())

        n_docs = counts.shape[0]
        self.df = np.bincount(
            counts.indices,
            minlength=counts.shape[1],
        ).astype(np.int64)

        self.idf = (
            np.log((1 + n_docs) / (1 + self.df)) + 1
        ).astype(np.float32)

        del counts

        index = self._vectorize(texts)

        # Full (un-pruned) target vectors, needed only for re-ranking.
        self.full_targets = index.copy() if self.rerank_pool else None

        # Drop over-common n-grams from the retrieval index.
        index = self._prune(index, copy=self.full_targets is not None)

        self.index_t = index.T.tocsr()

        self.df_kept = np.where(self.df <= self.max_df, self.df, 0)

    def index_nbytes(self) -> int:

        total = 0

        for m in (self.index_t, self.full_targets):
            if m is not None:
                total += m.data.nbytes + m.indices.nbytes + m.indptr.nbytes

        return total

    def _chunks(self, queries: sp.csr_matrix) -> list[tuple[int, int]]:
        """Split query rows so each chunk touches ~chunk_work postings."""

        binary = queries.copy()
        binary.data[:] = 1
        work = np.asarray(binary @ self.df_kept).ravel()

        bounds = []
        start = 0
        total = 0

        for row, cost in enumerate(work):
            total += cost

            if total >= self.chunk_work:
                bounds.append((start, row + 1))
                start = row + 1
                total = 0

        if start < len(work):
            bounds.append((start, len(work)))

        self.last_work = work

        return bounds

    def retrieve(self, queries: pd.DataFrame) -> Iterator[CandidateBatch]:

        full = self._vectorize(queries[self.column])
        matrix = self._prune(full) if self.renormalize_pruned else full
        bounds = self._chunks(matrix)

        _SHARED.update(
            queries=matrix,
            full_queries=full,
            index_t=self.index_t,
            full_targets=self.full_targets,
            k=self.k,
            pool_size=self.rerank_pool or self.k,
        )

        try:
            with mp.get_context("fork").Pool(self.workers) as pool:
                for result in pool.imap(_topk_rows, bounds):
                    yield CandidateBatch(*result)
        finally:
            _SHARED.clear()


# ----------------------------------------------------------------------
# D. Address character n-gram TF-IDF top-K
# ----------------------------------------------------------------------


class AddressChannel(CharNgramChannel):
    """
    Character n-gram retrieval over the normalized address.

    Same machinery as ``CharNgramChannel`` (hashed trigrams, target-side
    IDF, df-capped sparse index, chunked forked workers, optional
    full-cosine rerank); only the text column differs.
    """

    def __init__(self, *, column: str = "norm_address", **kwargs):
        super().__init__(column=column, **kwargs)

        self.name = f"address_{self.name}"


# ----------------------------------------------------------------------
# C. Transliterated-name character n-gram TF-IDF top-K
# ----------------------------------------------------------------------


class TransliteratedNameChannel(CharNgramChannel):
    """
    Character n-gram retrieval over ``transliterate_text(norm_name)``.

    Same machinery as ``CharNgramChannel``; the Latin-script column is
    derived on the fly from ``source_column`` for each partition, so no
    cache change is needed.
    """

    def __init__(
        self,
        *,
        source_column: str = "norm_name",
        column: str = "translit_name",
        **kwargs,
    ):
        super().__init__(column=column, **kwargs)

        self.source_column = source_column

    def _transliterated(self, df: pd.DataFrame) -> pd.DataFrame:

        return pd.DataFrame(
            {
                self.column: df[self.source_column]
                .fillna("")
                .map(transliterate_text)
            }
        )

    def fit(self, targets: pd.DataFrame) -> None:

        super().fit(self._transliterated(targets))

    def retrieve(self, queries: pd.DataFrame) -> Iterator[CandidateBatch]:

        yield from super().retrieve(self._transliterated(queries))


# ----------------------------------------------------------------------
# E. Exact house-number + city blocking
# ----------------------------------------------------------------------


class HouseCityChannel(ExactNameChannel):
    """
    All targets sharing the query's ``house|city`` address key, for
    queries whose block holds at most ``max_block`` targets.

    Keys come from ``address_keys.extract_keys`` with a city lexicon
    learned from S1 (``address_keys.learn_city_lexicon``).
    """

    name = "house_city"

    def __init__(
        self,
        *,
        country: str,
        lexicon: frozenset[str],
        max_block: int = 1000,
        chunk: int = 200_000,
    ):
        super().__init__(column="house_city", chunk=chunk)

        self.country = country
        self.lexicon = lexicon
        self.max_block = max_block

    def _keys(self, df: pd.DataFrame) -> pd.DataFrame:

        from src.blocking.address_keys import extract_keys

        keys = extract_keys(df["norm_address"], self.country, self.lexicon)

        return keys[["house_city"]].reset_index(drop=True)

    def fit(self, targets: pd.DataFrame) -> None:

        super().fit(self._keys(targets))

    def retrieve(self, queries: pd.DataFrame) -> Iterator[CandidateBatch]:

        keys = self._keys(queries)
        hashes = hash_strings(keys["house_city"])

        sizes = (
            np.searchsorted(self.sorted_hashes, hashes, "right")
            - np.searchsorted(self.sorted_hashes, hashes, "left")
        )

        # Blank out keys whose block exceeds the cap.
        keys.loc[sizes > self.max_block, "house_city"] = ""

        yield from super().retrieve(keys)
