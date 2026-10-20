"""Hit/miss accounting that makes prefix-cache reuse observable."""

from dataclasses import dataclass


@dataclass
class CacheMetrics:
    """Running chunk-level totals across many cache lookups.

    Counting happens per chunk rather than per request, because a prefix
    cache rarely hits "all or nothing": a request typically reuses its
    first few chunks and recomputes the rest.

    A lookup stops probing at its first miss, so every complete chunk
    after that point still has to be computed by the caller. Those are
    all counted as misses, which makes `hit_rate` readable as "what
    fraction of reusable work was actually reused". Incomplete trailing
    chunks never participate in reuse, so they are counted on neither
    side.
    """

    hit_chunks: int = 0
    miss_chunks: int = 0
    hit_tokens: int = 0

    @property
    def total_chunks(self) -> int:
        """Number of reuse-eligible chunks seen across all lookups."""
        return self.hit_chunks + self.miss_chunks

    @property
    def hit_rate(self) -> float:
        """Share of reuse-eligible chunks served from cache, 0.0 if none."""
        if self.total_chunks == 0:
            return 0.0

        return self.hit_chunks / self.total_chunks

    def record_lookup(
        self,
        *,
        hit_chunks: int,
        miss_chunks: int,
        hit_tokens: int,
    ) -> None:
        """Fold one lookup's outcome into the running totals."""
        self.hit_chunks += hit_chunks
        self.miss_chunks += miss_chunks
        self.hit_tokens += hit_tokens
