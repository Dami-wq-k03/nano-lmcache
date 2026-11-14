"""Dependency-free executable reference for logical KV page ownership."""

from dataclasses import dataclass

from .layout import KVLayout
from .page import KVPageDirectory, KVPageRef, PageLocation, PageState


@dataclass(frozen=True, slots=True)
class PageReservation:
    """A logical page reservation and the bytes expected for its token range."""

    ref: KVPageRef
    expected_nbytes: int


class ReferencePageStore:
    """Store page bytes while exercising the data-plane invariants.

    This is intentionally not a performance backend. It is the small CPU
    specification used by later CUDA and storage adapters.
    """

    def __init__(self) -> None:
        self.directory = KVPageDirectory()
        self._payloads: dict[tuple[PageLocation, str | int], bytes] = {}
        self._generations: dict[str, int] = {}

    def reserve(
        self,
        *,
        logical_key: str,
        page_id: str | int,
        token_range: tuple[int, int],
        location: PageLocation,
        layout: KVLayout,
        generation: int = 0,
        compression: str = "none",
    ) -> PageReservation:
        """Reserve a physical page, replacing older generations atomically."""
        latest = self._generations.get(logical_key, -1)
        if generation < latest:
            raise ValueError(
                f"generation {generation} is older than current {latest}"
            )
        if generation > latest:
            for old in self.directory.replicas(logical_key):
                self._payloads.pop(old.identity, None)
                self.directory.remove(old)
            self._generations[logical_key] = generation

        ref = KVPageRef(
            logical_key=logical_key,
            page_id=page_id,
            token_range=token_range,
            location=location,
            layout=layout,
            generation=generation,
            compression=compression,
        )
        self.directory.register(ref)
        return PageReservation(
            ref, layout.nbytes_for_tokens(token_range[1] - token_range[0])
        )

    def complete(self, reservation: PageReservation, payload: bytes) -> bool:
        """Publish a transfer only if its reservation is still current."""
        ref = reservation.ref
        if len(payload) != reservation.expected_nbytes:
            raise ValueError(
                f"payload has {len(payload)} bytes; expected "
                f"{reservation.expected_nbytes}"
            )
        registered = next(
            (candidate for candidate in self.directory.replicas(ref.logical_key)
             if candidate is ref),
            None,
        )
        if registered is None:
            return False
        if ref.state is not PageState.RESERVED:
            raise ValueError(f"cannot complete a {ref.state.value} page")
        ref.begin_transfer()
        self._payloads[ref.identity] = bytes(payload)
        ref.mark_ready()
        if self.directory.authoritative(ref.logical_key) is None:
            self.directory.set_authoritative(ref)
        return True

    def load(self, logical_key: str) -> bytes | None:
        """Read bytes from the ready authoritative replica."""
        ref = self.directory.get(logical_key)
        return None if ref is None else self._payloads.get(ref.identity)

    def evict(self, ref: KVPageRef) -> None:
        """Evict one replica and allow the directory to promote a fallback."""
        if ref.state is not PageState.READY:
            raise ValueError(f"cannot evict a {ref.state.value} page")
        ref.begin_eviction()
        ref.mark_evicted()
        self._payloads.pop(ref.identity, None)
        self.directory.remove(ref)
