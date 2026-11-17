"""Stateless leaf scores: lower scores are evicted first.

PrefixIndex owns all access state and decides which keys are eligible. A custom
policy only needs a name and __call__(key, frequency, last_access); it must not
mutate cache state. There are no lifecycle callbacks or policy factories.
"""

from typing import Protocol


class EvictionPolicy(Protocol):
    name: str

    def __call__(
        self, key: str, frequency: int, last_access: int
    ) -> tuple[float, ...]: ...


PolicyLike = str | EvictionPolicy


class LRUEvictionPolicy:
    name = "lru"

    def __call__(
        self, key: str, frequency: int, last_access: int
    ) -> tuple[float, ...]:
        return (last_access,)


class LFUEvictionPolicy:
    name = "lfu"

    def __call__(
        self, key: str, frequency: int, last_access: int
    ) -> tuple[float, ...]:
        return (frequency, last_access)


def make_eviction_policy(policy: PolicyLike) -> EvictionPolicy:
    if not isinstance(policy, str):
        return policy
    if policy.lower() == "lru":
        return LRUEvictionPolicy()
    if policy.lower() == "lfu":
        return LFUEvictionPolicy()
    raise ValueError(f"unknown eviction_policy {policy!r}; expected 'lru' or 'lfu'")
