"""Teaching version of prefill/decode (PD) disaggregation.

A prefill worker turns a prompt into KV and a first token, then hands the
KV to a separate decode worker that generates the rest. Everything here
runs on CPU with a deterministic toy model, so each property can be
checked exactly:

- `toy_model`: a tiny "language model" whose KV depends on the whole prefix.
- `handoff`: the versioned, checksummed message a prefill worker sends.
- `transport`: how that message travels (memory or a shared directory).
- `workers`: prefill and decode roles, plus the colocated baseline.
- `simulation`: a timeline model of why splitting the roles helps.

See docs/pd-disaggregation.md.
"""

from nano_lmcache.disaggregation.handoff import (
    ConfigMismatchError,
    CorruptHandoffError,
    HandoffError,
    IncompleteHandoffError,
    KVHandoff,
)
from nano_lmcache.disaggregation.simulation import (
    CostModel,
    RequestTiming,
    SimRequest,
    simulate_colocated,
    simulate_disaggregated,
)
from nano_lmcache.disaggregation.toy_model import (
    KVEntry,
    ToyLanguageModel,
    ToyModelConfig,
)
from nano_lmcache.disaggregation.transport import (
    DirectoryTransport,
    DuplicateHandoffError,
    InMemoryTransport,
    KVTransport,
)
from nano_lmcache.disaggregation.workers import (
    DecodeResult,
    DecodeWorker,
    HandoffNotFoundError,
    PrefillResult,
    PrefillWorker,
    generate_colocated,
)

__all__ = [
    "ConfigMismatchError",
    "CorruptHandoffError",
    "CostModel",
    "DecodeResult",
    "DecodeWorker",
    "DirectoryTransport",
    "DuplicateHandoffError",
    "HandoffError",
    "HandoffNotFoundError",
    "IncompleteHandoffError",
    "InMemoryTransport",
    "KVEntry",
    "KVHandoff",
    "KVTransport",
    "PrefillResult",
    "PrefillWorker",
    "RequestTiming",
    "SimRequest",
    "ToyLanguageModel",
    "ToyModelConfig",
    "generate_colocated",
    "simulate_colocated",
    "simulate_disaggregated",
]
