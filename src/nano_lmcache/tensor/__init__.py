"""Code that needs PyTorch (real KV tensors, GPU offload).

Nothing here may be imported by the top-level `nano_lmcache` package or by
`core` and `storage`: the core must work without PyTorch installed. Modules
in this package import torch at their top and are only used by people who
installed the `torch` or `hf` extra.
"""
