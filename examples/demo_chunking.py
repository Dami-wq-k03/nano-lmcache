
from nano_lmcache.core.chunking import (
    chunk_tokens,
    make_chunk_key,
)


def build_chunk_keys(tokens: list[int], chunk_size: int) -> list[str]:
    chunks = chunk_tokens(tokens, chunk_size=chunk_size)

    return [
        make_chunk_key(chunk, namespace="demo-model")
        for chunk in chunks
    ]


def main() -> None:
    # 两个请求共享前 4 个 Token，后续 Token 不同。
    request_a = [101, 205, 307, 409, 511, 613]
    request_b = [101, 205, 307, 409, 777, 888]

    chunk_size = 2

    chunks_a = chunk_tokens(request_a, chunk_size)
    chunks_b = chunk_tokens(request_b, chunk_size)

    keys_a = build_chunk_keys(request_a, chunk_size)
    keys_b = build_chunk_keys(request_b, chunk_size)

    print("=== Nano-LMCache: Prefix Cache Demo ===\n")

    print(f"Chunk size: {chunk_size}")
    print(f"Request A: {request_a}")
    print(f"Request B: {request_b}")

    print("\n=== Chunk comparison ===")

    for i, (chunk_a, chunk_b) in enumerate(
        zip(chunks_a, chunks_b, strict=False)
    ):
        hit = keys_a[i] == keys_b[i]
        status = "HIT" if hit else "MISS"

        print(f"Chunk {i}:")
        print(f"  A: {chunk_a}")
        print(f"  B: {chunk_b}")
        print(f"  Key A: {keys_a[i][:16]}...")
        print(f"  Key B: {keys_b[i][:16]}...")
        print(f"  Result: {status}\n")

    # 统计两个请求开头连续匹配的 Chunk 数量。
    shared_chunks = 0

    for key_a, key_b in zip(keys_a, keys_b, strict=False):
        if key_a != key_b:
            break
        shared_chunks += 1

    print("=== Summary ===")
    print(f"Request A chunks: {len(chunks_a)}")
    print(f"Request B chunks: {len(chunks_b)}")
    print(f"Shared prefix chunks: {shared_chunks}")
    print(f"Reusable prefix tokens: {shared_chunks * chunk_size}")


if __name__ == "__main__":
    main()
