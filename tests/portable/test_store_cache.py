"""Tests for the Zarr 3 encoded-store cache."""

import asyncio
from collections import Counter

import fsspec
import numpy as np
import zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.codecs import Shuffle, ZstdCodec
from zarr.core.buffer import default_buffer_prototype

from src.map_data.store_cache import (
    GIBIBYTE,
    MEBIBYTE,
    CachedStore,
    EncodedStoreCache,
    store_cache_size,
)


class CountingStore(zarr.storage.WrapperStore):
    def __init__(self, store) -> None:
        super().__init__(store)
        self.gets = Counter()

    async def get(self, key, prototype, byte_range=None):
        self.gets[(key, repr(byte_range))] += 1
        return await self._store.get(key, prototype, byte_range)


def _memory_store_with_payload(key="payload", value=b"abcdef"):
    prototype = default_buffer_prototype()
    store = zarr.storage.MemoryStore()
    asyncio.run(store.set(key, prototype.buffer.from_bytes(value)))
    return store.with_read_only(True), prototype


def test_cached_store_reuses_full_range_and_missing_reads():
    memory_store, prototype = _memory_store_with_payload()
    counting_store = CountingStore(memory_store)
    cache = EncodedStoreCache(1024)
    store = CachedStore(counting_store, cache)

    async def read_values():
        full_first = await store.get("payload", prototype)
        full_second = await store.get("payload", prototype)
        byte_requests = (
            (RangeByteRequest(1, 4), b"bcd"),
            (OffsetByteRequest(2), b"cdef"),
            (SuffixByteRequest(3), b"def"),
        )
        range_values = []
        for byte_request, expected in byte_requests:
            first = await store.get("payload", prototype, byte_request)
            second = await store.get("payload", prototype, byte_request)
            range_values.append((first, second, expected))
        missing_first = await store.get("missing", prototype)
        missing_second = await store.get("missing", prototype)
        missing_exists = await store.exists("missing")
        partial_values = await store.get_partial_values(
            prototype,
            [("payload", byte_request) for byte_request, _ in byte_requests] + [("missing", None)],
        )
        many_values = [
            value
            async for _, value in store._get_many(
                [("payload", prototype, None), ("missing", prototype, None)],
            )
        ]
        return (
            full_first,
            full_second,
            range_values,
            missing_first,
            missing_second,
            missing_exists,
            partial_values,
            many_values,
        )

    (
        full_first,
        full_second,
        range_values,
        missing_first,
        missing_second,
        missing_exists,
        partial_values,
        many_values,
    ) = asyncio.run(read_values())

    assert full_first.to_bytes() == full_second.to_bytes() == b"abcdef"
    for first, second, expected in range_values:
        assert first.to_bytes() == second.to_bytes() == expected
    assert missing_first is missing_second is None
    assert missing_exists is False
    assert [value.to_bytes() for value in partial_values[:-1]] == [b"bcd", b"cdef", b"def"]
    assert partial_values[-1] is None
    assert many_values[0].to_bytes() == b"abcdef"
    assert many_values[1] is None
    assert sum(counting_store.gets.values()) == 5
    assert cache.entry_count == 5
    assert cache.used == 17


def test_cached_store_reads_copick_style_multichunk_shard(tmp_path):
    path = tmp_path / "copick-style.zarr"
    expected = np.arange(8**3, dtype=np.float32).reshape((8, 8, 8))
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    array = root.create_array(
        "0",
        shape=expected.shape,
        chunks=(2, 2, 2),
        shards=(8, 8, 8),
        dtype=expected.dtype,
        dimension_names=("z", "y", "x"),
        compressors=(Shuffle(elementsize=expected.dtype.itemsize), ZstdCodec(level=3)),
        chunk_key_encoding={"name": "v2", "separator": "/"},
    )
    array[:] = expected

    filesystem = fsspec.filesystem("file", asynchronous=True)
    source = zarr.storage.FsspecStore.from_mapper(filesystem.get_mapper(str(path)), read_only=True)
    counting_store = CountingStore(source)
    cache = EncodedStoreCache(MEBIBYTE)
    group = zarr.open_group(store=CachedStore(counting_store, cache), mode="r")
    selection = np.s_[2:6, 2:6, 2:6]

    np.testing.assert_array_equal(group["0"][selection], expected[selection])
    reads_after_first_slice = sum(counting_store.gets.values())
    np.testing.assert_array_equal(group["0"][selection], expected[selection])

    assert reads_after_first_slice > 0
    assert sum(counting_store.gets.values()) == reads_after_first_slice


def test_encoded_store_cache_invalidates_keys_prefixes_and_namespaces():
    cache = EncodedStoreCache(1024)
    namespace = object()
    other_namespace = object()
    byte_request = RangeByteRequest(1, 4)
    root_metadata = (namespace, "root/zarr.json", None)
    chunk_range = (namespace, "root/0/0/0", byte_request)
    unrelated = (namespace, "other/zarr.json", None)
    other_store = (other_namespace, "root/0/0/0", byte_request)
    for cache_key in (root_metadata, chunk_range, unrelated, other_store):
        cache.store(cache_key, b"value")

    cache.invalidate_key(namespace, "root/zarr.json")
    assert cache.lookup(root_metadata) == (False, None)
    assert cache.lookup(chunk_range) == (True, b"value")

    cache.invalidate_prefix(namespace, "root/")
    assert cache.lookup(chunk_range) == (False, None)
    assert cache.lookup(unrelated) == (True, b"value")
    assert cache.lookup(other_store) == (True, b"value")

    cache.invalidate_namespace(namespace)
    assert cache.lookup(unrelated) == (False, None)
    assert cache.lookup(other_store) == (True, b"value")


def test_encoded_store_cache_evicts_lru_and_skips_oversized_values():
    cache = EncodedStoreCache(5)
    namespace = object()
    first = (namespace, "first", None)
    second = (namespace, "second", None)
    third = (namespace, "third", None)

    cache.store(first, b"123")
    cache.store(second, b"45")
    assert cache.lookup(first) == (True, b"123")
    cache.store(third, b"abc")

    assert cache.lookup(second) == (False, None)
    assert cache.lookup(first) == (False, None)
    assert cache.lookup(third) == (True, b"abc")
    cache.store(first, b"123456")
    assert cache.lookup(first) == (False, None)
    assert cache.used == 3


def test_store_cache_size_is_adaptive_and_bounded():
    assert store_cache_size(128 * MEBIBYTE) == 256 * MEBIBYTE
    assert store_cache_size(16 * GIBIBYTE) == 2 * GIBIBYTE
    assert store_cache_size(64 * GIBIBYTE) == 4 * GIBIBYTE
