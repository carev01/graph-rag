import asyncpg
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_cursor_roundtrip(state_store):
    assert await state_store.get_cursor() is None
    await state_store.set_cursor("cur-1")
    assert await state_store.get_cursor() == "cur-1"
    await state_store.set_cursor("cur-2")
    assert await state_store.get_cursor() == "cur-2"


async def test_delivery_dedup(state_store):
    assert await state_store.seen_delivery("sig-abc") is False
    assert await state_store.seen_delivery("sig-abc") is True


async def test_bootstrap_watermarks(state_store):
    await state_store.upsert_bootstrap("vendorA", "wm-5", "id-9", "complete")
    await state_store.upsert_bootstrap("vendorB", "wm-2", "id-3", "complete")
    assert set(await state_store.all_bootstrap_watermarks()) == {"wm-5", "wm-2"}


async def test_dead_letter(state_store):
    before = await state_store.dead_letter_count()
    await state_store.record_dead_letter("{bad json", "ctx")
    assert await state_store.dead_letter_count() == before + 1


async def test_advisory_lock_refuses_a_second_connection(state_store):
    # Cross-process single-flight: while the store holds the advisory lock, a
    # SECOND distinct session must be refused it (pg_try_advisory_lock -> False),
    # and must succeed only after the store releases. (Same-session reentrancy of
    # pg_try_advisory_lock returning True is a Postgres property we deliberately
    # do NOT rely on for single-flight — SyncCore's in-process guard handles the
    # same-process case; this lock is for cross-process mutual exclusion.)
    assert await state_store.try_lock() is True
    other = await asyncpg.connect(state_store._dsn)
    try:
        assert await other.fetchval("SELECT pg_try_advisory_lock(911222333)") is False
        await state_store.unlock()
        assert await other.fetchval("SELECT pg_try_advisory_lock(911222333)") is True
        await other.execute("SELECT pg_advisory_unlock(911222333)")
    finally:
        await other.close()
