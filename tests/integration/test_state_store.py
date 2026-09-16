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


async def test_warmup_lock_excludes_another_session_and_releases(state_store):
    """The real cross-process guarantee: while one worker holds the warm-up lock
    no other session can take it, and it is released on exit. The unit tests use
    a fake, so this is the only thing proving `pg_advisory_lock` is actually
    doing the excluding."""
    other = await asyncpg.connect(state_store._dsn)
    try:
        async with state_store.warmup_lock(timeout=5.0) as held:
            assert held is True
            assert await other.fetchval(
                "SELECT pg_try_advisory_lock(911222334)") is False, \
                "a second worker took the warm-up lock while it was held"
        # released on exit, so the next worker's cold article can proceed
        assert await other.fetchval("SELECT pg_try_advisory_lock(911222334)") is True
        await other.execute("SELECT pg_advisory_unlock(911222334)")
    finally:
        await other.close()


async def test_warmup_lock_times_out_and_proceeds_without_the_lock(state_store):
    """A wedged holder must not stall a multi-week bootstrap. The contract is to
    give up, say so, and yield False -- NOT to block forever and not to raise.
    The caller still runs the article; the duplicates that admits are exact-name
    ones merge_duplicates can clean."""
    other = await asyncpg.connect(state_store._dsn)
    try:
        assert await other.fetchval("SELECT pg_try_advisory_lock(911222334)") is True
        async with state_store.warmup_lock(timeout=0.3, poll=0.05) as held:
            assert held is False, "it must report that it did NOT get the lock"
        # The real holder still holds it: giving up must not disturb it. (Not a
        # test that we skipped our own unlock -- advisory locks are
        # session-scoped, so one session cannot release another's.)
        assert await other.fetchval(
            "SELECT count(*) FROM pg_locks WHERE locktype='advisory' "
            "AND objid=911222334") >= 1
    finally:
        await other.execute("SELECT pg_advisory_unlock(911222334)")
        await other.close()


async def test_warmup_lock_releases_when_the_body_raises(state_store):
    """A failing article must not leave the lock held: every other worker's cold
    articles would block until this process exits.

    This pins the CONTRACT (released by the time the block exits), not the
    explicit unlock statement -- asyncpg's pool reset releases advisory locks
    too, so the contract holds by two mechanisms and this test cannot tell them
    apart. See StateStore.warmup_lock."""
    other = await asyncpg.connect(state_store._dsn)
    try:
        with pytest.raises(RuntimeError, match="article blew up"):
            async with state_store.warmup_lock(timeout=5.0):
                raise RuntimeError("article blew up")
        assert await other.fetchval("SELECT pg_try_advisory_lock(911222334)") is True
        await other.execute("SELECT pg_advisory_unlock(911222334)")
    finally:
        await other.close()
