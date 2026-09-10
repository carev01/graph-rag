"""A community can never be written as verified AND staged in the same rebuild.

write_communities writes the verified rows first (each with an embedding) and then
`SET c +=` the staged ones. A cid in both maps would keep the embedding on the
staged node -- and the absent embedding is the ONLY thing that hides a staged
report from shortlist_communities and from DRIFT."""
import pytest

from theme_builder.report import CommunityReport
from theme_builder.writeback import write_communities

REP = CommunityReport(title="T", summary="S", full_report="[]", rating=1.0,
                      rating_explanation="", tags=[], cited_fact_uuids=[])


@pytest.mark.asyncio
async def test_a_cid_in_both_reports_and_pending_is_refused():
    with pytest.raises(AssertionError, match="verified and staged"):
        # driver/embedder are never reached: the assert is the first statement.
        await write_communities(None, None, "g", [], {"c1": REP},
                                corpus_cursor=None, pending={"c1": REP})
