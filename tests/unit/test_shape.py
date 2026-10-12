"""Question shape and facets (spec §2)."""
import asyncio

import pytest

from answer_api.scope import Scope, ScopeResolver
from answer_api.shape import (Facet, classify_shape, could_not_check_note,
                              group_by_facet, raise_if_cancelled)

_PRODUCTS = [("Veeam Backup & Replication", "Veeam"), ("Veeam ONE", "Veeam"),
             ("Veeam Backup for Microsoft 365", "Veeam"), ("AWS Backup", "AWS"),
             ("Azure Backup", "Microsoft"), ("Cohesity DataProtect", "Cohesity")]
_SOURCES = [("Veeam Agent for Linux", "Veeam Backup & Replication"),
            ("Veeam Agent for Mac", "Veeam Backup & Replication"),
            ("User Guide", "Veeam ONE")]
_ALIASES = {"Veeam Agent": ["Veeam Agent for Linux", "Veeam Agent for Mac"]}


def _resolver(content=("AWS", "Cohesity", "Microsoft", "Veeam")):
    return ScopeResolver(["AWS", "Cohesity", "Microsoft", "Veeam"], _PRODUCTS,
                         _ALIASES, _SOURCES, content_vendors=content)


def _shape(q, **kw):
    r = _resolver()
    return classify_shape(q, r, r.resolve(q), **kw)


def test_single_product_question_is_single_with_the_detected_scope():
    q = "How does Veeam Backup & Replication detect malware?"
    s = _shape(q)
    assert s.kind == "single"
    assert s.facets == (Facet("Veeam Backup & Replication", _resolver().resolve(q)),)


def test_unscoped_question_is_single_with_an_empty_facet():
    s = _shape("How do I restore a file?")
    assert s.kind == "single"
    assert s.facets == (Facet("", Scope()),)


def test_two_named_sides_make_a_comparison_in_question_order():
    s = _shape("Compare how Veeam hardened repositories and AWS Backup Vault Lock "
               "protect backups.")
    assert s.kind == "comparison"
    assert [f.label for f in s.facets] == ["Veeam", "AWS Backup"]
    assert s.facets[0].scope == Scope(vendors=("Veeam",), source="detected")
    assert s.facets[1].scope == Scope(products=("AWS Backup",), source="detected")


def test_a_multi_target_alias_is_one_facet():
    s = _shape("What should I consider for Veeam Agent backups?")
    assert s.kind == "single"
    assert s.facets[0].label == "Veeam Agent"
    assert set(s.facets[0].scope.sources) == {"Veeam Agent for Linux",
                                              "Veeam Agent for Mac"}


def test_a_vendor_mention_is_subsumed_by_its_own_product():
    s = _shape("In Veeam's Veeam ONE, how are alarms approved?")
    assert s.kind == "single"
    assert s.facets[0].label == "Veeam ONE"


def test_two_products_of_one_vendor_are_a_comparison():
    s = _shape("Compare Veeam ONE with Veeam Backup & Replication reporting.")
    assert s.kind == "comparison"
    assert [f.label for f in s.facets] == ["Veeam ONE", "Veeam Backup & Replication"]


def test_cross_vendor_wording_is_an_enumeration_over_content_vendors():
    s = _shape("Which backup vendors support immutable backups?")
    assert s.kind == "enumeration"
    assert [f.label for f in s.facets] == ["AWS", "Cohesity", "Microsoft", "Veeam"]
    assert s.facets[2].scope == Scope(vendors=("Microsoft",), source="detected")


def test_cross_vendor_wording_naming_a_workload_is_still_an_enumeration():
    s = _shape("Which vendors can protect Azure Backup vaults?")
    assert s.kind == "enumeration"


def test_explicit_scope_is_single_whatever_the_wording():
    r = _resolver()
    explicit = r.resolve("Which vendors support immutability?", vendors=["AWS"])
    s = classify_shape("Which vendors support immutability?", r, explicit)
    assert s.kind == "single"
    assert s.facets == (Facet("AWS", explicit),)


def test_scope_disabled_is_single_unscoped():
    r = _resolver()
    s = classify_shape("Compare Veeam and AWS Backup", r, Scope(), scope_disabled=True)
    assert s == classify_shape("x", r, Scope(), scope_disabled=True)
    assert s.kind == "single" and s.facets == (Facet("", Scope()),)


def test_enumeration_without_content_vendors_falls_back_to_single():
    r = _resolver(content=())
    s = classify_shape("Which vendors support immutability?", r, Scope())
    assert s.kind == "single"


def test_vendor_of_source():
    r = _resolver()
    assert r.vendor_of_source("Veeam Agent for Linux") == "Veeam"
    assert r.vendor_of_source("nope") is None


def test_group_by_facet_assigns_each_item_to_its_first_matching_facet():
    facets = [Facet("Veeam", Scope(vendors=("Veeam",))),
              Facet("AWS Backup", Scope(products=("AWS Backup",)))]
    items = [{"id": 1, "sources": [{"vendor": "AWS", "product": "AWS Backup"}]},
             {"id": 2, "sources": [{"vendor": "Veeam", "product": "Veeam ONE"}]},
             {"id": 3, "sources": [{"vendor": "Cohesity", "product": "X"}]}]
    groups = group_by_facet(items, facets)
    assert [(f.label, [i["id"] for i in xs]) for f, xs in groups] == [
        ("Veeam", [2]), ("AWS Backup", [1])]


def test_could_not_check_note():
    assert could_not_check_note([]) == ""
    assert could_not_check_note(["Cohesity", "AWS"]) == "Could not check: Cohesity, AWS."


def test_raise_if_cancelled_reraises_a_cancellation_among_results():
    with pytest.raises(asyncio.CancelledError):
        raise_if_cancelled([{"ok": 1}, RuntimeError("x"), asyncio.CancelledError()])


def test_raise_if_cancelled_leaves_ordinary_exceptions_and_values_alone():
    raise_if_cancelled([{"ok": 1}, RuntimeError("x"), None])
    raise_if_cancelled([])
