from answer_api.attribution import applies_to, fact_line, in_scope, label, pairs_of
from answer_api.scope import Scope

V = {"vendor": "Veeam", "product": "Veeam Backup & Replication", "url": "u1"}
V2 = {"vendor": "Veeam", "product": "Veeam Backup & Replication", "url": "u2"}
C = {"vendor": "Commvault", "product": "Commvault Cloud", "url": "u3"}
NONE = {"vendor": None, "product": None, "url": "u4"}

def test_pairs_are_ordered_unique_and_skip_unlabelled_sources():
    assert pairs_of([V, V2, C, NONE]) == [("Veeam", "Veeam Backup & Replication"),
                                           ("Commvault", "Commvault Cloud")]

def test_label_format_is_exact():
    assert label([V, C]) == "(Veeam · Veeam Backup & Replication; Commvault · Commvault Cloud)"
    assert label([NONE]) == "" and label([]) == ""

def test_fact_line_with_and_without_label():
    assert fact_line(3, "Immutability lasts 7 days.", [V]) == \
        "[3] (Veeam · Veeam Backup & Replication) Immutability lasts 7 days."
    assert fact_line(4, "x", []) == "[4] x"

def test_applies_to_counts_facts_per_vendor_not_sources():
    cits = [{"sources": [V, V2]}, {"sources": [V, C]}, {"sources": [NONE]}]
    assert applies_to(cits) == [
        {"vendor": "Veeam", "products": ["Veeam Backup & Replication"], "facts": 2},
        {"vendor": "Commvault", "products": ["Commvault Cloud"], "facts": 1}]

def test_in_scope_matches_vendor_or_product_and_empty_scope_matches_all():
    assert in_scope([V], Scope(("Veeam",), (), "x"))
    assert in_scope([C], Scope((), ("Commvault Cloud",), "x"))
    assert not in_scope([C], Scope(("Veeam",), (), "x"))
    assert in_scope([C], Scope())
