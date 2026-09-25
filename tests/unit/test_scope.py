import pytest
from answer_api.scope import Scope, ScopeResolver, UnknownScopeName

VENDORS = ["AWS", "Microsoft", "Veeam", "Cohesity"]
PRODUCTS = [("AWS Backup", "AWS"), ("Azure Backup", "Microsoft"),
            ("Microsoft 365 Backup", "Microsoft"),
            ("Veeam Backup & Replication", "Veeam"),
            ("Veeam Backup for Microsoft 365", "Veeam"), ("FortKnox", "Cohesity")]
ALIASES = {"VBR": "Veeam Backup & Replication", "Azure": "Microsoft", "Amazon": "AWS",
           "Ghost": "No Such Product"}


def _r():
    return ScopeResolver(VENDORS, PRODUCTS, ALIASES)


def test_products_and_vendors_are_detected_case_insensitively():
    s = _r().detect("compare aws backup and Azure Backup encryption")
    assert s == Scope((), ("AWS Backup", "Azure Backup"), "detected")


def test_longest_match_wins_so_a_product_is_not_split_into_vendors():
    s = _r().detect("How does Veeam Backup for Microsoft 365 restore mail?")
    assert s.products == ("Veeam Backup for Microsoft 365",) and s.vendors == ()


def test_a_bare_vendor_name_scopes_to_the_vendor():
    assert _r().detect("What does Cohesity offer for ransomware?").vendors == ("Cohesity",)


def test_aliases_resolve_to_canonical_names_and_dangling_aliases_are_ignored():
    assert _r().detect("VBR hardened repository").products == ("Veeam Backup & Replication",)
    assert _r().detect("Azure soft delete").vendors == ("Microsoft",)
    assert _r().detect("Ghost feature").is_empty()


def test_whole_words_only():
    assert _r().detect("aws-style awsome tooling").vendors == ("AWS",)   # "aws" matches, "awsome" does not
    assert _r().detect("backupsome FortKnoxes").is_empty()


def test_cross_vendor_phrasing_naming_nobody_is_unscoped():
    s = _r().detect("How should I plan retention across all vendors?")
    assert s.is_empty() and s.source == "none"


def test_explicit_names_override_detection_and_are_canonicalised():
    s = _r().resolve("Compare AWS Backup and Azure Backup", vendors=["veeam"])
    assert s == Scope(("Veeam",), (), "explicit")


def test_unknown_explicit_name_raises():
    with pytest.raises(UnknownScopeName) as e:
        _r().resolve("q", products=["Nope"])
    assert e.value.names == ["Nope"]


def test_disabled_returns_an_empty_scope():
    assert _r().resolve("Compare AWS Backup and Azure Backup", disabled=True).is_empty()


def test_as_dict_and_vendor_of():
    r = _r()
    assert Scope(("AWS",), ("FortKnox",), "detected").as_dict() == {
        "vendors": ["AWS"], "products": ["FortKnox"], "source": "detected"}
    assert r.vendor_of("FortKnox") == "Cohesity" and r.vendor_of("x") is None


# Name collisions: 11 real vendors (afi.ai, bacula, cloudcasa, druva, eon, gearset,
# grax, keepit, slide, velero, zerto) each have a product of the same name.
COLLISION_VENDORS = VENDORS + ["Keepit"]
COLLISION_PRODUCTS = PRODUCTS + [("Keepit", "Keepit")]


def _rc():
    return ScopeResolver(COLLISION_VENDORS, COLLISION_PRODUCTS, ALIASES)


def test_vendor_wins_a_name_collision_in_detection():
    s = _rc().detect("Does Keepit support X?")
    assert s.vendors == ("Keepit",) and s.products == ()


def test_explicit_vendor_param_resolves_by_declared_kind_on_collision():
    assert _rc().resolve("q", vendors=["keepit"]) == Scope(("Keepit",), (), "explicit")


def test_explicit_product_param_resolves_by_declared_kind_on_collision():
    assert _rc().resolve("q", products=["Keepit"]) == Scope((), ("Keepit",), "explicit")


def test_explicit_vendor_param_accepts_an_alias_of_matching_kind():
    assert _rc().resolve("q", vendors=["Azure"]) == Scope(("Microsoft",), (), "explicit")


def test_explicit_product_param_rejects_an_alias_of_mismatched_kind():
    with pytest.raises(UnknownScopeName) as e:
        _rc().resolve("q", products=["Azure"])
    assert e.value.names == ["Azure"]


@pytest.mark.parametrize("q", [
    "Which backup vendors can protect Azure VMs?",
    "What options exist to back up Amazon RDS across vendors?",
    "Which third-party tools back up AWS Backup vaults?",
])
def test_cross_vendor_wording_stays_unscoped_even_when_names_appear(q):
    """A platform the question is ABOUT (Azure VMs, Amazon RDS) is not the vendor
    whose documentation answers it; narrowing to Microsoft/AWS would drop every
    other vendor's facts from an explicitly cross-vendor question (final review)."""
    assert _r().detect(q).is_empty()


def test_explicit_params_still_override_cross_vendor_wording():
    s = _r().resolve("Which vendors support Azure?", vendors=["Veeam"])
    assert s == Scope(("Veeam",), (), "explicit")


@pytest.mark.parametrize("q", [
    "How do I copy backups across regions in AWS Backup?",
    "How does AWS Backup share recovery points across accounts?",
    "Can Azure Backup restore across subscriptions?",
    "How is data replicated across availability zones in Azure Backup?",
    "How do I configure a third-party KMS key in AWS Backup?",
    "Which vendor-supplied agents does Veeam Backup & Replication need?",
    "What tools does AWS Backup provide for vault lock?",
    "Which options in Azure Backup enable soft delete?",
    "What backup options does Azure Backup offer for SQL?",
    "Which products does Veeam offer for Microsoft 365?",
])
def test_single_vendor_questions_with_everyday_wording_stay_scoped(q):
    """Scoped re-review: 'across regions', 'third-party KMS', 'which options' are
    ordinary single-vendor backup vocabulary, not cross-vendor intent."""
    assert not _r().detect(q).is_empty(), q
