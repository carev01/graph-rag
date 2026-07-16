from itertools import combinations

from graph_extract.vendor_aliases import VENDOR_ALIASES, accepted_forms, normalize


def test_aws_forms():
    forms = accepted_forms("AWS")
    assert "amazon web services" in forms and "aws" in forms


def test_microsoft_forms():
    assert "azure" in accepted_forms("Microsoft")


def test_normalize():
    assert normalize("  Amazon Web Services ") == "amazon web services"


def test_unknown_vendor_forms_is_just_itself():
    assert accepted_forms("Veeam") == {"veeam"}  # no aliases yet -> its own normalized name


def test_no_cross_vendor_collision():
    """No alias form of one vendor may equal the normalized name or any alias
    form of a different vendor in the map -- otherwise reconciliation could
    link a structural node to the wrong vendor's semantic entity."""
    for a, b in combinations(VENDOR_ALIASES, 2):
        assert accepted_forms(a).isdisjoint(accepted_forms(b))
