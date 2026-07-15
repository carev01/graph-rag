import pytest

from graph_extract import quality_labels
from graph_extract.region_names import is_region
from test_noise_filter import KEEP as NOISE_FILTER_KEEP

TRUE_CASES = [
    # suffix form
    "India West",
    "Central India",  # real Azure region -- was mislisted as a false case upstream
    # prefix form
    "East Asia",
    "West Europe",
    "North Europe",
    # embedded
    "South Central US",
    "Germany West Central",
    "Australia East",
    "UK South",
    "Israel Central",
    # parenthetical
    "Asia Pacific (Malaysia)",
    "Asia Pacific (Tokyo)",
    "US East (N. Virginia)",
    # codes
    "us-east-1",
    "eu-west-2",
    "ap-southeast-4",
    "il-central-1",
    "us-gov-west-1",
    "us-central1",
    "europe-west4",
    "asia-northeast1",
    # vendor-prefixed forms
    "Azure East US",
    "AWS us-east-1",
]


@pytest.mark.parametrize("name", TRUE_CASES)
def test_is_region_true(name: str) -> None:
    assert is_region(name) is True


FALSE_CASES = [
    "immutability",
    "Amazon S3",
    "Azure Blob Storage",
    "soft delete",
    "cross-region copy",
    "AWS Backup",
    "recovery point",
    "retention policy",
]


@pytest.mark.parametrize("name", FALSE_CASES)
def test_is_region_false(name: str) -> None:
    assert is_region(name) is False


def _flatten(value: str | list[str]) -> list[str]:
    """SHOULD_DISTINCT members are `str | list[str]`; normalise to a list of
    forms for iteration."""
    return value if isinstance(value, list) else [value]


def test_keep_guard_noise_filter_keep_terms_are_not_regions() -> None:
    for name in NOISE_FILTER_KEEP:
        assert is_region(name) is False, f"gazetteer shadows KEEP domain term: {name!r}"


def test_keep_guard_should_merge_terms_are_not_regions() -> None:
    for name in quality_labels.SHOULD_MERGE:
        assert is_region(name) is False, f"gazetteer shadows SHOULD_MERGE term: {name!r}"


def test_keep_guard_should_distinct_terms_are_not_regions() -> None:
    for pair in quality_labels.SHOULD_DISTINCT:
        for member in pair:
            for form in _flatten(member):
                assert is_region(form) is False, (
                    f"gazetteer shadows SHOULD_DISTINCT term: {form!r}"
                )
