import pytest

from scripts.catalog.movies.rejection import RejectReason, listing_rejection_reason
from tests.catalog.samples import discover_item


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, None),
        ({"poster_path": None}, RejectReason.NO_POSTER),
        ({"overview": ""}, RejectReason.NO_RUSSIAN_OVERVIEW),
        ({"adult": True}, RejectReason.ADULT),
    ],
)
def test_listing_rejection_reason(overrides, expected):
    assert listing_rejection_reason(discover_item(1, **overrides)) is expected
