"""Plain-text pieces shared by the package's reports."""

from exportspredictor.wording import percent_text


def test_a_change_is_a_signed_whole_percentage():
    assert percent_text(0.123) == "+12%"
    assert percent_text(-0.05) == "-5%"
    assert percent_text(0.0) == "+0%"


def test_a_missing_change_is_not_available():
    assert percent_text(None) == "n/a"
