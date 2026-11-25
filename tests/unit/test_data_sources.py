"""Source text preservation and bounded file-resolution regression tests."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from mqf_research.data.build_event_text_folders import (
    next_open,
    safe_source,
    timing_decision,
    to_text,
)
from mqf_research.data.build_warehouse import utc_if_explicit
from mqf_research.data.download_raw import extract_ex99, submission_records


def test_html_preserves_table_values_and_excludes_hidden_text():
    """Visible guidance-table values survive extraction while hidden metadata stays excluded."""
    html = '<html><head><title>hidden header</title></head><body><p>Next quarter</p><table><tr><td>Revenue</td><td>$100–120 million</td></tr></table><div style="display:none">hidden number 999</div><p>Visible outlook</p></body></html>'
    text = to_text(html)
    assert "Revenue" in text
    assert "$100–120 million" in text
    assert "Visible outlook" in text
    assert "hidden" not in text
    assert "999" not in text


def test_source_resolution_rejects_parent_and_absolute_escape(tmp_path):
    """Both relative traversal and an external absolute reference fail raw-root confinement."""
    raw = tmp_path / "raw"
    raw.mkdir()
    assert safe_source(raw, "sec/ABC/document.txt") == raw / "sec/ABC/document.txt"
    for reference in ("../outside.txt", str(tmp_path / "outside.txt")):
        with pytest.raises(ValueError, match="escapes raw directory"):
            safe_source(raw, reference)


def test_sec_timestamp_parser_requires_explicit_zone():
    """Unzoned SEC timestamps are unverifiable; a known offset converts to UTC exactly."""
    assert utc_if_explicit("2024-01-02T16:05:00") == (None, 0)
    assert utc_if_explicit("2024-01-02") == (None, 0)
    assert utc_if_explicit("2024-01-02T16:05:00-05:00") == ("2024-01-02T21:05:00+00:00", 1)


def test_next_open_respects_weekend_and_dst_transition():
    """Friday after-market news maps to Monday 09:30 Eastern after the DST change."""
    sessions = {"IWV": {"2024-03-08", "2024-03-11"}, "ABC": {"2024-03-08", "2024-03-11"}}
    opening = next_open("2024-03-08", "ABC", sessions, ZoneInfo("America/New_York"))
    assert opening == datetime(2024, 3, 11, 13, 30, tzinfo=timezone.utc)


def test_disagreeing_company_and_benchmark_sessions_fail_closed():
    """A missing company or benchmark session cannot be guessed into an earnings label."""
    sessions = {"IWV": {"2024-03-11"}, "ABC": {"2024-03-12"}}
    with pytest.raises(ValueError, match="disagree"):
        next_open("2024-03-08", "ABC", sessions, ZoneInfo("America/New_York"))


def test_source_cutoff_applies_processing_lag_and_requires_matching_timestamps():
    """Source acceptance, normalized time and safety lag must agree at the cutoff."""
    stamp = "2024-03-11T13:29:59Z"
    filing = {"exact_timestamp_verified": 1, "acceptance_utc": stamp, "acceptance_raw": stamp}
    document = {"acceptance_utc": stamp}
    cutoff = datetime(2024, 3, 11, 13, 30, tzinfo=timezone.utc)
    assert timing_decision(document, filing, cutoff, 1)["status"] == "ELIGIBLE"
    assert timing_decision(document, filing, cutoff, 2)["status"] == "AFTER_CUTOFF"
    filing["acceptance_raw"] = "2024-03-11T13:29:59"
    assert timing_decision(document, filing, cutoff, 1)["status"] == "UNVERIFIED_TIME"


def test_sec_exhibit_extraction_retains_multiple_release_attachments():
    """Distinct EX-99 attachments survive extraction while the parent filing is excluded."""
    filing = "".join(
        [
            "<DOCUMENT><TYPE>8-K\n<TEXT>" + "P" * 300 + "</TEXT></DOCUMENT>",
            "<DOCUMENT><TYPE>EX-99.1\n<FILENAME>release.htm\n<TEXT>"
            + "A" * 300
            + "</TEXT></DOCUMENT>",
            "<DOCUMENT><TYPE>EX-99.2\n<FILENAME>guidance.htm\n<TEXT>"
            + "B" * 300
            + "</TEXT></DOCUMENT>",
        ]
    )
    result = extract_ex99(filing)
    assert [(kind, name) for kind, name, text in result] == [
        ("EX-99.1", "release.htm"),
        ("EX-99.2", "guidance.htm"),
    ]
    assert [len(text) for kind, name, text in result] == [300, 300]


def test_sec_submission_columns_keep_accession_alignment():
    """Missing optional SEC arrays cannot shift identities across filing records."""
    rows = submission_records(
        {
            "accessionNumber": ["a1", "a2"],
            "form": ["8-K", "10-Q"],
            "filingDate": ["2024-01-02", "2024-02-03"],
        }
    )
    assert [row["accessionNumber"] for row in rows] == ["a1", "a2"]
    assert rows[0]["primaryDocument"] == ""
    assert rows[1]["form"] == "10-Q"
