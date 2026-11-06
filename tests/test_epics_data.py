import datetime
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import requests


# `meme` is an optional runtime dependency of slac_tools.epics_data that is
# not declared in pyproject and typically absent in test environments.
# Install a stub `meme.archive` module so the lazy `from meme.archive import
# get_data_at_time` inside get_archived can resolve; individual tests patch
# `meme.archive.get_data_at_time` to control behavior.
if "meme" not in sys.modules:
    _fake_meme = types.ModuleType("meme")
    _fake_meme_archive = types.ModuleType("meme.archive")
    _fake_meme_archive.get_data_at_time = lambda *a, **kw: {}
    _fake_meme_archive.ArchiveRetrievalError = type(
        "ArchiveRetrievalError", (Exception,), {}
    )
    _fake_meme.archive = _fake_meme_archive
    sys.modules["meme"] = _fake_meme
    sys.modules["meme.archive"] = _fake_meme_archive


from meme.archive import ArchiveRetrievalError


import slac_tools.epics_data
from slac_tools.epics_data import get_archived, get_live


# ---------------------------------------------------------------------------
# get_live
# ---------------------------------------------------------------------------


def test_get_live_all_resolve():
    pvs = ["A", "B", "C"]
    with patch("epics.caget_many", return_value=[1.0, 2.0, 3.0]) as mock:
        result = get_live(pvs)
    mock.assert_called_once_with(pvs)
    assert result == {"A": 1.0, "B": 2.0, "C": 3.0}


def test_get_live_raises_on_none_value():
    pvs = ["A", "B", "C"]
    with patch("epics.caget_many", return_value=[1.0, None, 3.0]):
        with pytest.raises(ValueError, match=r"Could not read 1 PV\(s\).*'B'"):
            get_live(pvs)


def test_get_live_empty_list():
    with patch("epics.caget_many", return_value=[]) as mock:
        result = get_live([])
    mock.assert_called_once_with([])
    assert result == {}


# ---------------------------------------------------------------------------
# get_archived
# ---------------------------------------------------------------------------


def test_get_archived_single_call_default_timeout():
    pvs = ["A", "B"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    expected = {"A": 1.0, "B": 2.0}
    with patch("meme.archive.get_data_at_time", return_value=expected) as mock:
        result = get_archived(pvs, when)
    mock.assert_called_once_with(pvs, at=when, timeout=20)
    assert result == expected


def test_get_archived_custom_timeout_forwarded():
    pvs = ["A"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    with patch("meme.archive.get_data_at_time", return_value={"A": 1.0}) as mock:
        get_archived(pvs, when, timeout=5)
    mock.assert_called_once_with(pvs, at=when, timeout=5)


def test_get_archived_datetime_bypasses_dateparser():
    # Passing a datetime should never cause dateparser to be imported.
    # Simulate dateparser being unavailable; the call should still succeed.
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    with patch.dict(sys.modules, {"dateparser": None}):
        with patch("meme.archive.get_data_at_time", return_value={"A": 1.0}) as mock:
            result = get_archived(["A"], when)
    assert mock.call_args.kwargs["at"] is when
    assert result == {"A": 1.0}


def test_get_archived_batching_merges_results():
    pvs = ["A", "B", "C", "D", "E"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    # batch_n=2 → chunks ["A","B"], ["C","D"], ["E"]
    side_effect = [
        {"A": 1.0, "B": 2.0},
        {"C": 3.0, "D": 4.0},
        {"E": 5.0},
    ]
    with patch(
        "meme.archive.get_data_at_time", side_effect=side_effect
    ) as mock:
        result = get_archived(pvs, when, batch_n=2, timeout=7)
    assert mock.call_count == 3
    for call in mock.call_args_list:
        assert call.kwargs == {"at": when, "timeout": 7}
    chunk_args = [call.args[0] for call in mock.call_args_list]
    assert chunk_args == [["A", "B"], ["C", "D"], ["E"]]
    assert result == {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}


def test_get_archived_retries_on_requests_exception_then_succeeds(caplog):
    pvs = ["A", "B"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    side_effect = [
        requests.ConnectionError("transient"),
        {"A": 1.0, "B": 2.0},
    ]
    caplog.set_level(logging.WARNING, logger="slac_tools.epics_data")
    with patch("meme.archive.get_data_at_time", side_effect=side_effect) as mock:
        result = get_archived(pvs, when, batch_n=2)
    assert mock.call_count == 2
    assert result == {"A": 1.0, "B": 2.0}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "attempt 1/3 failed" in warnings[0].getMessage()


def test_get_archived_retries_on_archive_retrieval_error(caplog):
    pvs = ["A"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    side_effect = [ArchiveRetrievalError("archiver blip"), {"A": 1.0}]
    caplog.set_level(logging.WARNING, logger="slac_tools.epics_data")
    with patch("meme.archive.get_data_at_time", side_effect=side_effect) as mock:
        result = get_archived(pvs, when, batch_n=1)
    assert mock.call_count == 2
    assert result == {"A": 1.0}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_get_archived_all_three_attempts_fail():
    pvs = ["A", "B"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    boom = requests.ConnectionError("archiver down")
    with patch("meme.archive.get_data_at_time", side_effect=boom) as mock:
        with pytest.raises(
            ValueError,
            match=r"Batch 1/1: all 3 attempts failed for 2 PV\(s\): \['A', 'B'\]",
        ) as excinfo:
            get_archived(pvs, when, batch_n=2)
    assert mock.call_count == 3
    assert excinfo.value.__cause__ is boom


def test_get_archived_missing_pv_fails_fast_without_retry():
    pvs = ["A", "B"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    with patch(
        "meme.archive.get_data_at_time", return_value={"A": 1.0, "B": None}
    ) as mock:
        with pytest.raises(
            ValueError,
            match=r"Batch 1/1: no archived value for 1 PV\(s\): \['B'\]",
        ):
            get_archived(pvs, when, batch_n=2)
    assert mock.call_count == 1


def test_get_archived_unrelated_exception_not_retried():
    pvs = ["A", "B"]
    when = datetime.datetime(2024, 1, 15, 10, 30, 0)
    boom = RuntimeError("programming error")
    with patch("meme.archive.get_data_at_time", side_effect=boom) as mock:
        with pytest.raises(RuntimeError, match="programming error"):
            get_archived(pvs, when, batch_n=2)
    assert mock.call_count == 1


def test_get_archived_string_date_time_resolved_once(monkeypatch):
    # Patch _resolve_date_time to count calls and return a fixed datetime,
    # then verify every underlying call receives that same instant.
    resolved = datetime.datetime(2024, 1, 15, 10, 30, 0)
    resolver = MagicMock(return_value=resolved)
    monkeypatch.setattr(slac_tools.epics_data, "_resolve_date_time", resolver)
    pvs = ["A", "B", "C", "D"]
    side_effect = [{"A": 1.0, "B": 2.0}, {"C": 3.0, "D": 4.0}]
    with patch("meme.archive.get_data_at_time", side_effect=side_effect) as mock:
        get_archived(pvs, "1 hour ago", batch_n=2)
    assert resolver.call_count == 1
    assert resolver.call_args.args == ("1 hour ago",)
    for call in mock.call_args_list:
        assert call.kwargs["at"] is resolved
