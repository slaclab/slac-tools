"""Live and archived EPICS data access, independent of any simulation model.

This module is a landing spot for simulation-agnostic utilities that read
real control-system data.  Expect it to be absorbed into a broader
live-modeling module later; for now, keep it narrow and dependency-light.
"""

import ast
import logging
import re
from datetime import datetime

log = logging.getLogger(__name__)


def _resolve_date_time(date_time):
    """Resolve a date_time arg to a single absolute ``datetime``.

    Accepts the same formats as ``meme.archive.get_data_at_time``:
    an ISO-ish string, a relative string like ``'1 hour ago'``, or
    a ``datetime``.  Returns a ``datetime`` so every downstream
    archive call shares one absolute instant (no drift across
    chunks or retries for relative strings).

    Raises
    ------
    ValueError
        If the string cannot be parsed, or if ``dateparser`` is not
        installed and the input is a string.
    """
    if isinstance(date_time, datetime):
        return date_time
    try:
        import dateparser
    except ImportError as exc:
        raise ValueError(
            f"date_time={date_time!r} is a string, but dateparser is "
            "not installed; install dateparser or pass a datetime."
        ) from exc
    resolved = dateparser.parse(str(date_time))
    if resolved is None:
        raise ValueError(f"Could not parse date_time={date_time!r}")
    return resolved


def get_live(pvs):
    """Read current live values for a list of EPICS PV names via Channel Access.

    Parameters
    ----------
    pvs : list of str
        Full EPICS PV names, e.g. ``['QUAD:IN20:361:BDES', 'LCAV:IN20:2:ADES']``.

    Returns
    -------
    dict
        Mapping of PV name → value, e.g.
        ``{'QUAD:IN20:361:BDES': 3.5, 'LCAV:IN20:2:ADES': 65.0}``.

    Raises
    ------
    ValueError
        If any PV could not be read (timed out or does not exist).
    """
    import epics
    values = epics.caget_many(pvs)
    failed = [pv for pv, val in zip(pvs, values) if val is None]
    if failed:
        raise ValueError(
            f"Could not read {len(failed)} PV(s): {failed}"
        )
    return dict(zip(pvs, values))


def _missing_pvs_from_error(exc):
    """Return the PV names listed in meme's missing-PV error, or ``[]``.

    ``meme.archive.get_data_at_time`` discards the whole response when any
    PV is absent, and only reports which ones in the exception message.
    """
    match = re.search(r"not found in archiver response: (\[.*\])\s*$", str(exc))
    if match is None:
        return []
    return ast.literal_eval(match.group(1))


def _get_data_allow_missing(get_data_at_time, error_type, pvs, at, timeout):
    """Call ``get_data_at_time``, re-requesting without PVs the archiver lacks.

    Returns a mapping of every PV in *pvs* → value, with ``None`` for PVs
    that are not archived.  Errors that are not about missing PVs propagate.
    """
    remaining = list(pvs)
    found = {}
    while remaining:
        try:
            found = get_data_at_time(remaining, at=at, timeout=timeout)
            break
        except error_type as exc:
            missing = set(_missing_pvs_from_error(exc)) & set(remaining)
            if not missing:
                raise
            remaining = [pv for pv in remaining if pv not in missing]
    return {pv: found.get(pv) for pv in pvs}


def get_archived(pvs, date_time, batch_n=0, timeout=20, allow_missing=False):
    """Fetch archived EPICS values at a single point in time.

    Reads from the LCLS Archiver Appliance via
    ``meme.archive.get_data_at_time``.  Relative strings in *date_time*
    are resolved once to one absolute instant and reused across every
    chunk/retry, so a batched request cannot combine values from
    different historical instants.

    Parameters
    ----------
    pvs : list of str
        Full EPICS PV names.
    date_time : str or datetime
        The instant to fetch values at.  Accepts the same formats as
        ``meme.archive.get_data_at_time``: a string like
        ``'2024-01-15T10:30:00'`` or ``'1 hour ago'``, or a Python
        datetime object.
    batch_n : int, optional
        If greater than 0, the PV list is split into chunks of this size
        and each chunk is fetched in a separate ``get_data_at_time`` call.
        Each chunk is retried up to 3 times on
        ``ArchiveRetrievalError`` or ``requests.RequestException``;
        a warning is logged after each failed attempt.  Other exceptions
        propagate immediately.  When ``batch_n`` is 0 (the default), all
        PVs are fetched in a single call with no retry logic.
    timeout : float, optional
        Per-call timeout forwarded to ``get_data_at_time`` (default 20).
    allow_missing : bool, optional
        If True, PVs with no archived value are returned as ``None``
        instead of raising.  Other archiver
        failures still raise.  Default False.

    Returns
    -------
    dict
        Mapping of PV name → value.  With ``allow_missing=True``, PVs with
        no archived value map to ``None``.

    Raises
    ------
    ValueError
        If a batch chunk fails all 3 retries, or if the archiver
        responds successfully but any requested PV has no archived
        value (short-circuits — missing-PV failures are not retried).
        Not raised for missing PVs when ``allow_missing`` is True.
    meme.archive.archive.ArchiveRetrievalError
        If the archiver request fails in the no-batch path.
    """
    import requests
    # ArchiveRetrievalError is not in meme.archive's __all__, so import it from the submodule
    from meme.archive.archive import ArchiveRetrievalError, get_data_at_time

    at = _resolve_date_time(date_time)

    def _fetch(chunk):
        if allow_missing:
            return _get_data_allow_missing(
                get_data_at_time, ArchiveRetrievalError, chunk, at, timeout
            )
        return get_data_at_time(chunk, at=at, timeout=timeout)

    if batch_n <= 0:
        result = _fetch(pvs)
        if allow_missing:
            return result
        missing = [pv for pv in pvs if result.get(pv) is None]
        if missing:
            raise ValueError(
                f"no archived value for {len(missing)} PV(s): {missing}"
            )
        return result

    def _fetch_chunk(chunk, label):
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                chunk_result = _fetch(chunk)
                break
            except (ArchiveRetrievalError, requests.RequestException) as exc:
                if attempt == max_attempts:
                    raise ValueError(
                        f"{label}: all {max_attempts} attempts failed "
                        f"for {len(chunk)} PV(s): {chunk}"
                    ) from exc
                log.warning(
                    "%s: attempt %d/%d failed (%s), retrying...",
                    label, attempt, max_attempts, exc,
                )
        if allow_missing:
            return chunk_result
        missing = [pv for pv in chunk if chunk_result.get(pv) is None]
        if missing:
            raise ValueError(
                f"{label}: no archived value for {len(missing)} PV(s): {missing}"
            )
        return chunk_result

    chunks = [pvs[i:i + batch_n] for i in range(0, len(pvs), batch_n)]
    n_chunks = len(chunks)
    result = {}
    for idx, chunk in enumerate(chunks, start=1):
        result.update(_fetch_chunk(chunk, label=f"Batch {idx}/{n_chunks}"))
    return result
