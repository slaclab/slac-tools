"""Live and archived EPICS data access, independent of any simulation model.

This module is a landing spot for simulation-agnostic utilities that read
real control-system data.  Expect it to be absorbed into a broader
live-modeling module later; for now, keep it narrow and dependency-light.
"""

from datetime import datetime


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


def get_pv_data(pvs, date_time=None, batch_n=0):
    """Fetch EPICS values for a list of PV names.

    By default reads live values from the control system.  If *date_time* is
    provided, archived values at that point in time are fetched from the LCLS
    Archiver Appliance via ``meme.archive.get_data_at_time`` instead.

    Parameters
    ----------
    pvs : list of str
        Full EPICS PV names, e.g. ``['QUAD:IN20:361:BDES', 'LCAV:IN20:2:ADES']``.
    date_time : str or datetime, optional
        If provided, fetch archived values at this point in time instead of
        reading live EPICS values.  Accepts the same formats as
        ``meme.archive.get_data_at_time``: a string like
        ``'2024-01-15T10:30:00'`` or ``'1 hour ago'``, or a Python datetime
        object.  Relative strings are resolved to one absolute instant
        up front and reused for every chunk/retry, so a batched request
        cannot combine values from different historical instants.
    batch_n : int, optional
        Only used when *date_time* is provided (archive mode).  If greater
        than 0, the PV list is split into chunks of this size and each chunk
        is fetched in a separate ``get_data_at_time`` call.  Each chunk is
        retried up to 3 times on failure (exception raised or any PV missing
        from the response).  A message is printed to stdout after each failed
        attempt.  If all 3 attempts for a chunk fail, a ``ValueError`` is
        raised listing the affected PVs and chunk index.  When ``batch_n``
        is 0 (the default) or *date_time* is ``None``, all PVs are fetched
        in a single call with no retry logic.

    Returns
    -------
    dict
        Mapping of PV name → value, e.g.
        ``{'QUAD:IN20:361:BDES': 3.5, 'LCAV:IN20:2:ADES': 65.0}``.

    Raises
    ------
    ValueError
        If any PV could not be read in live mode (timed out or does not
        exist), or if a batch chunk still has failing PVs after 3 retries
        in archive mode.
    meme.archive.ArchiveRetrievalError
        If the archiver request fails in archive mode (no-batch path).
    """
    if date_time is not None:
        from meme.archive import get_data_at_time

        at = _resolve_date_time(date_time)

        if batch_n <= 0:
            return get_data_at_time(pvs, at=at, timeout=20)

        # Split into chunks and fetch with retry logic.
        chunks = [pvs[i:i + batch_n] for i in range(0, len(pvs), batch_n)]
        n_chunks = len(chunks)
        result = {}
        for chunk_idx, chunk in enumerate(chunks):
            max_attempts = 3
            last_error = None
            last_failed = None
            for attempt in range(1, max_attempts + 1):
                try:
                    chunk_result = get_data_at_time(chunk, at=at, timeout=20)
                    failed = [pv for pv in chunk if chunk_result.get(pv) is None]
                    if failed:
                        last_failed = failed
                        raise ValueError(
                            f"Missing {len(failed)} PV(s) in archive response: {failed}"
                        )
                    result.update(chunk_result)
                    last_error = None
                    last_failed = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < max_attempts:
                        print(
                            f"Batch {chunk_idx + 1}/{n_chunks}: attempt {attempt}/{max_attempts} "
                            f"failed ({exc}), retrying..."
                        )
            if last_error is not None:
                failed_pvs = last_failed if last_failed is not None else chunk
                raise ValueError(
                    f"Batch {chunk_idx + 1}/{n_chunks}: all {max_attempts} attempts failed "
                    f"for {len(failed_pvs)} PV(s): {failed_pvs}"
                ) from last_error
        return result

    import epics
    values = epics.caget_many(pvs)
    failed = [pv for pv, val in zip(pvs, values) if val is None]
    if failed:
        raise ValueError(
            f"Could not read {len(failed)} PV(s): {failed}"
        )
    return dict(zip(pvs, values))
