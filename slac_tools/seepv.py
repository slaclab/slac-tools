"""Print the EPICS record definition for an exact PV name.

Python port of the "seepv" bash script.  Given a full, exact PV name,
locate the IOC(s) serving it (via "findpv") and print that PV's record
definition from each IOC's "$IOC_DATA/<ioc>/iocInfo/IOC.dbDumpRecord"
file.

The PV name must be a complete 1:1 match with an existing PV.  No
wildcards or partial names are accepted.

Usage:
    python seepv.py PVNAME

Author: Chris Myers, 11/16/2026
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

WILDCARD_CHARS = ("%", "*")


def find_iocs(pv, findpv=None):
    """Return a sorted list of IOC names serving exactly "pv".

    Runs "findpv -s <pv>", which prints "<ioc> <pvname>" lines for
    substring matches, and keeps only lines whose PV name equals "pv".

    Raises FileNotFoundError if "findpv" cannot be found on "PATH".
    """
    findpv = findpv or shutil.which("findpv")
    if not findpv:
        raise FileNotFoundError("findpv not found on PATH.")
    result = subprocess.run(
        [findpv, "-s", pv],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    iocs = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == pv:
            iocs.add(fields[0])
    return sorted(iocs)


def dump_record_path(ioc, ioc_data=None):
    """Return the path to an IOC's "IOC.dbDumpRecord" file."""
    ioc_data = ioc_data if ioc_data is not None else os.environ.get("IOC_DATA", "")
    return os.path.join(ioc_data, ioc, "iocInfo", "IOC.dbDumpRecord")


def extract_record(lines, pv):
    """Return the record block for "pv" from dbDumpRecord "lines".

    Starts at the "record(<type>, "<pv>")" header and tracks brace depth
    until the block closes.  Returns a list of lines (without trailing
    newlines); empty if no matching record is found.
    """
    header = re.compile(r'^record\s*\([^,]*,\s*"' + re.escape(pv) + r'"\s*\)')
    out = []
    in_rec = False
    depth = 0
    opened = False
    for line in lines:
        line = line.rstrip("\n")
        if not in_rec and header.search(line):
            in_rec, depth, opened = True, 0, False
        if in_rec:
            out.append(line)
            n_open = line.count("{")
            depth += n_open - line.count("}")
            if n_open:
                opened = True
            if opened and depth <= 0:
                in_rec = False
    return out


def _bold(text):
    if sys.stdout.isatty():
        return f"\033[1m{text}\033[0m"
    return text


def build_parser():
    parser = argparse.ArgumentParser(
        prog="seepv",
        description="Print the EPICS record definition for an exact PV name.",
        epilog="Note: PVNAME must be a full, exact match "
        "(no wildcards or partial names).",
    )
    parser.add_argument("pv", metavar="PVNAME", nargs="?", help="exact PV name")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pv = args.pv

    if not pv:
        print("Invalid PV name. See help (-h).")
        return 1

    if any(c in pv for c in WILDCARD_CHARS):
        print("\nseepv requires a full, exact PV name (no wildcards).\n")
        return 1

    try:
        iocs = find_iocs(pv)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if not iocs:
        print(f'\nNo PV exactly matching "{pv}" was found.\n')
        return 1

    found_any = False
    for ioc in iocs:
        print(f"\n{_bold(ioc)}:")
        dump = dump_record_path(ioc)
        if not os.path.isfile(dump):
            print(f"  (dbDumpRecord file not found: {dump})")
            continue

        print(f"  # from: {dump}")
        with open(dump, errors="replace") as fh:
            record = extract_record(fh, pv)

        if not record:
            print(f"  (PV is served by {ioc} but no record was found in {dump}.)")
            continue
        print("\n".join(record))
        found_any = True

    print()
    return 0 if found_any else 1


if __name__ == "__main__":
    sys.exit(main())
