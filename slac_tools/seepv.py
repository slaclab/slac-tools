"""Return the EPICS record definition for an exact PV name.

Python port of the "seepv" bash script.  Given a full, exact PV name,
locate the IOC serving it (via "findpv") and return that PV's record
definition from the IOC's "$IOC_DATA/<ioc>/iocInfo/IOC.dbDumpRecord"
file.

The PV name must be a complete 1:1 match with an existing PV.  No
wildcards or partial names are accepted.

Author: Chris Myers, 11/16/2026
"""

import os
import re
import shutil
import subprocess

class SeePV:
    WILDCARD_CHARS = ("%", "*")

    def __init__(self):
        """Raises FileNotFoundError if "findpv" cannot be found on "PATH".
        Raises OSError if "IOC_DATA" is not set.
        """

        self.findpv = shutil.which("findpv")
        if not self.findpv:
            raise FileNotFoundError("findpv not found on PATH.")

        self.ioc_data = os.environ.get("IOC_DATA")
        if not self.ioc_data:
            raise OSError("IOC_DATA is not set.")

    def find_iocs(self, pv):
        """Return a list of IOC names serving exactly "pv".

        Runs "findpv -s <pv>", which prints "<ioc> <pvname>" lines for
        substring matches, and keeps only lines whose PV name equals "pv".
        """
        result = subprocess.run(
            [self.findpv, "-s", pv],
            capture_output=True,
            text=True,
            check=True,
        )
        iocs = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[1] == pv:
                iocs.append(fields[0])
        return list(dict.fromkeys(iocs))

    def dump_record_path(self, ioc, ioc_data=None):
        """Return the path to an IOC's "IOC.dbDumpRecord" file."""
        ioc_data = ioc_data if ioc_data is not None else self.ioc_data
        return os.path.join(ioc_data, ioc, "iocInfo", "IOC.dbDumpRecord")

    def extract_record(self, db_lines, pv):
        """Return the record block for "pv" from dbDumpRecord "db_lines".

        Starts at the "record(<type>, "<pv>")" header and tracks brace depth
        until the block closes.  Returns a list of lines (without trailing
        newlines); empty if no matching record is found.
        """
        header = re.compile(r'^record\s*\([^,]*,\s*"' + re.escape(pv) + r'"\s*\)')
        record = []
        in_record = False
        depth = 0
        opened = False
        for line in db_lines:
            line = line.rstrip("\n")
            if not in_record and header.search(line):
                in_record, depth, opened = True, 0, False
            if in_record:
                record.append(line)
                n_open = line.count("{")
                depth += n_open - line.count("}")
                if n_open:
                    opened = True
                if opened and depth <= 0:
                    in_record = False
                    break
        return record

    def see_record(self, pv):
        """Return the record definition given a PV name.
        If a PV is found on more than one IOC, only the record on the first IOC
        returned by findpv is returned.
        Raises ValueError if "pv" is empty or contains wildcard characters.
        Raises LookupError if no IOC is found or if the record is not found in the IOC's
        .dbDumpRecord file.
        Raises FileNotFoundError if the .dbDumpRecord file does not exist.
        Raises subprocess.CalledProcessError if findpv exits with an error.
        """
        if not pv:
            raise ValueError("PV name must be provided.")

        if any(c in pv for c in self.WILDCARD_CHARS):
            raise ValueError(f"PV name '{pv}' contains wildcard characters; exact PV name required.")

        iocs = self.find_iocs(pv)

        if not iocs:
            raise LookupError(f"No PV exactly matching '{pv}' was found.")

        ioc = iocs[0]
        dump = self.dump_record_path(ioc)
        if not os.path.isfile(dump):
            raise FileNotFoundError(f"dbDumpRecord file not found: {dump}")
        with open(dump, errors="replace") as fh:
            record_lines = self.extract_record(fh, pv)

        if not record_lines:
            raise LookupError(f"No record for PV '{pv}' found in {dump}.")
        record = "\n".join(record_lines)
        return record
