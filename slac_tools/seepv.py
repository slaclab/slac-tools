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

    def __init__(self, pv=None):
        """Optionally store a default PV name for the methods to use.

        If "pv" is given, methods called without a PV use it.  A PV passed
        directly to a method always takes precedence.

        Raises ValueError if "pv" is given but empty or contains wildcard characters.
        Raises FileNotFoundError if "findpv" cannot be found on "PATH".
        Raises OSError if "IOC_DATA" is not set.
        """

        self.pv = pv
        # Validate PV name if given
        if pv is not None:
            self._resolve_pv(pv)

        self.findpv = shutil.which("findpv")
        if not self.findpv:
            raise FileNotFoundError("findpv not found on PATH.")

        self.ioc_data = os.environ.get("IOC_DATA")
        if not self.ioc_data:
            raise OSError("IOC_DATA is not set.")

    def _resolve_pv(self, pv):
        """Return "pv" if given, else the instance's PV, after validating it.

        Raises ValueError if no PV is available or it contains wildcards.
        """
        pv = pv if pv is not None else self.pv
        if not pv:
            raise ValueError("PV name must be provided, either to the method or to SeePV(pv).")
        if any(c in pv for c in self.WILDCARD_CHARS):
            raise ValueError(f"PV name '{pv}' contains wildcard characters; exact PV name required.")
        return pv

    def find_iocs(self, pv=None):
        """Return a list of IOC names serving exactly "pv".

        Runs "findpv -s <pv>", which prints "<ioc> <pvname>" lines for
        substring matches, and keeps only lines whose PV name equals "pv".
        Uses the instance's PV if "pv" is not given.
        """
        pv = self._resolve_pv(pv)
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

    def extract_record(self, db_lines, pv=None):
        """Return the record block for "pv" from dbDumpRecord "db_lines".

        Starts at the "record(<type>, "<pv>")" header and tracks brace depth
        until the block closes.  Returns a list of lines (without trailing
        newlines); empty if no matching record is found.
        Uses the instance's PV if "pv" is not given.
        """
        pv = self._resolve_pv(pv)
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

    def see_record(self, pv=None):
        """Return the record definition given a PV name.
        Uses the instance's PV if "pv" is not given.
        If a PV is found on more than one IOC, only the record on the first IOC
        returned by findpv is returned.
        Raises ValueError if no PV is given or it contains wildcard characters.
        Raises LookupError if no IOC is found or if the record is not found in the IOC's
        .dbDumpRecord file.
        Raises FileNotFoundError if the .dbDumpRecord file does not exist.
        Raises subprocess.CalledProcessError if findpv exits with an error.
        """
        pv = self._resolve_pv(pv)

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

    def see_fields(self, pv=None):
        """Return the PV's record fields as a {field_name: value} dict.

        Uses the instance's PV if "pv" is not given.  Values are returned as
        strings exactly as they appear in the .dbDumpRecord file.  The record
        header and info() lines are skipped.

        Raises the same exceptions as see_record.
        """
        field_re = re.compile(r'^\s*field\(\s*(\w+)\s*,\s*"(.*)"\s*\)\s*$')
        record = self.see_record(pv)
        return {
            m[1]: m[2]
            for line in record.splitlines()
            if (m := field_re.match(line))
        }
