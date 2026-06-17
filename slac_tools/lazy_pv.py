from epics import PV


class LazyPV:
    """Wraps an EPICS PV name and defers connection until first use."""

    def __init__(self, pvname: str):
        self._pvname = pvname
        self._pv = None

    @property
    def pvname(self) -> str:
        return self._pvname

    def _ensure_connected(self) -> PV:
        if self._pv is None:
            self._pv = PV(self._pvname)
        return self._pv

    def get(self, *args, **kwargs):
        return self._ensure_connected().get(*args, **kwargs)

    def put(self, *args, **kwargs):
        return self._ensure_connected().put(*args, **kwargs)

    def get_ctrlvars(self, *args, **kwargs):
        return self._ensure_connected().get_ctrlvars(*args, **kwargs)

    def add_callback(self, *args, **kwargs):
        return self._ensure_connected().add_callback(*args, **kwargs)

    def remove_callback(self, *args, **kwargs):
        return self._ensure_connected().remove_callback(*args, **kwargs)

    def disconnect(self):
        if self._pv is not None:
            self._pv.disconnect()
            self._pv = None

    def __eq__(self, other):
        if isinstance(other, LazyPV):
            return self._pvname == other._pvname
        return NotImplemented

    def __hash__(self):
        return hash(self._pvname)

    def __repr__(self):
        connected = self._pv.connected if self._pv else False
        return f"LazyPV({self._pvname!r}, connected={connected})"
