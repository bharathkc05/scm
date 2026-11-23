"""Outbound-network guard for the server process (lightweight, in-process).

The server needs no network: stdio carries the protocol and every query is local. Any attempt to resolve or
connect to a non-loopback address raises NetworkBlocked and is counted. Loopback stays allowed because
asyncio on Windows builds its internal socketpair over 127.0.0.1.
"""
import ipaddress
import socket

_state = {"active": False, "blocked": 0}
_orig: dict = {}


class NetworkBlocked(PermissionError):
    pass


def _is_loopback(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode(errors="ignore")
    host = str(host).strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False          # hostnames other than localhost are not resolved, just refused


def _block(target) -> None:
    _state["blocked"] += 1
    raise NetworkBlocked(f"outbound network access to {target!r} is blocked in the local MCP server")


def install() -> None:
    if _state["active"]:
        return
    _orig.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex,
                 getaddrinfo=socket.getaddrinfo)

    def connect(self, address):
        if isinstance(address, tuple) and not _is_loopback(address[0]):
            _block(address)
        return _orig["connect"](self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple) and not _is_loopback(address[0]):
            _block(address)
        return _orig["connect_ex"](self, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_loopback(host):
            _block(host)
        return _orig["getaddrinfo"](host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    _state["active"] = True


def uninstall() -> None:
    """For tests only."""
    if not _state["active"]:
        return
    socket.socket.connect = _orig["connect"]
    socket.socket.connect_ex = _orig["connect_ex"]
    socket.getaddrinfo = _orig["getaddrinfo"]
    _state["active"] = False


def state() -> dict:
    return {"active": _state["active"], "blocked_attempts": _state["blocked"]}
