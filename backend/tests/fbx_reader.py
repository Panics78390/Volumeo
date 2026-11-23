"""Lecteur FBX binaire minimal, utilisé uniquement par les tests pour valider l'écrivain."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

_ARR = {b"d": "<f8", b"f": "<f4", b"i": "<i4", b"l": "<i8", b"b": "u1"}
_SCALAR = {b"Y": "<h", b"C": "<?", b"I": "<i", b"F": "<f", b"D": "<d", b"L": "<q"}


@dataclass
class N:
    name: str
    props: list
    children: list = field(default_factory=list)

    def find(self, name: str) -> "N | None":
        return next((c for c in self.children if c.name == name), None)

    def find_all(self, name: str) -> list["N"]:
        return [c for c in self.children if c.name == name]


def _read_node(buf: bytes, pos: int) -> tuple[N | None, int]:
    end, nprops, _plen = struct.unpack_from("<3I", buf, pos)
    name_len = buf[pos + 12]
    if end == 0:
        return None, pos + 13
    name = buf[pos + 13 : pos + 13 + name_len].decode()
    pos += 13 + name_len
    props = []
    for _ in range(nprops):
        t = buf[pos : pos + 1]
        pos += 1
        if t in _SCALAR:
            fmt = _SCALAR[t]
            props.append(struct.unpack_from(fmt, buf, pos)[0])
            pos += struct.calcsize(fmt)
        elif t in _ARR:
            n, enc, clen = struct.unpack_from("<3I", buf, pos)
            pos += 12
            raw = buf[pos : pos + clen]
            pos += clen
            if enc == 1:
                raw = zlib.decompress(raw)
            arr = np.frombuffer(raw, dtype=_ARR[t])
            assert len(arr) == n, f"taille de tableau incohérente dans {name}"
            props.append(arr)
        elif t in (b"S", b"R"):
            (n,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            data = buf[pos : pos + n]
            pos += n
            props.append(data.decode() if t == b"S" else bytes(data))
        else:
            raise ValueError(f"type de propriété inconnu {t!r}")
    node = N(name, props)
    while pos < end:
        child, pos = _read_node(buf, pos)
        if child is not None:
            node.children.append(child)
    assert pos == end, f"décalage de fin incorrect pour {name}"
    return node, pos


def read_fbx(data: bytes) -> tuple[int, list[N]]:
    assert data[:23] == b"Kaydara FBX Binary\x20\x20\x00\x1a\x00"
    (version,) = struct.unpack_from("<I", data, 23)
    pos = 27
    nodes = []
    while True:
        node, pos = _read_node(data, pos)
        if node is None:
            break
        nodes.append(node)
    return version, nodes
