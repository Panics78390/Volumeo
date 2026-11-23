"""Écrivain FBX binaire 7.4 minimal, sans dépendance (ni SDK Autodesk, ni Blender).

Le fichier contient un seul maillage (normales et UV par sommet de polygone),
un matériau Phong et sa texture diffuse **embarquée** (PNG dans le nœud
``Video.Content``) : un seul fichier à importer dans Blender, Unity, Unreal,
Maya ou 3ds Max.

Références du format binaire : la documentation de Blender
(``io_scene_fbx/encode_bin.py``) et le billet « FBX binary file format
specification » de Blender developer blog. Version 7400 : métadonnées de nœud
sur 3 × uint32 et sentinelle de fin de bloc de 13 octets nuls.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

FBX_VERSION = 7400
_HEAD_MAGIC = b"Kaydara FBX Binary\x20\x20\x00\x1a\x00"
# Valeurs fixes utilisées par Blender : le couple FileId/CreationTime/pied de page
# est vérifié par certaines versions du SDK FBX.
_FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
_TIME_ID = "1970-01-01 10:00:00:000"
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
_SENTINEL = b"\x00" * 13
_SEP = "\x00\x01"  # séparateur nom/classe dans les noms d'objets binaires


# ---------------------------------------------------------------------------
# Encodage bas niveau
# ---------------------------------------------------------------------------
class Arr:
    """Tableau typé FBX : 'd' float64, 'f' float32, 'i' int32, 'l' int64, 'b' bool."""

    _dtypes = {"d": "<f8", "f": "<f4", "i": "<i4", "l": "<i8", "b": "u1"}

    def __init__(self, code: str, values: Any):
        self.code = code
        self.data = np.ascontiguousarray(np.asarray(values).ravel(), dtype=self._dtypes[code])


class Raw(bytes):
    """Données binaires brutes (propriété de type 'R')."""


def _encode_prop(v: Any) -> bytes:
    if isinstance(v, Arr):
        raw = v.data.tobytes()
        if len(raw) > 128:
            payload, enc = zlib.compress(raw, 6), 1
        else:
            payload, enc = raw, 0
        return v.code.encode() + struct.pack("<3I", len(v.data), enc, len(payload)) + payload
    if isinstance(v, Raw):
        return b"R" + struct.pack("<I", len(v)) + bytes(v)
    if isinstance(v, bool):
        return b"C" + struct.pack("<?", v)
    if isinstance(v, int):
        # Les entiers « longs » (identifiants d'objets) doivent être wrappés dans L().
        return b"I" + struct.pack("<i", v)
    if isinstance(v, L):
        return b"L" + struct.pack("<q", v.value)
    if isinstance(v, float):
        return b"D" + struct.pack("<d", v)
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"S" + struct.pack("<I", len(b)) + b
    raise TypeError(f"Type de propriété FBX non géré : {type(v)!r}")


@dataclass(frozen=True)
class L:
    """Entier 64 bits (identifiants d'objets, temps KTime)."""

    value: int


@dataclass
class Node:
    name: str
    props: list = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)

    def add(self, name: str, *props: Any) -> "Node":
        child = Node(name, list(props))
        self.children.append(child)
        return child

    def encode(self, offset: int) -> bytes:
        name = self.name.encode()
        props = b"".join(_encode_prop(p) for p in self.props)
        head_len = 12 + 1 + len(name)
        body = io.BytesIO()
        pos = offset + head_len + len(props)
        for child in self.children:
            chunk = child.encode(pos)
            body.write(chunk)
            pos += len(chunk)
        if self.children or not self.props:
            body.write(_SENTINEL)
        end = offset + head_len + len(props) + body.tell()
        return struct.pack("<3I", end, len(self.props), len(props)) + bytes([len(name)]) + name + props + body.getvalue()


def _p70(parent: Node, *entries: tuple) -> Node:
    """Bloc Properties70 : chaque entrée = (nom, type, libellé, drapeaux, valeurs...)."""
    p = parent.add("Properties70")
    for e in entries:
        p.add("P", *e)
    return p


def write_fbx(nodes: list[Node]) -> bytes:
    out = io.BytesIO()
    out.write(_HEAD_MAGIC)
    out.write(struct.pack("<I", FBX_VERSION))
    for node in nodes:
        out.write(node.encode(out.tell()))
    out.write(_SENTINEL)
    # Pied de page (cf. encode_bin.py de Blender).
    out.write(_FOOT_ID)
    out.write(b"\x00" * 4)
    pad = ((out.tell() + 15) & ~15) - out.tell()
    out.write(b"\x00" * (pad or 16))
    out.write(struct.pack("<I", FBX_VERSION))
    out.write(b"\x00" * 120)
    out.write(_FOOT_MAGIC)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Construction de la scène
# ---------------------------------------------------------------------------
def build_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    texture_png: bytes | None,
    name: str = "model",
    texture_name: str = "texture.png",
    unit_scale_cm: float = 100.0,
) -> list[Node]:
    """Construit l'arbre FBX.

    ``vertices``/``normals``/``uvs`` sont indexés par sommet UV (comme
    ``UVMesh``) ; les positions sont ressoudées pour obtenir des « control
    points » partagés, les UV restent indexées par sommet de polygone.
    """
    faces = np.asarray(faces, dtype=np.int64)
    # Soudure des positions identiques (les sommets sont dupliqués aux coutures UV).
    welded, inverse = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    poly = inverse[faces].astype(np.int64)
    poly[:, 2] = -poly[:, 2] - 1  # dernier indice d'un polygone : bitwise NOT
    corner_normals = normals[faces].reshape(-1, 3)

    now = datetime.now()
    ids = {"doc": 1_000_000_001, "geom": 2_000_000_001, "model": 3_000_000_001, "mat": 4_000_000_001,
           "tex": 5_000_000_001, "video": 6_000_000_001}
    nodes: list[Node] = []

    hdr = Node("FBXHeaderExtension")
    hdr.add("FBXHeaderVersion", 1003)
    hdr.add("FBXVersion", FBX_VERSION)
    hdr.add("EncryptionType", 0)
    ts = hdr.add("CreationTimeStamp")
    for k, v in (("Version", 1000), ("Year", now.year), ("Month", now.month), ("Day", now.day), ("Hour", now.hour),
                 ("Minute", now.minute), ("Second", now.second), ("Millisecond", now.microsecond // 1000)):
        ts.add(k, v)
    hdr.add("Creator", "Volumeo FBX writer")
    nodes += [hdr, Node("FileId", [Raw(_FILE_ID)]), Node("CreationTime", [_TIME_ID]),
              Node("Creator", ["Volumeo FBX writer"])]

    gs = Node("GlobalSettings")
    gs.add("Version", 1000)
    _p70(
        gs,
        ("UpAxis", "int", "Integer", "", 1),
        ("UpAxisSign", "int", "Integer", "", 1),
        ("FrontAxis", "int", "Integer", "", 2),
        ("FrontAxisSign", "int", "Integer", "", 1),
        ("CoordAxis", "int", "Integer", "", 0),
        ("CoordAxisSign", "int", "Integer", "", 1),
        ("OriginalUpAxis", "int", "Integer", "", 1),
        ("OriginalUpAxisSign", "int", "Integer", "", 1),
        ("UnitScaleFactor", "double", "Number", "", float(unit_scale_cm)),
        ("OriginalUnitScaleFactor", "double", "Number", "", float(unit_scale_cm)),
        ("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0),
        ("DefaultCamera", "KString", "", "", "Producer Perspective"),
        ("TimeMode", "enum", "", "", 11),
        ("TimeSpanStart", "KTime", "Time", "", L(0)),
        ("TimeSpanStop", "KTime", "Time", "", L(46186158000)),
        ("CustomFrameRate", "double", "Number", "", -1.0),
    )
    nodes.append(gs)

    docs = Node("Documents")
    docs.add("Count", 1)
    doc = docs.add("Document", L(ids["doc"]), "", "Scene")
    _p70(doc, ("SourceObject", "object", "", ""), ("ActiveAnimStackName", "KString", "", "", ""))
    doc.add("RootNode", L(0))
    nodes += [docs, Node("References")]

    types = [("GlobalSettings", 1), ("Model", 1), ("Geometry", 1), ("Material", 1)]
    if texture_png is not None:
        types += [("Texture", 1), ("Video", 1)]
    defs = Node("Definitions")
    defs.add("Version", 100)
    defs.add("Count", sum(c for _, c in types))
    for t, c in types:
        defs.add("ObjectType", t).add("Count", c)
    nodes.append(defs)

    objs = Node("Objects")
    geom = objs.add("Geometry", L(ids["geom"]), f"{name}{_SEP}Geometry", "Mesh")
    geom.add("Properties70")
    geom.add("GeometryVersion", 124)
    geom.add("Vertices", Arr("d", welded))
    geom.add("PolygonVertexIndex", Arr("i", poly))
    ln = geom.add("LayerElementNormal", 0)
    ln.add("Version", 101)
    ln.add("Name", "")
    ln.add("MappingInformationType", "ByPolygonVertex")
    ln.add("ReferenceInformationType", "Direct")
    ln.add("Normals", Arr("d", corner_normals))
    lu = geom.add("LayerElementUV", 0)
    lu.add("Version", 101)
    lu.add("Name", "UVMap")
    lu.add("MappingInformationType", "ByPolygonVertex")
    lu.add("ReferenceInformationType", "IndexToDirect")
    lu.add("UV", Arr("d", uvs))
    lu.add("UVIndex", Arr("i", faces))
    lm = geom.add("LayerElementMaterial", 0)
    lm.add("Version", 101)
    lm.add("Name", "")
    lm.add("MappingInformationType", "AllSame")
    lm.add("ReferenceInformationType", "IndexToDirect")
    lm.add("Materials", Arr("i", [0]))
    layer = geom.add("Layer", 0)
    layer.add("Version", 100)
    for t in ("LayerElementNormal", "LayerElementUV", "LayerElementMaterial"):
        le = layer.add("LayerElement")
        le.add("Type", t)
        le.add("TypedIndex", 0)

    model = objs.add("Model", L(ids["model"]), f"{name}{_SEP}Model", "Mesh")
    model.add("Version", 232)
    _p70(model, ("DefaultAttributeIndex", "int", "Integer", "", 0))
    model.add("MultiLayer", 0)
    model.add("MultiTake", 0)
    model.add("Shading", True)
    model.add("Culling", "CullingOff")

    mat = objs.add("Material", L(ids["mat"]), f"material0{_SEP}Material", "")
    mat.add("Version", 102)
    mat.add("ShadingModel", "Phong")
    mat.add("MultiLayer", 0)
    _p70(
        mat,
        ("DiffuseColor", "Color", "", "A", 1.0, 1.0, 1.0),
        ("DiffuseFactor", "Number", "", "A", 1.0),
        ("SpecularColor", "Color", "", "A", 0.1, 0.1, 0.1),
        ("ShininessExponent", "Number", "", "A", 10.0),
        ("Opacity", "double", "Number", "", 1.0),
    )

    conns = Node("Connections")
    conns.add("C", "OO", L(ids["model"]), L(0))
    conns.add("C", "OO", L(ids["geom"]), L(ids["model"]))
    conns.add("C", "OO", L(ids["mat"]), L(ids["model"]))

    if texture_png is not None:
        tex = objs.add("Texture", L(ids["tex"]), f"texture{_SEP}Texture", "")
        tex.add("Type", "TextureVideoClip")
        tex.add("Version", 202)
        tex.add("TextureName", f"texture{_SEP}Texture")
        _p70(tex, ("UseMaterial", "bool", "", "", 1))
        tex.add("Media", f"texture{_SEP}Video")
        tex.add("FileName", texture_name)
        tex.add("RelativeFilename", texture_name)
        tex.add("ModelUVTranslation", 0.0, 0.0)
        tex.add("ModelUVScaling", 1.0, 1.0)
        tex.add("Texture_Alpha_Source", "None")
        tex.add("Cropping", 0, 0, 0, 0)
        vid = objs.add("Video", L(ids["video"]), f"texture{_SEP}Video", "Clip")
        vid.add("Type", "Clip")
        _p70(vid, ("Path", "KString", "XRefUrl", "", texture_name))
        vid.add("UseMipMap", 0)
        vid.add("Filename", texture_name)
        vid.add("RelativeFilename", texture_name)
        vid.add("Content", Raw(texture_png))
        conns.add("C", "OP", L(ids["tex"]), L(ids["mat"]), "DiffuseColor")
        conns.add("C", "OO", L(ids["video"]), L(ids["tex"]))

    nodes += [objs, conns]
    takes = Node("Takes")
    takes.add("Current", "")
    nodes.append(takes)
    return nodes


def export_fbx(uvmesh, path: Path) -> Path:
    png = None
    if uvmesh.texture is not None:
        buf = io.BytesIO()
        uvmesh.texture.save(buf, format="PNG")
        png = buf.getvalue()
    nodes = build_scene(uvmesh.vertices, uvmesh.faces, uvmesh.normals, uvmesh.uvs, png)
    path.write_bytes(write_fbx(nodes))
    return path
