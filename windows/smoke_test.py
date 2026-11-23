"""Test de fumée : importe une image, lance une génération et télécharge les 4 formats.

    python windows/smoke_test.py http://127.0.0.1:8765
"""

import io
import json
import sys
import time
import urllib.request
import uuid

from PIL import Image, ImageDraw

base = sys.argv[1].rstrip("/")


def call(path, data=None, headers=None):
    req = urllib.request.Request(base + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


for _ in range(120):  # attend le démarrage du serveur
    try:
        call("/api/health")
        break
    except OSError:
        time.sleep(0.5)
else:
    sys.exit("le serveur ne répond pas")

assert b"Volumeo" in call("/"), "l'interface n'est pas servie"
system = json.loads(call("/api/system"))
print("générateurs :", [(g["name"], g["available"], g["reason"]) for g in system["generators"]])

img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
ImageDraw.Draw(img).ellipse((50, 30, 206, 226), fill=(40, 160, 60, 255))
buf = io.BytesIO()
img.save(buf, "PNG")
boundary = uuid.uuid4().hex
body = (
    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\n'
    "Content-Type: image/png\r\n\r\n"
).encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
up = json.loads(call("/api/uploads", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))

job = json.loads(call("/api/jobs", json.dumps({"upload_id": up["id"], "preset": "draft", "generator": "procedural"}).encode(),
                      {"Content-Type": "application/json"}))
for _ in range(600):
    job = json.loads(call(f"/api/jobs/{job['id']}"))
    if job["status"] in ("done", "failed", "cancelled"):
        break
    time.sleep(0.5)
assert job["status"] == "done", job
print("modèle :", job["result"]["faces"], "triangles, étanche :", job["result"]["watertight"])
for fmt in ("glb", "obj", "stl", "fbx"):
    size = len(call(f"/api/jobs/{job['id']}/download/{fmt}"))
    assert size > 1000, fmt
    print(f"  {fmt}: {size} octets")
print("OK")
