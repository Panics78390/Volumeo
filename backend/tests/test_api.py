"""Tests de l'API HTTP : import → génération en file d'attente → historique → téléchargements."""

from tests.conftest import make_photo, to_bytes, wait_job


def _upload(client, fmt="JPEG"):
    r = client.post("/api/uploads", files={"file": (f"jouet.{fmt.lower()}", to_bytes(make_photo(), fmt), "image/jpeg")})
    assert r.status_code == 201, r.text
    return r.json()


def test_system_info(client):
    info = client.get("/api/system").json()
    assert info["default_generator"] == "procedural"
    assert {p["key"] for p in info["presets"]} == {"draft", "standard", "high"}
    assert info["texture_sizes"] == [512, 1024, 2048, 4096]
    assert any(g["name"] == "procedural" and g["available"] for g in info["generators"])
    assert "warning" in info["device"]


def test_upload_and_preview(client):
    up = _upload(client)
    assert up["background_method"] == "fallback"
    r = client.get(up["urls"]["cutout"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_upload_rejects_bad_file(client):
    r = client.post("/api/uploads", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_full_generation_flow(client):
    up = _upload(client, "PNG")
    r = client.post("/api/jobs", json={"upload_id": up["id"], "preset": "custom", "target_faces": 3000,
                                       "texture_size": 512, "generator": "procedural", "smoothing": 2})
    assert r.status_code == 201, r.text
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert abs(job["result"]["faces"] - 3000) <= 150
    assert job["name"] == "jouet"

    glb = client.get(job["urls"]["model"])
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    for fmt, magic in {"obj": b"PK", "stl": None, "fbx": b"Kaydara"}.items():
        d = client.get(job["urls"]["downloads"][fmt])
        assert d.status_code == 200, fmt
        assert "attachment" in d.headers["content-disposition"]
        if magic:
            assert d.content.startswith(magic)

    hist = client.get("/api/jobs").json()
    assert hist["total"] == 1 and hist["items"][0]["id"] == job["id"]

    # Ré-ajustement sans relancer le générateur.
    r = client.post(f"/api/jobs/{job['id']}/reprocess", json={"preset": "draft", "texture_size": 512})
    assert r.status_code == 201, r.text
    child = wait_job(client, r.json()["id"])
    assert child["status"] == "done" and child["parent_id"] == job["id"]
    assert abs(child["result"]["faces"] - 5000) <= 250

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_validation_errors(client):
    up = _upload(client)
    bad = [
        {"upload_id": "000000000000", "preset": "draft"},
        {"upload_id": up["id"], "preset": "custom"},
        {"upload_id": up["id"], "preset": "custom", "target_faces": 10_000, "texture_size": 300},
        {"upload_id": up["id"], "preset": "draft", "generator": "inconnu"},
    ]
    for body in bad:
        assert client.post("/api/jobs", json=body).status_code in (404, 422), body


def test_queue_runs_jobs_in_order_and_cancel(client):
    up = _upload(client)
    body = {"upload_id": up["id"], "preset": "custom", "target_faces": 2000, "texture_size": 512}
    ids = [client.post("/api/jobs", json=body).json()["id"] for _ in range(3)]
    client.post(f"/api/jobs/{ids[2]}/cancel")
    jobs = [wait_job(client, i) for i in ids]
    assert [j["status"] for j in jobs] == ["done", "done", "cancelled"]
    assert jobs[0]["finished_at"] <= jobs[1]["started_at"], "une seule génération à la fois"
