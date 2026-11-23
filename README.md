# Image vers 3D

Application web qui transforme une image (JPG, PNG, WebP) d'un objet, d'un personnage ou d'un véhicule en modèle 3D texturé, à visualiser dans le navigateur puis à télécharger en GLB, OBJ, STL ou FBX.

![Capture de l'application](docs/capture.png)

## Fonctionnalités

- Import par glisser-déposer, suppression automatique de l'arrière-plan (rembg) et aperçu avant/après.
- Résolution du maillage : Brouillon (~5 000 triangles), Standard (~50 000), Haute (~200 000) ou Personnalisée (500 à 500 000 triangles, texture 512 / 1024 / 2048 / 4096 px).
- Génération par un modèle image-to-3D interchangeable : TripoSR par défaut, générateur procédural CPU en secours.
- Post-traitement : nettoyage, décimation ou subdivision vers le nombre de triangles visé, lissage de Taubin, remaillage isotrope optionnel, dépliage UV (xatlas) et bake de la texture.
- Visionneuse Three.js : rotation, zoom, éclairage réglable, modes texturé / uni / fil de fer, arêtes, nombre de triangles.
- Exports GLB, OBJ + MTL + PNG (zip), STL (Z vers le haut, 100 mm, prêt pour l'impression) et FBX 7.4 binaire avec texture embarquée.
- File d'attente (une génération à la fois), barre de progression, annulation, historique persistant.
- « Ajuster le modèle affiché » : change résolution, texture et lissage d'une génération existante sans relancer l'IA.

## Démarrage rapide (Docker)

```bash
docker compose up --build
# puis ouvrir http://localhost:8000
```

Avec un GPU NVIDIA :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Image légère sans PyTorch ni TripoSR (générateur procédural uniquement) :

```bash
INSTALL_AI=0 docker compose up --build
```

Les volumes `data` (images, modèles générés, historique) et `models` (poids téléchargés) survivent aux redémarrages.

## Prérequis GPU

| | Minimum | Recommandé |
|---|---|---|
| GPU | aucun (CPU) | NVIDIA, 8 Go de VRAM (TripoSR en utilise environ 6) |
| Pilote | — | pilote NVIDIA récent + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) pour Docker |
| RAM | 8 Go | 16 Go |
| Disque | 3 Go | 10 Go (image CUDA + poids) |

Sans GPU CUDA, l'application démarre quand même : un avertissement s'affiche dans les logs et dans l'interface (« CPU uniquement »), et TripoSR tourne sur CPU (de 30 s à plusieurs minutes par génération selon la résolution). Le périphérique se force avec `I3D_DEVICE=cpu` ou `I3D_DEVICE=cuda:1`.

## Poids des modèles

Rien à télécharger à la main dans le cas général : au premier lancement, les poids sont récupérés automatiquement et mis en cache dans le volume `models`.

| Modèle | Source | Taille | Usage |
|---|---|---|---|
| TripoSR | Hugging Face `stabilityai/TripoSR` (`config.yaml`, `model.ckpt`) | ~1,7 Go | génération 3D |
| DINO ViT-B/16 | Hugging Face `facebook/dino-vitb16` (`config.json` seulement) | quelques Ko | encodeur d'image de TripoSR |
| IS-Net | GitHub (rembg) | ~170 Mo | suppression de l'arrière-plan |

Les poids de TripoSR sont publics, aucun compte ni jeton Hugging Face n'est requis.

Pour une machine sans accès Internet, téléchargez-les ailleurs puis copiez-les :

```bash
pip install huggingface_hub
huggingface-cli download stabilityai/TripoSR config.yaml model.ckpt --local-dir models/TripoSR
huggingface-cli download facebook/dino-vitb16 config.json        # reste dans le cache HF
# puis : I3D_TRIPOSR_MODEL=/models/TripoSR  et  HF_HUB_OFFLINE=1
```

## Installation locale (développement)

Prérequis : Python 3.10 à 3.13, Node.js 20+, et sous Debian/Ubuntu `sudo apt install libgl1 libopengl0 libglx0 libegl1` (plugins PyMeshLab).

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Optionnel : TripoSR
pip install torch                       # ou --index-url https://download.pytorch.org/whl/cpu pour une version CPU
pip install -r requirements-ai.txt
git clone https://github.com/VAST-AI-Research/TripoSR ../third_party/TripoSR
export PYTHONPATH=$PWD/../third_party/TripoSR

uvicorn app.main:app --reload --port 8000

# Frontend (autre terminal)
cd frontend
npm install
npm run dev          # http://localhost:5173, /api est relayé vers le port 8000
```

`torchmcubes`, requis par le dépôt TripoSR d'origine et pénible à compiler, n'est pas nécessaire : l'extraction de surface passe par scikit-image.

## Utilisation

1. Déposez une image dans la zone « Image ». Le détourage s'affiche ; « Voir l'original » permet de comparer.
2. Choisissez une résolution, et si besoin la taille de texture et le lissage.
3. Choisissez le modèle de génération, puis « Générer le modèle 3D ».
4. Inspectez le résultat dans la visionneuse, ajustez-le si nécessaire, puis téléchargez-le.

Conseils pour de bons résultats : un seul sujet, entier, bien éclairé, vu de trois quarts, sur un fond simple.

### Exemples avec l'API

```bash
# 1. Importer une image (détourage automatique)
curl -F file=@chaise.jpg http://localhost:8000/api/uploads
# -> {"id": "3f2a…", "background_method": "rembg", "urls": {…}}

# 2. Lancer une génération
curl -H 'Content-Type: application/json' \
     -d '{"upload_id": "3f2a…", "preset": "standard"}' \
     http://localhost:8000/api/jobs

# Résolution personnalisée
curl -H 'Content-Type: application/json' \
     -d '{"upload_id": "3f2a…", "preset": "custom", "target_faces": 120000, "texture_size": 4096, "smoothing": 3, "generator": "triposr"}' \
     http://localhost:8000/api/jobs

# 3. Suivre la progression
curl http://localhost:8000/api/jobs/<job_id>

# 4. Télécharger
curl -OJ http://localhost:8000/api/jobs/<job_id>/download/glb    # glb | obj | stl | fbx
```

La documentation interactive de l'API est sur `http://localhost:8000/docs`.

## Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `I3D_GENERATOR` | `auto` | `auto` (TripoSR s'il est installé, sinon procédural), `triposr`, `procedural` |
| `I3D_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`… |
| `I3D_TRIPOSR_MODEL` | `stabilityai/TripoSR` | identifiant Hugging Face ou dossier local |
| `I3D_CHUNK_SIZE` | `8192` | taille des lots du réseau (baisser si la VRAM manque) |
| `I3D_REMBG_MODEL` | `isnet-general-use` | modèle rembg (`u2net`, `u2netp`, `birefnet-general`…) |
| `I3D_USE_REMBG` | `1` | `0` force le détourage simple par couleur de fond |
| `I3D_DATA_DIR` | `./data` | uploads, résultats, historique SQLite |
| `I3D_MAX_UPLOAD_MB` | `20` | taille maximale d'une image |
| `I3D_HISTORY_LIMIT` | `100` | générations conservées (0 = illimité) |
| `I3D_CORS_ORIGINS` | `http://localhost:5173` | origines autorisées en développement |

## Architecture

```
image-to-3d/
├─ backend/
│  ├─ app/
│  │  ├─ main.py            création de l'app FastAPI, sert aussi le frontend compilé
│  │  ├─ api.py             routes /api (uploads, jobs, téléchargements)
│  │  ├─ config.py          configuration par variables d'environnement
│  │  ├─ uploads.py         stockage des images et de leur détourage
│  │  ├─ jobs/
│  │  │  ├─ queue.py        file d'attente asyncio + thread dédié au pipeline
│  │  │  └─ store.py        historique SQLite
│  │  └─ pipeline/
│  │     ├─ background.py   chargement, rembg, détourage de secours, cadrage
│  │     ├─ presets.py      préréglages de résolution
│  │     ├─ device.py       détection GPU / CPU
│  │     ├─ generators/     base.py (interface), triposr.py, procedural.py
│  │     ├─ mesh.py         nettoyage, décimation, subdivision, lissage
│  │     ├─ texture.py      dépliage UV xatlas + bake CPU
│  │     ├─ exporters/      GLB, OBJ, STL, fbx.py (écrivain FBX binaire)
│  │     └─ runner.py       orchestration et progression
│  └─ tests/
├─ frontend/src/            App.tsx, api.ts, components/ (Dropzone, ResolutionPicker, Viewer, ViewerToolbar, History)
├─ Dockerfile, docker-compose.yml, docker-compose.gpu.yml
└─ README.md
```

### Pipeline

1. **Détourage** : rembg, ou transparence d'origine si l'image en a déjà une, ou à défaut estimation de la couleur de fond.
2. **Cadrage** : sujet centré dans un carré 512 px, occupant 85 % du cadre.
3. **Génération** : le modèle produit un champ de densité et de couleur ; marching cubes sur une grille de 160³ à 320³ selon la résolution.
4. **Nettoyage** : fusion des sommets, suppression des fragments flottants, normalisation (hauteur ≤ 1, posé sur le sol).
5. **Remaillage** : décimation quadrique PyMeshLab qui préserve la topologie (repli sur fast-simplification), subdivision si le maillage brut est trop léger.
6. **UV** : xatlas, appliqué par blocs de ~6 000 triangles puis empaqueté dans un atlas unique (≈10 s pour 200 000 triangles au lieu de ≈140 s).
7. **Bake** : chaque texel interroge le champ couleur du modèle à sa position 3D, puis dilatation des bords.
8. **Export** : GLB immédiatement, les autres formats à la demande (mis en cache).

Temps mesurés sur 4 cœurs CPU avec le générateur procédural : Brouillon ~2 s, Standard ~10 s, Haute ~20 s (hors temps du modèle neuronal).

### Ajouter un modèle (SF3D, InstantMesh, Hunyuan3D-2…)

Créez une sous-classe de `Generator` dans `backend/app/pipeline/generators/` qui renvoie un `RawResult` (maillage brut Y vers le haut + `color_fn(points) -> rgb`), puis ajoutez-la à `REGISTRY` dans `generators/__init__.py`. Elle apparaît automatiquement dans l'API et dans le sélecteur de l'interface. Si le modèle fournit déjà des couleurs par sommet, `color_fn_from_vertex_colors` fait l'affaire.

## Tests

```bash
cd backend
python -m pytest
```

36 tests couvrent le détourage, les préréglages, le remaillage, le bake, le pipeline complet image → maillage → GLB/OBJ/STL/FBX (relus avec trimesh, un lecteur FBX de test et `assimp` s'il est installé), l'API et la file d'attente, ainsi que l'adaptateur TripoSR avec un faux modèle. Ils n'ont besoin ni de GPU ni de poids.

## Limites connues

- TripoSR reconstruit à partir d'une seule vue : l'arrière du modèle est inventé et les textures restent douces.
- Le générateur procédural n'est pas une reconstruction 3D : il gonfle la silhouette et projette l'image dessus.
- Une seule génération à la fois, pas d'authentification : à placer derrière un proxy si l'application est exposée.
- L'export FBX est validé avec Assimp ; vérifiez l'import dans votre logiciel cible (Blender, Unity, Unreal).

## Licences des composants

TripoSR (MIT), rembg (MIT), xatlas (MIT), trimesh (MIT), PyMeshLab (GPL-3.0), Three.js et react-three-fiber (MIT). PyMeshLab étant sous GPL, retirez-le de `requirements.txt` si cette licence ne vous convient pas : la décimation bascule alors sur fast-simplification (MIT).
