# syntax=docker/dockerfile:1
# Image unique : le frontend compilé est servi par FastAPI sur le port 8000.

# ---------------------------------------------------------------- frontend
FROM node:22-slim AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------- backend
FROM python:3.11-slim AS app

# Index des roues PyTorch : CPU par défaut. Pour un GPU NVIDIA, utilisez
# docker-compose.gpu.yml (qui passe l'index PyPI, dont les roues Linux incluent
# CUDA). Les roues CUDA embarquent
# leurs bibliothèques : l'hôte n'a besoin que du pilote NVIDIA et du
# NVIDIA Container Toolkit.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# INSTALL_AI=0 produit une image légère, limitée au générateur procédural.
ARG INSTALL_AI=1
ARG TRIPOSR_REF=107cefdc244c39106fa830359024f6a2f1c78871
ARG TRIPOSG_REF=fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git : récupération du code TripoSR et TripoSG ; libGL/libOpenGL/libEGL : plugins PyMeshLab ;
# libgomp : onnxruntime (rembg).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 libopengl0 libglx0 libegl1 libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements-ai.txt ./
RUN pip install -r requirements.txt
RUN if [ "$INSTALL_AI" = "1" ]; then \
      pip install torch --index-url "$TORCH_INDEX_URL" \
   && pip install -r requirements-ai.txt \
   && git clone https://github.com/VAST-AI-Research/TripoSR /opt/TripoSR \
   && git -C /opt/TripoSR checkout "$TRIPOSR_REF" \
   && rm -rf /opt/TripoSR/.git /opt/TripoSR/examples /opt/TripoSR/figures \
   && git clone https://github.com/VAST-AI-Research/TripoSG /opt/TripoSG \
   && git -C /opt/TripoSG checkout "$TRIPOSG_REF" \
   && rm -rf /opt/TripoSG/.git /opt/TripoSG/assets; \
    fi

COPY backend/app ./app
COPY --from=frontend /src/dist ./static

ENV PYTHONPATH=/opt/TripoSR:/opt/TripoSG \
    I3D_DATA_DIR=/data \
    I3D_MODELS_DIR=/models \
    I3D_STATIC_DIR=/app/static \
    HF_HOME=/models/huggingface \
    U2NET_HOME=/models/rembg

VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
