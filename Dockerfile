FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---------- system build deps ----------
# - curl/ca-certificates: fetch liboqs release tarball
# - cmake/ninja/build-essential/pkg-config: compile liboqs
# - openssl/libssl-dev: required by liboqs CMake configuration
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    file \
    cmake ninja-build build-essential pkg-config \
    openssl libssl-dev \
    postgresql-client \
  && rm -rf /var/lib/apt/lists/*

# ---------- build & install liboqs (no git clone) ----------
# Pin a liboqs release tag for reproducibility.
ARG LIBOQS_TAG=0.14.0
RUN set -eux; \
  url="https://github.com/open-quantum-safe/liboqs/archive/refs/tags/${LIBOQS_TAG}.tar.gz"; \
  curl -fL --retry 5 --retry-delay 2 -o /tmp/liboqs.tar.gz "$url"; \
  file /tmp/liboqs.tar.gz | grep -qi 'gzip compressed' ; \
  mkdir -p /tmp/liboqs-src; \
  tar -xzf /tmp/liboqs.tar.gz -C /tmp/liboqs-src --strip-components=1; \
  mkdir -p /tmp/liboqs-src/build; \
  cd /tmp/liboqs-src/build; \
  cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..; \
  ninja; \
  ninja install; \
  ldconfig; \
  rm -rf /tmp/liboqs.tar.gz /tmp/liboqs-src

# ---------- python deps ----------
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  # Remove the unrelated PyPI package "oqs" if it ever gets pulled in
  && pip uninstall -y oqs || true \
  && pip install -r /app/requirements.txt

# ---------- app code ----------
COPY app/ /app/app/
COPY scripts/ /app/scripts/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "app.stage2"]
