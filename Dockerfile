# Reproducible environment for the iceberg drift physics engine.
# Build:  docker build -t iceberg-model .
# Run tests:      docker run --rm iceberg-model pytest -q
# Run an example: docker run --rm iceberg-model python examples/run_real_data.py
# Run live fetch: docker run --rm --env-file .env iceberg-model python examples/fetch_and_run_live.py ...

FROM python:3.12-slim

# GDAL/GEOS system libraries are required by rasterio/geopandas/shapely.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

COPY tests ./tests
COPY examples ./examples
COPY configs ./configs
COPY docs ./docs
COPY README.md ./

# Optional: uncomment to bake in live-data fetcher dependencies
# (cdsapi, earthaccess, copernicusmarine). Left out of the default image
# since they add significant size and most use cases (running the
# physics engine on already-fetched data) don't need them.
# RUN pip install --no-cache-dir -e ".[live-data]"

CMD ["pytest", "-q"]
