# syntax=docker/dockerfile:1
# One image for the agent, the sandbox and the harness CLI; the entrypoint is `booking-truth`.
# Default command: `agent serve --host 0.0.0.0 --port 8000`. The sandbox: `sandbox serve --host 0.0.0.0` (port 8100).

ARG PYTHON_IMAGE=python:3.12-slim

# Build stage: the wheel (with scenarios, datasets, schemas, pricing and the widget as package data) and the
# locked, hash-pinned runtime requirements exported from uv.lock.
FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
RUN pip install uv==0.12.9
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE pricing.yaml ./
COPY src ./src
COPY scenarios ./scenarios
COPY datasets ./datasets
COPY schemas ./schemas
COPY widget ./widget
RUN uv export --frozen --no-dev --no-emit-project --format requirements-txt --output-file /dist/requirements.txt \
    && uv build --wheel --out-dir /dist

# Runtime stage: pip installs the locked requirements (hashes checked), then the wheel without dependencies.
FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="booking-truth" \
      org.opencontainers.image.description="Tests appointment-setting agents on the calendar's end state, and ships a guarded booking agent." \
      org.opencontainers.image.source="https://github.com/B0yko/booking-truth" \
      org.opencontainers.image.licenses="Apache-2.0"
# PYTHONTZPATH="" makes zoneinfo read the pinned tzdata Python package (the same data the time zone resolver
# reads zone.tab and iso3166.tab from), not whatever tzdata release the base image's OS ships.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONTZPATH=""
RUN --mount=type=bind,from=build,source=/dist,target=/dist \
    pip install --require-hashes --requirement /dist/requirements.txt \
    && pip install --no-deps /dist/*.whl \
    && useradd --create-home --uid 10001 --user-group bt \
    && install -d -o bt -g bt /data
USER bt
WORKDIR /home/bt
EXPOSE 8000 8100
ENTRYPOINT ["booking-truth"]
CMD ["agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
