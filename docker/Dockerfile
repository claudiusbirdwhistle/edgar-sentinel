# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir build

COPY pyproject.toml .
COPY src/ src/

RUN python -m build --wheel --outdir /build/dist

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 sentinel \
    && useradd --uid 1000 --gid sentinel --create-home sentinel

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[api,backtest] \
    && rm -f /tmp/*.whl

RUN mkdir -p /app/data && chown sentinel:sentinel /app/data

USER sentinel

ENV EDGAR_SENTINEL_CONFIG=/app/edgar-sentinel.yml

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health').raise_for_status()"

EXPOSE 8000

ENTRYPOINT ["edgar-sentinel"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
