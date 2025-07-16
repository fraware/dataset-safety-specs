# Multi-stage Dockerfile for ds-guard

FROM rust:1.70 as rust-builder
WORKDIR /app
COPY extracted/rust/ .
RUN cargo build --release

FROM python:3.9-slim as python-builder
WORKDIR /app
COPY extracted/python/ .
RUN pip install -e .

FROM python:3.9-slim
WORKDIR /app

# Install Rust runtime
RUN apt-get update && apt-get install -y libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy Rust binary
COPY --from=rust-builder /app/target/release/ds_guard /usr/local/bin/

# Copy Python package
COPY --from=python-builder /app /app/python
RUN pip install -e /app/python

# Copy monitoring
COPY extracted/monitoring/ /app/monitoring/

EXPOSE 8080

CMD ["python", "-m", "ds_guard.server"]