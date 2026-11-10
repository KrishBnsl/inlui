# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for the sim_engine Rust workspace
# ──────────────────────────────────────────────────────────────────────────────

# Stage 1: Build
FROM rust:1.87-slim AS builder

# Install system deps needed by sqlx (openssl, pkg-config)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for layer-cached dependency compilation
COPY Cargo.toml Cargo.lock ./

# Create stub sources so cargo can resolve the workspace without the real code
RUN mkdir -p src/bin && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/ingest.rs && \
    echo 'fn main() {}' > src/bin/server.rs && \
    echo '' > src/lib.rs

RUN cargo build --release --bins 2>/dev/null; exit 0

# Now copy the real source and rebuild (only changed files are re-compiled)
COPY src ./src
RUN touch src/main.rs src/bin/ingest.rs src/bin/server.rs src/lib.rs && \
    cargo build --release --bins

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy both compiled binaries
COPY --from=builder /build/target/release/ingest ./ingest
COPY --from=builder /build/target/release/server ./server

# Default: run the HTTP server (overridden by docker-compose for ingest)
CMD ["./server"]
