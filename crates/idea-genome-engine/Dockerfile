FROM rust:1.77-alpine AS builder
RUN apk add --no-cache musl-dev sqlite-dev
WORKDIR /app
# Copy workspace Cargo files
COPY Cargo.toml Cargo.lock ./
COPY crates/idea-genome-engine ./crates/idea-genome-engine
# Build only the genome engine crate
RUN cargo build --release -p idea-genome-engine

FROM alpine:3.19
RUN apk add --no-cache ca-certificates sqlite
COPY --from=builder /app/target/release/idea-genome-engine /usr/local/bin/
ENTRYPOINT ["idea-genome-engine"]
