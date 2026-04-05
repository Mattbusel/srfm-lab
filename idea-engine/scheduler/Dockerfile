FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -o /idea-scheduler .

FROM alpine:3.19
RUN apk add --no-cache ca-certificates sqlite python3
WORKDIR /app
COPY --from=builder /idea-scheduler .
EXPOSE 8769
ENTRYPOINT ["/app/idea-scheduler"]
