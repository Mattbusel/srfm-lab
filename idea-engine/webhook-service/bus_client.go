// Package main — bus_client.go
//
// This file is intentionally minimal. The real BusClient implementation lives
// in handlers/bus_client.go (package handlers) and is used by all webhook
// handlers. It is documented there in full.
//
// See handlers/bus_client.go for:
//   - BusClient struct and constructor
//   - Publish / PublishWithRetry
//   - PublishSignal / PublishRegimeChange / PublishNewsEvent helpers
//   - Exponential back-off logic
package main
