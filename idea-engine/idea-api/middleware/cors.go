// Package middleware provides HTTP middleware for the idea-api service.
package middleware

import (
	"net/http"
	"strings"
)

// allowedOrigins is the set of origins permitted to make cross-origin requests
// to the idea-api. These cover the Vite dev server and its fallback ports as
// well as the idea dashboard.
var allowedOrigins = map[string]bool{
	"http://localhost:3000": true,
	"http://localhost:5173": true,
	"http://localhost:5174": true,
	"http://localhost:5175": true,
	"http://127.0.0.1:3000": true,
	"http://127.0.0.1:5173": true,
	"http://127.0.0.1:5174": true,
	"http://127.0.0.1:5175": true,
}

const corsAllowedMethods = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
const corsAllowedHeaders = "Accept, Authorization, Content-Type, X-Requested-With, X-Request-ID"
const corsExposedHeaders = "Content-Length, X-Request-ID"

// CORS is a middleware that adds Cross-Origin Resource Sharing headers for the
// allowed localhost development origins. Preflight OPTIONS requests receive an
// immediate 204 No Content response.
func CORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		if origin != "" {
			if isAllowedOrigin(origin) {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Vary", "Origin")
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			} else {
				// Unknown origin: reflect back without credentials.
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Vary", "Origin")
			}
		}

		w.Header().Set("Access-Control-Allow-Methods", corsAllowedMethods)
		w.Header().Set("Access-Control-Allow-Headers", corsAllowedHeaders)
		w.Header().Set("Access-Control-Expose-Headers", corsExposedHeaders)
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// isAllowedOrigin reports whether origin is in the allow-list.
// The comparison is case-insensitive and trailing slashes are ignored.
func isAllowedOrigin(origin string) bool {
	o := strings.TrimRight(strings.ToLower(origin), "/")
	return allowedOrigins[o]
}
