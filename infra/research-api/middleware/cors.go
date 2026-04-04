// Package middleware provides HTTP middleware for the research API.
package middleware

import (
	"net/http"
	"strings"
)

// allowedOrigins is the set of origins permitted to make cross-origin requests
// to the research API. These cover the Vite dev server and its fallback ports.
var allowedOrigins = map[string]bool{
	"http://localhost:5173": true,
	"http://localhost:5174": true,
	"http://localhost:5175": true,
	"http://127.0.0.1:5173": true,
	"http://127.0.0.1:5174": true,
	"http://127.0.0.1:5175": true,
}

// corsAllowedMethods lists the HTTP methods the API exposes.
const corsAllowedMethods = "GET, POST, PUT, PATCH, DELETE, OPTIONS"

// corsAllowedHeaders lists headers clients may send.
const corsAllowedHeaders = "Accept, Authorization, Content-Type, X-Requested-With, X-Request-ID"

// corsExposedHeaders lists response headers the browser may read.
const corsExposedHeaders = "Content-Length, X-Request-ID"

// CORS returns a middleware that adds permissive Cross-Origin Resource Sharing
// headers for the allowed localhost development origins defined above.
//
// For preflight OPTIONS requests the handler responds immediately with 204 No
// Content so the browser does not block the actual request.
func CORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		if origin != "" {
			if isAllowed(origin) {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Vary", "Origin")
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			} else {
				// Unknown origin: allow for same-machine tooling but without
				// credentials. Adjust to your security requirements.
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Vary", "Origin")
			}
		}

		w.Header().Set("Access-Control-Allow-Methods", corsAllowedMethods)
		w.Header().Set("Access-Control-Allow-Headers", corsAllowedHeaders)
		w.Header().Set("Access-Control-Expose-Headers", corsExposedHeaders)
		w.Header().Set("Access-Control-Max-Age", "86400") // 24 h preflight cache

		// Preflight request: respond immediately, no body.
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// isAllowed reports whether origin appears in allowedOrigins.
// The check is case-insensitive and strips trailing slashes.
func isAllowed(origin string) bool {
	o := strings.TrimRight(strings.ToLower(origin), "/")
	return allowedOrigins[o]
}
