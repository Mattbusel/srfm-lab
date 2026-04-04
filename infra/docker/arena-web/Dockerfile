FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --prefer-offline 2>/dev/null || npm install
COPY . .
RUN npm run build 2>/dev/null || echo "No build script, using static files"

# ---- Final stage ----
FROM nginx:1.25-alpine

# Copy nginx config.
COPY --from=builder /app/dist /usr/share/nginx/html 2>/dev/null || true
COPY --from=builder /app/build /usr/share/nginx/html 2>/dev/null || true

# Fallback: copy all HTML if no build output.
COPY --from=builder /app /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf 2>/dev/null || true

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
