# ---------- build stage ----------
FROM node:20-alpine AS builder

WORKDIR /app

RUN npm install -g npm@11.17.0

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------- serve stage ----------
FROM nginx:1.27-alpine

ENV API_PROXY_TARGET=http://backend:8000
ENV AL_MEDLIT_AUTH_RATE=10r/m

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY nginx/rate-limit.conf.template /etc/nginx/templates/00-rate-limit.conf.template

EXPOSE 80
