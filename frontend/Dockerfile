# Multi-stage build: compile the Vite bundle, then serve it with `vite preview`.
FROM node:20-alpine AS build

WORKDIR /app

# VITE_API_BASE_URL is baked into the static bundle at build time.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

# Runtime stage: a lean image that serves the built assets.
FROM node:20-alpine AS runtime

WORKDIR /app

COPY --from=build /app/package.json ./package.json
COPY --from=build /app/vite.config.ts ./vite.config.ts
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist

EXPOSE 4173

# `vite preview` serves the production build on 0.0.0.0:4173.
CMD ["npm", "run", "preview"]
