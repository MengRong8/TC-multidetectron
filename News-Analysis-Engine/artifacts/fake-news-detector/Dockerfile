FROM node:20-bookworm-slim

WORKDIR /workspace

RUN corepack enable

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml /workspace/
COPY lib /workspace/lib
COPY artifacts/fake-news-detector /workspace/artifacts/fake-news-detector
COPY scripts /workspace/scripts

RUN pnpm install --frozen-lockfile

ENV PORT=3000 \
    BASE_PATH=/ \
    VITE_API_URL=http://backend:5000

EXPOSE 3000

CMD ["pnpm", "--filter", "@workspace/fake-news-detector", "dev"]
