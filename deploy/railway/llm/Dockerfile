# Model service for the hosted Keel demo: llama.cpp server, CPU build, serving Qwen2.5-3B-Instruct
# Q4_K_M over the OpenAI-compatible API on port 8081. The GGUF (about 2 GB) is fetched into /models,
# a volume, on first start and reused after that. See deploy/railway/README.md.
#
# Build context is the repository root (Railway uploads the whole repo and reads
# RAILWAY_DOCKERFILE_PATH=deploy/railway/llm/Dockerfile on the keel-llm service):
#   docker build -f deploy/railway/llm/Dockerfile -t keel-llm .
# Pin the tag (server-b<build>) before anything beyond a demo depends on it.
FROM ghcr.io/ggml-org/llama.cpp:server

# The upstream image already carries curl and ca-certificates (its own healthcheck uses curl) and
# puts llama-server at /app/llama-server.
RUN mkdir -p /models

COPY deploy/railway/llm/entrypoint.sh /usr/local/bin/keel-llm-entrypoint
RUN chmod 0755 /usr/local/bin/keel-llm-entrypoint

ENV MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf \
    MODEL_FILE=qwen2.5-3b-instruct-q4_k_m.gguf \
    MODEL_ALIAS=qwen2.5-3b-instruct \
    LLAMA_CTX=8192 \
    LLAMA_PARALLEL=2 \
    LLAMA_PORT=8081

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
  CMD curl -sf "http://127.0.0.1:${LLAMA_PORT:-8081}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/keel-llm-entrypoint"]
