FROM node:22-slim

# Pinned: bumping requires re-running at least one TriMul sweep against the
# new CLI version, since CLI behavior (model-alias resolver, settings.json
# schema, default thinkingLevel) can shift between minor releases — see
# ``_GEMINI_MODEL_ALIASES`` and ``verify_thinking_level.py``.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @google/gemini-cli@0.38.2 \
 && npm cache clean --force

WORKDIR /work
