# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/chatbot/.cache/huggingface \
    MINERU_MODEL_SOURCE=huggingface \
    MINERU_DEVICE_MODE=cpu

# MinerU/OpenCV needs GL and GLib; Node/npm supports device-local MCP servers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 chatbot \
    && useradd --uid 10001 --gid chatbot --create-home chatbot
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

WORKDIR /app
COPY pyproject.toml demo_requirements.txt ./
# Resolve Linux dependencies from pyproject, without the frozen Windows/CUDA
# environment. Install CPU wheels first to avoid pulling the CUDA toolchain.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/runtime-requirements.txt \
    && pip install -r /tmp/runtime-requirements.txt -r demo_requirements.txt \
        'psycopg[binary]>=3.3.1,<4' 'mineru[pipeline]==3.4.4' \
    && pip check

COPY . .
RUN pip install --no-deps . \
    && mkdir -p app/temp app/storage /home/chatbot/.cache \
        /home/chatbot/workspace /home/chatbot/profile/skills /home/chatbot/beat \
    && chown -R chatbot:chatbot /app /home/chatbot

USER chatbot
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
