# ┌─────────────────────────────────────────────────────────────────────────┐
# │                              Dockerfile                               │
# └─────────────────────────────────────────────────────────────────────────┘

# 1. Use a minimal Python base
FROM python:3.12-slim

# 2. Create a non-root user & switch to it
RUN useradd --create-home --shell /bin/bash appuser

# 3. Set a locked-down working directory
WORKDIR /app

# 4. Ensure model directory exists and set ownership
RUN mkdir -p /app/model && chown appuser:appuser /app/model

# 5. Copy & install only dependencies first (cache-friendly)
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY --chown=appuser:appuser . ./

# 7. Expose only the port your app needs
EXPOSE 9999

# 8. Run as non-root user default entrypoint
ENTRYPOINT ["python", "main.py", "9999"]

