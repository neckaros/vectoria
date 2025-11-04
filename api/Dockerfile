# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv with no cache
RUN pip install --no-cache-dir uv

# Copy dependency files AND readme FIRST (for better layer caching)
COPY pyproject.toml uv.lock* README.md ./

# Set UV link mode to copy (needed for Docker)
ENV UV_LINK_MODE=copy

# Install dependencies without dev dependencies
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy only the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app/ ./app/
COPY dev.py ./

# Expose port
EXPOSE 7979

# Run the API using the venv Python directly
CMD [".venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7979"]
