# Use a Python version that is known to work well with llama-cpp-python
FROM python:3.9-slim

WORKDIR /app

# Install cmake and a C compiler, essential for building llama-cpp-python if needed
# Some base images might need build-essential or similar
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    pkg-config \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Consider setting CMAKE_ARGS for llama.cpp build if specific features are needed (e.g., GPU support via CMAKE_ARGS="-DLLAMA_CUBLAS=ON")
# For CPU-only, this is often not needed, but it's good to be aware of.
# Ensure pip uses the version from requirements.txt for llama-cpp-python
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# MODEL_PATH will be passed from docker-compose.yml
# EXPOSE 5000 (already handled by docker-compose 'ports')

CMD ["python", "app.py"]
