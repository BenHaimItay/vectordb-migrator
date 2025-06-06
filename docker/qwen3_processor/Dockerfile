# Step 1: Base Image
FROM python:3.9-slim

# Step 2: System Dependencies & Utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    g++ \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Clone and Build llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp
# RUN git checkout <specific_commit_hash> # Optional: Pin to a specific llama.cpp version for stability
RUN cmake -B build -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_AVX512=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF -DLLAMA_OPENBLAS=OFF
RUN cmake --build build --config Release

# Step 4: Set up Python environment /app
WORKDIR /app
COPY ./docker/qwen3_processor/requirements.txt .

# Step 5: Install Python Dependencies from requirements.txt
# Ensure llama-cpp-python is built against the llama.cpp we just built
# Set CMAKE_ARGS to point to the local llama.cpp build
# The exact CMAKE_ARGS might need tuning based on llama-cpp-python version
# For llama-cpp-python, it often tries to build its own bundled llama.cpp unless told otherwise.
# One way is to ensure the headers and library are found, or to install it in editable mode linking to the source.
# However, often llama-cpp-python's pip install will build its own version of llama.cpp.
# For simplicity in this step, we'll let llama-cpp-python's pip install handle its own llama.cpp build,
# The pre-built llama.cpp above might be redundant if pip builds its own, or useful if llama-cpp-python can link to it.
# Let's assume pip install of llama-cpp-python will handle its C++ deps correctly for now.
# The cmake build of llama.cpp above is good for having the llama.cpp tools, but llama-cpp-python might ignore it.
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Download GGUF Model
RUN mkdir -p /app/models
# huggingface-hub is in requirements.txt
RUN huggingface-cli download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir /app/models --local-dir-use-symlinks False --quiet

# Step 7: Copy the rest of the application code
COPY . /app/

# Step 8: Install the current project (vectordb-migration tool)
RUN pip install .

# Step 9: Define default command
# This will be the script that orchestrates embedding, loading to pgvector, and migration
CMD ["python", "examples/qwen3_process_and_migrate.py"]
