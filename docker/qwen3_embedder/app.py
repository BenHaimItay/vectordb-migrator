import os
from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH environment variable not set.")

try:
    print(f"Loading model from: {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        embedding=True,
        verbose=True # Enable verbose logging from llama.cpp
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None # Ensure llm is None if loading failed

@app.route("/embed", methods=["POST"])
def embed():
    if llm is None:
        return jsonify({"error": "Model not loaded or failed to load."}), 500

    data = request.get_json()
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Invalid input. 'texts' (list of strings) is required."}), 400

    texts_to_embed = data["texts"]

    try:
        print(f"Received {len(texts_to_embed)} texts for embedding.")
        embedding_results = llm.create_embedding(texts_to_embed)

        # Extract embeddings correctly based on llama-cpp-python structure
        # For a list of input texts, embedding_results['data'] is a list of dicts,
        # each dict is like {'object': 'embedding', 'embedding': [vector], 'index': i}
        embeddings = [item['embedding'] for item in embedding_results['data']]
        print(f"Generated {len(embeddings)} embeddings.")

        return jsonify({"embeddings": embeddings})
    except Exception as e:
        print(f"Error during embedding: {e}")
        return jsonify({"error": f"Error during embedding: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
