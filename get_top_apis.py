import json
import torch
import faiss
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

# Load schema
with open("schema.json", "r") as f:
    schema = json.load(f)

# Get schema version
schema_version = schema.get('version', '1.0')  # Default to 1.0 if version not specified

# Cache directory
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache file paths
INDEX_FILE = os.path.join(CACHE_DIR, "api_index.bin")
METADATA_FILE = os.path.join(CACHE_DIR, "api_metadata.json")
VERSION_FILE = os.path.join(CACHE_DIR, "schema_version.txt")
TFIDF_FILE = os.path.join(CACHE_DIR, "tfidf_matrix.npy")
TFIDF_VECTORIZER_FILE = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")

def build_index(schema_entries):
    summaries = []
    metadata = []
    for entry in schema_entries["apis"]:
        # Start with operationId as it's the most descriptive
        operation_id = entry.get('operationId', '')
        description = entry.get("description", "")
        
        # Create a weighted summary that emphasizes the operation's purpose
        summary_parts = []
        
        # Add operationId with very high weight (repeated more times for emphasis)
        if operation_id:
            # Repeat operationId 5 times for stronger emphasis
            summary_parts.extend([operation_id] * 5)
            
            # Add operationId with spaces between words for better matching
            # e.g., "createEvent" becomes "create event"
            import re
            spaced_operation = ' '.join(re.findall('[A-Z][a-z]*', operation_id)).lower()
            summary_parts.extend([spaced_operation] * 3)
        
        # Add description if available
        if description:
            summary_parts.append(description)
            
        # Add method and path with lower weight
        summary_parts.append(f"{entry['method']} {entry['path']}")
        
        # Add request body or query params with lowest weight
        if entry.get("method") == "GET":
            query_params = entry.get("parameters", [])
            if query_params:
                summary_parts.append(f"Query params: {query_params}")
        else:
            request_body = entry.get("requestBody", {})
            if request_body:
                summary_parts.append(f"Request body: {request_body}")
        
        # Join all parts with spaces
        summary = " ".join(summary_parts)
        summaries.append(summary)
        metadata.append(entry)

    # Build embedding index
    embeddings = embed(summaries)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Build TF-IDF
    # vectorizer = TfidfVectorizer(
    #     stop_words='english',
    #     ngram_range=(1, 2),  # Use both unigrams and bigrams
    #     max_features=1000
    # )
    # tfidf_matrix = vectorizer.fit_transform(summaries)

    # return index, metadata, summaries, tfidf_matrix, vectorizer
    return index, metadata, summaries, None, None

def load_cached_index():
    """Load the cached index and metadata if they exist and schema version hasn't changed."""
    if not all(os.path.exists(f) for f in [INDEX_FILE, METADATA_FILE, VERSION_FILE, TFIDF_FILE, TFIDF_VECTORIZER_FILE]):
        return None, None, None, None, None
    
    # Check if schema version has changed
    with open(VERSION_FILE, 'r') as f:
        cached_version = f.read().strip()
    if cached_version != schema_version:
        print(f"Schema version changed from {cached_version} to {schema_version}")
        return None, None, None, None, None
    
    # Load cached data
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        # tfidf_matrix = np.load(TFIDF_FILE)
        # import pickle
        # with open(TFIDF_VECTORIZER_FILE, 'rb') as f:
        #     vectorizer = pickle.load(f)
        # return index, metadata, None, tfidf_matrix, vectorizer
        return index, metadata, None, None, None
    except Exception as e:
        print(f"Error loading cached index: {e}")
        return None, None, None, None, None

def save_index_to_cache(index, metadata, tfidf_matrix, vectorizer):
    """Save the index and metadata to cache."""
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
        with open(VERSION_FILE, 'w') as f:
            f.write(schema_version)
        np.save(TFIDF_FILE, tfidf_matrix)
        import pickle
        with open(TFIDF_VECTORIZER_FILE, 'wb') as f:
            pickle.dump(vectorizer, f)
    except Exception as e:
        print(f"Error saving index to cache: {e}")

def search_apis(query, index, metadata, tfidf_matrix, vectorizer, top_k=3):
    """
    Search APIs using both embedding and TF-IDF approaches.
    Returns combined results with scores.
    """
    # Embedding-based search
    query_emb = embed([query])
    embedding_scores, embedding_ids = index.search(query_emb, top_k)
    print("Cosine similarity between query and top 3 APIs: ", embedding_scores)
    
    # TF-IDF search
    # query_tfidf = vectorizer.transform([query])
    # tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    # tfidf_ids = np.argsort(tfidf_scores)[-top_k:][::-1]
    
    # Combine results
    results = []
    seen_ids = set()
    
    # Add embedding results
    for score, idx in zip(embedding_scores[0], embedding_ids[0]):
        if idx not in seen_ids:
            results.append({
                'api': metadata[idx],
                'embedding_score': float(score),
                # 'tfidf_score': float(tfidf_scores[idx]),
                # 'combined_score': float(score) * 0.7 + float(tfidf_scores[idx]) * 0.3
            })
            seen_ids.add(idx)
    
    # Add TF-IDF results
    # for idx in tfidf_ids:
    #     if idx not in seen_ids:
    #         results.append({
    #             'api': metadata[idx],
    #             'embedding_score': float(embedding_scores[0][0]),  # Use best embedding score
    #             'tfidf_score': float(tfidf_scores[idx]),
    #             'combined_score': float(embedding_scores[0][0]) * 0.7 + float(tfidf_scores[idx]) * 0.3
    #         })
    #         seen_ids.add(idx)
    
    # Sort by combined score
    results.sort(key=lambda x: x['embedding_score'], reverse=True)
    return results[:top_k]

def visualize_embeddings(query, index, metadata, tfidf_matrix, vectorizer, top_indices=None, use_pca=False):
    """Visualize the embeddings of APIs, the query, and highlight top search results."""
    # Get all API summaries
    texts, labels = [], []
    for entry in metadata:
        label = entry.get("operationId", entry["path"])
        summary = f"{entry['method']} {entry['path']} ({label})"
        if entry.get("method") == "GET":
            query_params = entry.get("parameters", [])
            if query_params:
                summary += f"Query params: {query_params}"
        else:
            request_body = entry.get("requestBody", {})
            if request_body:
                summary += f"Request body: {request_body}"
        texts.append(summary)
        labels.append(label)

    # Add query
    texts.append(query)
    labels.append("QUERY")

    # Get embeddings
    vecs = embed(texts)

    # t-SNE
    if use_pca:
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = reducer.fit_transform(vecs)

    # Plot
    plt.figure(figsize=(12, 8))
    n_apis = len(metadata)
    
    # Plot API embeddings (default blue)
    plt.scatter(reduced[:n_apis, 0], reduced[:n_apis, 1], c='blue', alpha=0.6, label='APIs')

    # Highlight top search results in red
    if top_indices is not None:
        plt.scatter(reduced[top_indices, 0], reduced[top_indices, 1], c='red', s=80, label='Top Results')

    # Plot query
    plt.scatter(reduced[-1, 0], reduced[-1, 1], c='green', s=100, label='Query')

    # Add labels
    for i, label in enumerate(labels):
        if label == "QUERY":
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), 
                        fontsize=10, color='green', weight='bold')
        elif top_indices is not None and i in top_indices:
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), 
                        fontsize=9, color='red', weight='bold')
        else:
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), 
                        fontsize=7, color='blue')

    if use_pca:
        plt.title("PCA Visualization of API Embeddings (Top Results Highlighted)")
    else:
        plt.title("t-SNE Visualization of API Embeddings (Top Results Highlighted)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

def main():
    query = "Delete event with id 123"
    parser = argparse.ArgumentParser(description='Search APIs using semantic search')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the embeddings')
    parser.add_argument('--pca', action='store_true', help='Use PCA instead of t-SNE for visualization')
    args = parser.parse_args()

    # Initialize the semantic search index
    index, metadata, _, _, _ = load_cached_index()
    if index is None:
        print("Building new index...")
        index, metadata, _, _, _ = build_index(schema)
        save_index_to_cache(index, metadata, None, None)
        print("Index built and cached successfully!")
    else:
        print("Using cached index...")

    # Search APIs
    results = search_apis(query, index, metadata, None, None)

    # Show results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualize if requested
    if args.visualize:
        # Find indices of top results in metadata
        top_labels = [r['api'].get('operationId', r['api']['path']) for r in results]
        top_indices = [i for i, entry in enumerate(metadata) if entry.get('operationId', entry['path']) in top_labels]
        visualize_embeddings(query, index, metadata, None, None, top_indices=top_indices, use_pca=args.pca)

if __name__ == "__main__":
    main() 