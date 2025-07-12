import json
import torch
import faiss
import numpy as np
import os
import argparse
try:
    import scann
    SCANN_AVAILABLE = True
except ImportError:
    print("Warning: SCANN not available. Using scikit-learn NearestNeighbors as fallback.")
    from sklearn.neighbors import NearestNeighbors
    SCANN_AVAILABLE = False
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

# SCANN cache file paths
SCANN_INDEX_FILE = os.path.join(CACHE_DIR, "scann_api_index.pkl")

def build_index(schema_entries):
    summaries = []
    metadata = []
    for entry in schema_entries["apis"]:
        # Start with operationId as it's the most descriptive
        operation_id = entry.get('operationId', '')
        description = entry.get("description", "")
        method = entry.get('method', '')
        
        # Create a weighted summary that emphasizes the operation's purpose
        summary_parts = []
        
        # Add HTTP method with HIGHEST weight and descriptive action words
        method_action_map = {
            'GET': 'fetch retrieve get read obtain',
            'POST': 'create add new insert',
            'PATCH': 'update modify edit change',
            'PUT': 'update replace modify edit change',
            'DELETE': 'remove delete destroy eliminate'
        }
        
        # Add method with very high weight (repeated many times for emphasis)
        if method in method_action_map:
            # Repeat method 8 times for strongest emphasis
            summary_parts.extend([method] * 8)
            # Add descriptive action words for the method
            summary_parts.extend(method_action_map[method].split() * 4)
        
        # Add operationId with high weight (repeated for emphasis)
        if operation_id:
            # Repeat operationId 5 times for emphasis
            summary_parts.extend([operation_id] * 5)
            
            # Add operationId with spaces between words for better matching
            # e.g., "createEvent" becomes "create event"
            import re
            spaced_operation = ' '.join(re.findall('[A-Z][a-z]*', operation_id)).lower()
            summary_parts.extend([spaced_operation] * 3)
        
        # Add description if available
        if description:
            summary_parts.append(description)
            
        # Add method and path with medium weight
        summary_parts.append(f"{method} {entry['path']}")
        
        # Add request body or query params with lowest weight
        if method == "GET":
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

def build_scann_index(schema_entries):
    """Build SCANN index for API search."""
    summaries = []
    metadata = []
    for entry in schema_entries["apis"]:
        # Start with operationId as it's the most descriptive
        operation_id = entry.get('operationId', '')
        description = entry.get("description", "")
        method = entry.get('method', '')
        
        # Create a weighted summary that emphasizes the operation's purpose
        summary_parts = []
        
        # Add HTTP method with HIGHEST weight and descriptive action words
        method_action_map = {
            'GET': 'fetch retrieve get read obtain',
            'POST': 'create add new insert',
            'PATCH': 'update modify edit change',
            'PUT': 'update replace modify edit change',
            'DELETE': 'remove delete destroy eliminate'
        }
        
        # Add method with very high weight (repeated many times for emphasis)
        if method in method_action_map:
            # Repeat method 8 times for strongest emphasis
            summary_parts.extend([method] * 8)
            # Add descriptive action words for the method
            summary_parts.extend(method_action_map[method].split() * 4)
        
        # Add operationId with high weight (repeated for emphasis)
        if operation_id:
            # Repeat operationId 5 times for emphasis
            summary_parts.extend([operation_id] * 5)
            
            # Add operationId with spaces between words for better matching
            # e.g., "createEvent" becomes "create event"
            import re
            spaced_operation = ' '.join(re.findall('[A-Z][a-z]*', operation_id)).lower()
            summary_parts.extend([spaced_operation] * 3)
        
        # Add description if available
        if description:
            summary_parts.append(description)
            
        # Add method and path with medium weight
        summary_parts.append(f"{method} {entry['path']}")
        
        # Add request body or query params with lowest weight
        if method == "GET":
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
    
    if SCANN_AVAILABLE:
        # Build SCANN index
        # Configure SCANN for cosine similarity search
        normalized_dataset = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # Configure SCANN parameters
        scann_index = scann.scann_ops_pybind.builder(normalized_dataset, top_k=10, distance_measure=scann.scann_ops_pybind.DotProduct())
        scann_index = scann_index.tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
        scann_index = scann_index.score_ah(2, anisotropic_quantization_threshold=0.2)
        scann_index = scann_index.reorder(100)
        scann_index = scann_index.build()
    else:
        # Use scikit-learn NearestNeighbors as fallback
        from sklearn.neighbors import NearestNeighbors
        scann_index = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
        scann_index.fit(embeddings)

    return scann_index, metadata, summaries, None, None

def load_cached_scann_index():
    """Load the cached SCANN index and metadata if they exist and schema version hasn't changed."""
    if not all(os.path.exists(f) for f in [SCANN_INDEX_FILE, METADATA_FILE, VERSION_FILE]):
        return None, None, None, None, None
    
    # Check if schema version has changed
    with open(VERSION_FILE, 'r') as f:
        cached_version = f.read().strip()
    if cached_version != schema_version:
        print(f"Schema version changed from {cached_version} to {schema_version}")
        return None, None, None, None, None
    
    # Load cached data
    try:
        import pickle
        with open(SCANN_INDEX_FILE, 'rb') as f:
            scann_index = pickle.load(f)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return scann_index, metadata, None, None, None
    except Exception as e:
        print(f"Error loading cached SCANN index: {e}")
        return None, None, None, None, None

def save_scann_index_to_cache(scann_index, metadata, tfidf_matrix, vectorizer):
    """Save the SCANN index and metadata to cache."""
    try:
        import pickle
        with open(SCANN_INDEX_FILE, 'wb') as f:
            pickle.dump(scann_index, f)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
        with open(VERSION_FILE, 'w') as f:
            f.write(schema_version)
        np.save(TFIDF_FILE, tfidf_matrix)
        with open(TFIDF_VECTORIZER_FILE, 'wb') as f:
            pickle.dump(vectorizer, f)
    except Exception as e:
        print(f"Error saving SCANN index to cache: {e}")

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

def search_apis_scann(query, scann_index, metadata, tfidf_matrix, vectorizer, top_k=3):
    """
    Search APIs using SCANN for nearest neighbor search.
    Returns results with scores.
    """
    # Embedding-based search using SCANN
    query_emb = embed([query])
    
    if SCANN_AVAILABLE:
        # Normalize query for cosine similarity
        normalized_query = query_emb / np.linalg.norm(query_emb, axis=1)[:, np.newaxis]
        
        # Search using SCANN
        neighbors, distances = scann_index.search_batched(normalized_query, final_num_neighbors=top_k)
        
        # Convert distances to similarities (SCANN returns dot products for normalized vectors)
        # For cosine similarity, dot product of normalized vectors equals cosine similarity
        similarities = distances[0]  # SCANN returns dot products which are cosine similarities for normalized vectors
        neighbor_indices = neighbors[0]
    else:
        # Use scikit-learn NearestNeighbors
        distances, neighbor_indices = scann_index.kneighbors(query_emb, n_neighbors=top_k)
        # Convert distances to similarities (cosine distance to cosine similarity)
        similarities = 1 - distances[0]  # cosine_similarity = 1 - cosine_distance
    
    print("Cosine similarity between query and top 3 APIs (SCANN/sklearn): ", similarities)
    
    # Combine results
    results = []
    seen_ids = set()
    
    # If sklearn, neighbor_indices may be 2D; flatten if needed
    if not SCANN_AVAILABLE and hasattr(neighbor_indices, 'ndim') and neighbor_indices.ndim == 2:
        neighbor_indices = neighbor_indices[0]
    # Add SCANN/sklearn results
    for similarity, idx in zip(similarities, neighbor_indices):
        idx_int = int(idx)
        if idx_int not in seen_ids:
            results.append({
                'api': metadata[idx_int],
                'scann_score': float(similarity),
            })
            seen_ids.add(idx_int)
    
    # Sort by SCANN score
    results.sort(key=lambda x: x['scann_score'], reverse=True)
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
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
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
    parser.add_argument('--method', choices=['faiss', 'scann', 'both'], default='faiss',
                       help='Search method to use: faiss, scann, or both for comparison')
    args = parser.parse_args()

    if args.method in ['faiss', 'both']:
        print("=== FAISS Search ===")
        # Initialize the FAISS semantic search index
        index, metadata, _, _, _ = load_cached_index()
        if index is None:
            print("Building new FAISS index...")
            index, metadata, _, _, _ = build_index(schema)
            save_index_to_cache(index, metadata, None, None)
            print("FAISS index built and cached successfully!")
        else:
            print("Using cached FAISS index...")

        # Search APIs using FAISS
        faiss_results = search_apis(query, index, metadata, None, None)

        # Show FAISS results
        with open("faiss_results.json", "w") as f:
            json.dump(faiss_results, f, indent=2)
        print("FAISS results saved to faiss_results.json")

    if args.method in ['scann', 'both']:
        print("\n=== SCANN Search ===")
        # Initialize the SCANN semantic search index
        scann_index, scann_metadata, _, _, _ = load_cached_scann_index()
        if scann_index is None:
            print("Building new SCANN index...")
            scann_index, scann_metadata, _, _, _ = build_scann_index(schema)
            save_scann_index_to_cache(scann_index, scann_metadata, None, None)
            print("SCANN index built and cached successfully!")
        else:
            print("Using cached SCANN index...")

        # Search APIs using SCANN
        scann_results = search_apis_scann(query, scann_index, scann_metadata, None, None)

        # Show SCANN results
        with open("scann_results.json", "w") as f:
            json.dump(scann_results, f, indent=2)
        print("SCANN results saved to scann_results.json")

    if args.method == 'both':
        print("\n=== Comparison ===")
        print("Comparing top results from both methods:")
        
        print("\nFAISS Top Results:")
        for i, result in enumerate(faiss_results[:3]):
            api = result['api']
            print(f"{i+1}. {api.get('method', '')} {api.get('path', '')} (Score: {result['embedding_score']:.4f})")
        
        print("\nSCANN Top Results:")
        for i, result in enumerate(scann_results[:3]):
            api = result['api']
            print(f"{i+1}. {api.get('method', '')} {api.get('path', '')} (Score: {result['scann_score']:.4f})")

    # Visualize if requested (using FAISS for visualization)
    if args.visualize and args.method in ['faiss', 'both']:
        # Find indices of top results in metadata
        top_labels = [r['api'].get('operationId', r['api']['path']) for r in faiss_results]
        top_indices = [i for i, entry in enumerate(metadata) if entry.get('operationId', entry['path']) in top_labels]
        visualize_embeddings(query, index, metadata, None, None, top_indices=top_indices, use_pca=args.pca)

if __name__ == "__main__":
    main() 