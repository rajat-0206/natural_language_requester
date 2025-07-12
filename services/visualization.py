"""
Visualization Service - Handles API search visualization and analysis.
"""

import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import base64
import io
from utils.index import embed, search_apis_scann

class VisualizationService:
    """Service for visualizing API search results and embeddings."""
    
    def __init__(self, search_service):
        self.search_service = search_service
    
    def visualize_embeddings(self, query, top_indices=None, use_pca=False):
        """Visualize embeddings for API search results."""
        try:
            # Get all API summaries for visualization
            summaries = []
            metadata = []
            
            for entry in self.search_service.api_schema["apis"]:
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
            
            # Generate embeddings
            embeddings = embed(summaries)
            
            # Dimensionality reduction
            if use_pca:
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(embeddings)
            else:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                reduced_embeddings = reducer.fit_transform(embeddings)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot all APIs
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                       alpha=0.6, s=50, c='lightblue', label='All APIs')
            
            # Highlight top results if provided
            if top_indices is not None:
                top_embeddings = reduced_embeddings[top_indices]
                plt.scatter(top_embeddings[:, 0], top_embeddings[:, 1], 
                           alpha=0.8, s=100, c='red', label='Top Results')
                
                # Add labels for top results
                for i, idx in enumerate(top_indices):
                    if idx < len(metadata):
                        api = metadata[idx]
                        label = f"{api.get('method', '')} {api.get('operationId', api.get('path', ''))}"
                        plt.annotate(label, (top_embeddings[i, 0], top_embeddings[i, 1]), 
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            plt.title(f'API Embeddings Visualization\nQuery: "{query}"')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return {
                'image': img_str,
                'total_apis': len(metadata),
                'top_results_count': len(top_indices) if top_indices else 0,
                'reduction_method': 'PCA' if use_pca else 't-SNE'
            }
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            print(traceback.format_exc())
            return {'error': str(e)}
    
    def get_search_analysis(self, query, top_k=5):
        """Get detailed analysis of search results."""
        try:
            # Get search results
            if self.search_service.search_model == "faiss":
                query_emb = embed([query])
                _, top_ids = self.search_service.index.search(query_emb, top_k)
                results = []
                for i, idx in enumerate(top_ids[0]):
                    if idx < len(self.search_service.metadata):
                        results.append({
                            'api': self.search_service.metadata[idx],
                            'score': float(top_ids[1][0][i]),
                            'rank': i + 1
                        })
            elif self.search_service.search_model == "scann":
                results = search_apis_scann(query, self.search_service.scann_index, 
                                         self.search_service.scann_metadata, None, None, top_k=top_k)
            else:
                return {'error': f'Unknown search model: {self.search_service.search_model}'}
            
            # Get top indices for visualization
            top_indices = []
            for result in results:
                if self.search_service.search_model == "faiss":
                    # Find the index of this API in the metadata
                    for i, api in enumerate(self.search_service.metadata):
                        if api == result['api']:
                            top_indices.append(i)
                            break
                else:
                    # For SCANN, we need to find the index differently
                    # This is a simplified approach
                    pass
            
            # Generate visualization
            viz_result = self.visualize_embeddings(query, top_indices)
            
            return {
                'query': query,
                'results': results,
                'visualization': viz_result,
                'search_model': self.search_service.search_model,
                'total_results': len(results)
            }
            
        except Exception as e:
            print(f"Error in search analysis: {e}")
            import traceback
            print(traceback.format_exc())
            return {'error': str(e)} 