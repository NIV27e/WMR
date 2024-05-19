import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class CollaborativeRecommender:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')

    def fit(self):
        self.model.fit(self.user_item_matrix)

    def recommend(self, user_id, top_n=10):
        user_vector = self.user_item_matrix[user_id].reshape(1, -1)  # Reshape to 2D array
        n_neighbors = min(top_n + 1, len(self.user_item_matrix))  # Ensure n_neighbors does not exceed available samples
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_neighbors)
        return indices.flatten()[1:]  # Exclude the first result as it is the input user itself
