# test_script.py

from web_ml_recommender.utils import load_sample_data
from web_ml_recommender import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
import numpy as np
import os

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load sample data using the absolute path
data_path = os.path.join(current_dir, 'data/sample_data.csv')
data = load_sample_data(data_path)

# Test Content-Based Recommender
print("Testing Content-Based Recommender")
content_rec = ContentBasedRecommender(data, 'description')
content_rec.fit()
content_recommendations = content_rec.recommend(item_id=1, top_n=5)
print("Content-Based Recommendations:")
print(content_recommendations)

# Test Collaborative Recommender
print("\nTesting Collaborative Recommender")
# Example user-item matrix
user_item_matrix = np.array([
    [4, 0, 0, 5, 1],
    [5, 5, 4, 0, 0],
    [0, 0, 5, 4, 4]
])
collab_rec = CollaborativeRecommender(user_item_matrix)
collab_rec.fit()
collab_recommendations = collab_rec.recommend(user_id=0, top_n=5)
print("Collaborative Recommendations:")
print(collab_recommendations)

# Test Hybrid Recommender
print("\nTesting Hybrid Recommender")
hybrid_rec = HybridRecommender(content_rec, collab_rec)
hybrid_recommendations = hybrid_rec.recommend(user_id=0, item_id=1, top_n=5)
print("Hybrid Recommendations:")
print(hybrid_recommendations)
