
# web_ml_recommender

`web_ml_recommender` is a Python package that provides machine learning-based recommendation systems for web development. It includes content-based filtering, collaborative filtering, and a hybrid recommendation system.

## Installation

```sh
pip install web_ml_recommender
```

## Usage

### Loading Sample Data

```python
from web_ml_recommender.utils import load_sample_data

data = load_sample_data('web_ml_recommender/data/sample_data.csv')
```

### Content-Based Recommender

```python
from web_ml_recommender import ContentBasedRecommender

content_rec = ContentBasedRecommender(data, 'description')
content_rec.fit()
recommendations = content_rec.recommend(item_id=1, top_n=5)
print(recommendations)
```

### Collaborative Recommender

```python
from web_ml_recommender import CollaborativeRecommender
import numpy as np

# Example user-item matrix
user_item_matrix = np.array([[4, 0, 0, 5, 1], [5, 5, 4, 0, 0], [0, 0, 5, 4, 4]])

collab_rec = CollaborativeRecommender(user_item_matrix)
collab_rec.fit()
recommendations = collab_rec.recommend(user_id=0, top_n=5)
print(recommendations)
```

### Hybrid Recommender

```python
from web_ml_recommender import HybridRecommender

hybrid_rec = HybridRecommender(content_rec, collab_rec)
recommendations = hybrid_rec.recommend(user_id=0, item_id=1, top_n=5)
print(recommendations)
```

## License

MIT
