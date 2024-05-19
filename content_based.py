
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, data, feature_column):
        """
        Initializes the ContentBasedRecommender with the data and the feature column.

        Parameters:
        data (pd.DataFrame): The dataset containing the items.
        feature_column (str): The column name containing the textual data for content-based filtering.
        """
        self.data = data
        self.feature_column = feature_column
        self.tfidf_matrix = None
        self.cosine_sim = None

    def fit(self):
        """Fits the TF-IDF vectorizer on the feature column and computes the cosine similarity matrix."""
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data[self.feature_column])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, item_id, top_n=10):
        """
        Recommends items similar to the given item_id.

        Parameters:
        item_id (int): The ID of the item to base the recommendations on.
        top_n (int): The number of recommendations to return (default is 10).

        Returns:
        pd.DataFrame: A DataFrame containing the recommended items.
        """
        if self.cosine_sim is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before calling 'recommend'.")
        
        idx = self.data.index[self.data['id'] == item_id][0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        item_indices = [i[0] for i in sim_scores]
        return self.data.iloc[item_indices]
