import pandas as pd


class HybridRecommender:
    def __init__(self, content_model, collaborative_model):
        self.content_model = content_model
        self.collaborative_model = collaborative_model

    def recommend(self, user_id, item_id, top_n=10):
        content_recs = self.content_model.recommend(item_id, top_n)
        collaborative_indices = self.collaborative_model.recommend(user_id, top_n)
        collaborative_recs = self.content_model.data.iloc[collaborative_indices]
        hybrid_recs = pd.concat([content_recs, collaborative_recs]).drop_duplicates().head(top_n)
        return hybrid_recs
