import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class MatrixFactorizationPredictor(Predictor):
    
    def fit(self, uim: UserItemData):
        self.rating_matrix = uim.data.pivot(index='userID', columns='movieID', values='rating').fillna(0)

    def predict(self, userID):
        ratings = self.rating_matrix.values
        # mean user ratings for each user
        ur_mean = np.mean(ratings, axis=1).reshape(-1, 1)
        normalized_r = ratings - ur_mean
        # Singular Value Decomposition
        users, sigma, values = svds(normalized_r, k=50)
        # convert to diagonal form for matrix multiplication
        sigma = np.diag(sigma)
        predictions = np.dot(np.dot(users, sigma), values) + ur_mean
        # Convert to dataframe for easier row retrieval
        preds_df = pd.DataFrame(predictions, columns=self.rating_matrix.columns)

        return {m: pred for m, pred in enumerate(preds_df.iloc[userID, :])}
