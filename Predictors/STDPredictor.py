import numpy as np

from Predictors.Predictor import Predictor


class STDPredictor(Predictor):
    def __init__(self, min_ratings):
        self.min_ratings = min_ratings

    def fit(self, uim):
        grouped = uim.data.groupby('movieID')
        self.data = uim.data[grouped.userID.transform(len) > self.min_ratings]
        # Aggregate each movie's ratings to their standard deviation
        ratings = grouped.agg(np.std, ddof=0)['rating']
        # Assume a movie is controversial if the standard deviation of its ratings is >= 1.5
        self.ratings_std_dev = ratings[ratings >= 1.5]

    def predict(self, userID):
        return {movieID: std_dev for movieID, std_dev in zip(self.ratings_std_dev.keys(), self.ratings_std_dev.values)}