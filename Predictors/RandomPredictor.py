import random

from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class RandomPredictor(Predictor):
    def __init__(self, min=1, max=5):
        self.min = min
        self.max = max

    def fit(self, uim: UserItemData):
        self.movies = list(set(uim.data['movieID']))

    def predict(self, userID):
        pred_dict = {}
        for movie in self.movies:
            pred_dict[movie] = random.randint(self.min, self.max)
        return pred_dict