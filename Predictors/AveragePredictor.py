from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class AveragePredictor(Predictor):
    def __init__(self, b):
        if b < 0:
            raise Exception("Invalid value for parameter b")
        else:
            self.b = b

    def fit(self, uim: UserItemData):
        movies = uim.data.groupby('movieID')
        n = movies.count()['rating']
        vs = movies.sum()['rating']
        g_avg = uim.data['rating'].mean()
        self.data_avg = (vs + self.b * g_avg) / (n + self.b)

    def predict(self, userID):
        return {movieID: rating for movieID, rating in zip(list(self.data_avg.index), self.data_avg.values)}
