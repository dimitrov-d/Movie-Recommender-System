import pandas as pd

from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class SlopeOnePredictor(Predictor):
    def fit(self, uim: UserItemData):
        matrix = pd.pivot_table(uim.data, index="userID", columns="movieID", values="rating").fillna("?")
        self.users = matrix.index.values
        self.movies = matrix.columns.values
        self.matrix = matrix

    def predict(self, userID: int):
        pred_dict = {}
        for movie in self.movies:
            # Find a movie that a user has not rated
            if self.matrix.loc[userID, movie] == "?":
                dev = []
                freq = []
                # Loop through each other movie and compare user ratings
                for col in self.movies:
                    if col == movie:
                        continue
                    diff = []

                    for row in self.users:
                        if self.matrix.loc[row, movie] == "?" or self.matrix.loc[row, col] == "?":
                            continue
                        diff.append(self.matrix.loc[row, movie] - self.matrix.loc[row, col])

                    dev.append(sum(diff) / len(diff))
                    freq.append(len(diff))
                pred_dict[movie] = round(self.get_prediction(userID, dev, freq), 1)
        return pred_dict

    def get_prediction(self, userID, dev, freq):
        rated_movies = list(filter(lambda m: m != "?", self.matrix.loc[userID][:].values))
        res = []
        for de, rm, fre in zip(dev, rated_movies, freq):
            res.append((de + rm) * fre)
        return round(sum(res) / sum(freq), 2)