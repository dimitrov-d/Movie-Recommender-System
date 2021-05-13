import random

from Models.MovieData import MovieData
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


# md = MovieData('../movielens/movies.dat')
# uim = UserItemData('../movielens/user_ratedmovies.dat')
# rp = RandomPredictor(1, 5)
# rp.fit(uim)
# pred = rp.predict(78)
# print(type(pred))
# items = [1, 3, 20, 50, 100]
# for item in items:
#     print("Movie: {}, score: {}".format(md.get_title(item), pred[item]))