import pandas as pd

from Models.MovieData import MovieData
from Models.Recommender import Recommender
from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class AveragePredictor(Predictor):
    def __init__(self, b):
        if b < 0:
            raise Exception("Invalid value for parameter: b")
        else:
            self.b = b

    def fit(self, uim: UserItemData):
        movies = uim.data.groupby('movieID')
        n = movies.count()['rating']
        vs = movies.sum()['rating']
        g_avg = uim.data['rating'].mean()
        avg = (vs + self.b * g_avg) / (n + self.b)
        self.data = uim.data
        self.data_avg = pd.merge(avg, uim.data, on='movieID', how='inner')

    def predict(self, userID):
        pred_dict = {}
        movies = list(set(self.data['movieID']))
        for movieID in movies:
            value = self.data_avg[self.data_avg['movieID'] == movieID]['rating_x'].values[0]
            pred_dict[movieID] = value
        return pred_dict

# a = AveragePredictor(b=100)
# u = UserItemData('../movielens/user_ratedmovies.dat')
# rec = Recommender(a)
# rec.fit(u)
# res = rec.recommend(75, 10, False)
# md = MovieData('../movielens/movies.dat')
#
# for idmovie, val in res:
#     print("Movie: {}, number of ratings: {}".format(md.get_title(idmovie), val))