import numpy as np

from Models.MovieData import MovieData
from Models.Recommender import Recommender
from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class STDPredictor(Predictor):
    def __init__(self, min_ratings):
        self.min_ratings = min_ratings

    def fit(self, uim):
        grouped = uim.data.groupby('movieID')
        self.data = uim.data[grouped.userID.transform(len) > self.min_ratings]
        ratings = grouped.agg(np.std, ddof=0)['rating']
        # Assume a movie is controversial if the standard deviation of its ratings is >= 1.5
        self.ratings_std_dev = ratings[ratings >= 1.5]

    def predict(self, userID):
        pred_dict = {}
        for movieID, std_dev in zip(self.ratings_std_dev.keys(), self.ratings_std_dev.values):
            pred_dict[movieID] = std_dev

        return pred_dict


# md = MovieData('../movielens/movies.dat')
# uim = UserItemData('../movielens/user_ratedmovies.dat')
# rp = STDPredictor(100)
# rec = Recommender(rp)
# rec.fit(uim)
# rec_items = rec.recommend(78, n=10, rec_seen=True)
# for idmovie, val in rec_items:
#     print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
