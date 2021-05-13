from Models.MovieData import MovieData
from Models.Recommender import Recommender
from Models.UserItemData import UserItemData

from Predictors.Predictor import Predictor


class ViewsPredictor(Predictor):
    def fit(self, uim: UserItemData):
        self.data = uim.data.groupby('movieID').count()

    def predict(self, userID):
        pred_dict = {}
        for movieID, views in zip(list(self.data.index), self.data['rating'].values):
            pred_dict[movieID] = views
        return pred_dict

# uim = UserItemData('../movielens/user_ratedmovies.dat')
# md = MovieData('../movielens/movies.dat')
# v = ViewsPredictor()
# rec = Recommender(v)
# rec.fit(uim)
# res = rec.recommend(75, n=5, rec_seen=True)
# for idmovie, val in res:
#     print("Movie: {}, number of ratings: {}".format(md.get_title(idmovie), val))
