import numpy as np
import pandas as pd
from scipy import spatial

from Models.MovieData import MovieData
from Models.Recommender import Recommender
from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class ItemBasedPredictor(Predictor):
    def __init__(self, min_values=0, threshold=0):
        self.min_values = min_values
        self.threshold = threshold

    def fit(self, uim: UserItemData):
        self.data = uim.data
        # Unique list of all rated movies
        movies = list(set(self.data['movieID'].values))
        self.movies = movies

        similarities = {}
        for i, movie1 in enumerate(movies):
            for j, movie2 in enumerate(movies):
                if j > i:
                    similarities[f'{movie1}-{movie2}'] = self.calculate_similarity(movie1, movie2)

        self.similarities = similarities

    def predict(self, userID):
        # Movies the user has watched
        userMovies = self.data[self.data['userID'] == userID]['movieID'].values
        pred_dict = {}
        for m in self.movies:
            nominator = []
            denominator = 0
            for um in userMovies:
                if um == m:
                    continue
                rating = self.data[(self.data['userID'] == userID) & (self.data['movieID'] == um)]['rating'].values[0]
                sim = self.similarity(m, um)
                nominator.append(sim * rating)
                denominator = denominator + sim
            pred_dict[m] = sum(nominator / np.float64(denominator))
        return pred_dict

    def calculate_similarity(self, movie1_ID, movie2_ID):
        movie1 = self.data[self.data['movieID'] == movie1_ID]
        movie2 = self.data[self.data['movieID'] == movie2_ID]
        common_ratings = pd.merge(movie1, movie2, on='userID', how='inner')
        if common_ratings.shape[0] < self.min_values:
            return 0
        m1_ratings = common_ratings['rating_x'].values
        m2_ratings = common_ratings['rating_y'].values

        cosine_sim = 1 - spatial.distance.cosine(m1_ratings, m2_ratings)
        return 0 if cosine_sim < self.threshold else cosine_sim

    def similarity(self, movie1, movie2):
        # In case the order of the movies in dictionary is different
        try:
            return self.similarities[f'{movie1}-{movie2}']
        except:
            return self.similarities[f'{movie2}-{movie1}']

    def most_similar_movies(self, md: MovieData):
        most_similar = sorted(self.similarities.items(), key=lambda item: item[1], reverse=True)[:20]
        for key, sim in most_similar:
            [m1, m2] = key.split('-')
            print(f'Movie1: {md.get_title(m1)}, Movie2: {md.get_title(m2)}, similarity: {sim}')

    def similar_items(self, movieID, n):
        similar_dict = {}
        for movie in self.movies:
            if movie != movieID:
                similar_dict[movie] = self.similarities[f'{movie}-{movieID}']
        return sorted(similar_dict.items(), key=lambda item: item[1], reverse=True)[:n]

    def self_recommendation(self):
        # I created a new user with id 99999 (me) and rated 20 movies
        me = 99999
        pred_dict = self.predict(me)
        self_rated_movies = self.data[self.data['userID'] == me]['movieID'].values
        result_dict = {x: pred_dict[x] for x in pred_dict.keys() if x not in self_rated_movies}

        return sorted(result_dict.items(), key=lambda item: item[1], reverse=True)[:10]


md = MovieData('../movielens/movies.dat')
uim = UserItemData('../movielens/user_ratedmovies.dat', min_ratings=1000)
rp = ItemBasedPredictor()
rec = Recommender(rp)
rec.fit(uim)