import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from Models.MovieData import MovieData
from Models.MovieTags import MovieTags
from Models.Recommender import Recommender
from Predictors.Predictor import Predictor


class ContentBasedPredictor(Predictor):
    def fit(self, mt: MovieTags):
        # use TF-IDF for content based predictions
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(mt.data['tags'])
        # Construct a similarity matrix between all movies
        self.data = mt.data
        self.similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        # Map movie IDs to index in tags data frame
        self.indices = pd.Series(mt.data.index, index=mt.data['movieID']).drop_duplicates()

    def predict(self, movieID: int):
        # Get the row index of the movie that matches the movie ID
        index = self.indices[movieID]
        similarities = list(enumerate(self.similarity_matrix[index]))
        return {self.data.iloc[sim[0]]['movieID']: sim[1] for sim in similarities}


# md = MovieData('../movielens/movies.dat')
# mt = MovieTags(md, '../movielens/movie_tags.dat', '../movielens/tags.dat')
# cb = ContentBasedPredictor()
# cb.fit(mt)
# res = cb.predict(2571)
# for idm, sim in sorted(res.items(), key=lambda item: item[1], reverse=True)[1:20]:
#     print(f'{md.get_title(idm)}: {sim}')
