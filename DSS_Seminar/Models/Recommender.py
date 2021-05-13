from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


# calculates average MAE, RMSE, recall, accuracy, F1.
# For recall, accuracy, and F1, you’ll need to choose a few recommended products for each user.
# I decided to take the ones that the user rated better than their average.
# Note that you do not recommend already viewed products and that the parameter n
# indicates the number of recommended products.

class Recommender:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def fit(self, uim: UserItemData):
        self.predictor.fit(uim)
        self.rated_movies = uim.data[['userID', 'movieID']]

    def recommend(self, userID, n=10, rec_seen=True):
        pred_dict = self.predictor.predict(userID)

        if not rec_seen:
            user_rated_movies = self.rated_movies[self.rated_movies['userID'] == userID]['movieID'].values
            pred_dict = {x: pred_dict[x] for x in pred_dict.keys() if x not in user_rated_movies}

        return sorted(pred_dict.items(), key=lambda item: item[1], reverse=True)[:n]
