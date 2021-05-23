import itertools
import math

import numpy as np

from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


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

    def evaluate(self, test_data, n=10):
        # intersect users from both data sets
        users1 = list(set(self.rated_movies['userID'].values))
        users2 = list(set(test_data.data['userID'].values))
        users = list(set(users1) & set(users2))

        predictions_init, predictions_test = self.get_predictions(users, test_data, n)
        mae = rmse = count = 0
        for values1, values2 in zip(predictions_init.values(), predictions_test.values()):
            if len(values1) == 0:
                continue
            # Here pred1 and pred2 contain matching movies with different rating predictions
            for pred1, pred2 in zip(values1, values2):
                mae = mae + abs(pred2[1] - pred1[1])
                rmse = rmse + (pred2[1] - pred1[1]) ** 2
                count = count + 1
        mae = mae / count
        rmse = math.sqrt(rmse / count)
        # Gather a list of ratings only (ignore users and movies)
        # Ratings are respective to the same movie in both lists on any index
        # Convert to int in order
        pred_init_ratings = [x[1] for x in list(itertools.chain.from_iterable(list(predictions_init.values())))]
        pred_test_ratings = [x[1] for x in list(itertools.chain.from_iterable(list(predictions_test.values())))]

        (precision, recall, f) = self.calculate_ratios(pred_init_ratings, pred_test_ratings)
        return rmse, mae, precision, recall, f

    def get_predictions(self, users, test_data, n):
        # Construct dictionaries of users and their respective recommendations
        # For both initial and test datasets
        predictions_init = {user: self.recommend(user, n, rec_seen=False) for user in users}
        self.fit(test_data)
        predictions_test = {user: self.recommend(user, n, rec_seen=False) for user in users}

        # Consider only movies which were rated by the same user in both datasets
        for user, pred1, pred2 in zip(users, predictions_init.values(), predictions_test.values()):
            pred1_movies, pred2_movies = [x[0] for x in pred1], [x[0] for x in pred2]
            intersect = set(pred1_movies) & set(pred2_movies)
            predictions_init[user] = list(filter(lambda item: item[0] in intersect, predictions_init[user]))
            predictions_test[user] = list(filter(lambda item: item[0] in intersect, predictions_test[user]))

        return predictions_init, predictions_test

    def calculate_ratios(self, init_ratings, test_ratings):
    	# Calculates precision, recall and F1.
        # The threshold is the mean of all the ratings
        threshold = np.mean(init_ratings)
        tp = fp = tn = fn = 0
        for true_r, test_r in zip(init_ratings, test_ratings):
            if true_r >= threshold:
                if test_r >= threshold:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if test_r >= threshold:
                    fp = fp + 1
                else:
                    tn = tn + 1
        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1
