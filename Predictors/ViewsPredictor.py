from Models.UserItemData import UserItemData
from Predictors.Predictor import Predictor


class ViewsPredictor(Predictor):
    def fit(self, uim: UserItemData):
        self.data = uim.data.groupby('movieID').count()

    def predict(self, userID):
        return {movieID: views for movieID, views in zip(list(self.data.index), self.data['rating'].values)}