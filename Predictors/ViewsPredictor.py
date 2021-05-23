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
