from abc import abstractmethod

from Models.UserItemData import UserItemData


# Predictor interface
class Predictor:
    @abstractmethod
    def fit(self, uim: UserItemData):
        """Examine and load the model"""
        pass

    @abstractmethod
    def predict(self, uid: int):
        """Calculate the recommended values for a given user"""
        pass
