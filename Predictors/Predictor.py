from abc import abstractmethod

from Models.MovieTags import MovieTags
from Models.UserItemData import UserItemData


# Predictor interface
class Predictor:
    @abstractmethod
    def fit(self, uim: UserItemData):
        # Load and examine the model
        pass

    @abstractmethod
    def fit(self, mt: MovieTags):
        # Use in content based predictor
        pass

    @abstractmethod
    def predict(self, uid: int):
        # Calculate the recommended values for a given user
        pass

    @abstractmethod
    def predict(self, mid: int):
        # Use in content based predictor
        pass
