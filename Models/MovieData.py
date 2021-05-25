import pandas as pd


class MovieData:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, sep="\t")

    def get_title(self, movieID):
        movieID = int(movieID)
        try:
            return self.data[self.data['id'] == movieID]['title'].values[0]
        except:
            return None
