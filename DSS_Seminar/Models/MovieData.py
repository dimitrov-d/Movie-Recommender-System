import pandas as pd


class MovieData:
    def __init__(self, file_path):
        data = pd.read_csv(file_path, sep="\t")
        self.data = data

    def get_title(self, movieID):
        movieID = int(movieID)
        try:
            return self.data[self.data['id'] == movieID]['title'].values[0]
        except:
            return None
