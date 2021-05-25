import pandas as pd

from Models.MovieData import MovieData


class MovieTags:
    def __init__(self, md: MovieData, movietags_file_path, tags_file_path):
        movie_tags = pd.read_csv(movietags_file_path, sep="\t")[['movieID', 'tagID']]
        tags = pd.read_csv(tags_file_path, sep="\t")
        # Merge movies.dat, movie_tags.dat and tags.dat to finally have movie title and tag name
        movie_tags = pd.merge(md.data, movie_tags, left_on="id", right_on="movieID", how="inner")[
            ['title', 'movieID', 'tagID']]
        movie_tag_names = pd.merge(movie_tags, tags, left_on="tagID", right_on="id", how="inner")[
            ['title', 'movieID', 'tagID', 'value']]
        # Join movie tag rows into one row containing string of all tags
        self.data = movie_tag_names.groupby('movieID')['value'].apply(' '.join).fillna('').reset_index(name='tags')
        # too slow: self.movie_tag_names['title'] = self.movie_tag_names['movieID'].apply(md.get_title)
