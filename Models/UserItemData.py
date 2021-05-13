import pandas as pd


class UserItemData:
    def __init__(self, file_path, from_date=None, to_date=None, min_ratings=0):
        data = pd.read_csv(file_path, sep="\t",
                           dtype={"userID": int, "movieID": int, "rating": float, "date_day": int,"date_month": int, "date_year":int})
        data = data[data.groupby('movieID').userID.transform(len) > min_ratings]

        # Join date columns into a single dot-separated date value
        if from_date != None or to_date != None:
            date_cols = ['date_day', 'date_month', 'date_year']
            data['date'] = pd.to_datetime(
                data[date_cols].apply(lambda row: '.'.join(row.values.astype(str)), axis=1))

        if from_date != None:
            data = data[data['date'] >= pd.to_datetime(from_date)]
        if to_date != None:
            data = data[data['date'] <= pd.to_datetime(to_date)]

        self.data = data

    def nratings(self):
        return self.data.shape[0]

# uim = UserItemData('../movielens/user_ratedmovies.dat')
# print(uim.nratings())
#
# uim = UserItemData('../movielens/user_ratedmovies.dat', '12.1.2007', '16.2.2008', min_ratings=100)
# print(uim.nratings())
