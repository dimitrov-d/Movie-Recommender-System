import pandas as pd


class UserItemData:
    def __init__(self, file_path, start_date=None, end_date=None, min_ratings=0):
        data = pd.read_csv(file_path, sep="\t",
                           dtype={"userID": int, "movieID": int, "rating": float, "date_day": int, "date_month": int,
                                  "date_year": int})
        data = data[data.groupby('movieID').userID.transform(len) > min_ratings]

        # Join date columns into a single dot-separated date value
        if start_date is not None or end_date is not None:
            date_cols = ['date_day', 'date_month', 'date_year']
            data['date'] = pd.to_datetime(
                data[date_cols].apply(lambda row: '.'.join(row.values.astype(str)), axis=1))

        if start_date is not None:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            data = data[data['date'] <= pd.to_datetime(end_date)]

        self.data = data

    def nratings(self):
        return self.data.shape[0]
