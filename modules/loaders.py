import abc

import pandas as pd


class DataLoaderABC(abc.ABC):
    @abc.abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class RestaurantDatasetLoader(DataLoaderABC):
    def __init__(self, paths: dict) -> None:
        self._paths = paths

    def load(self) -> pd.DataFrame:
        users_df = self._load_and_clean_users_df()
        rests_df = self._load_and_prepare_rest_cuisine_df()
        ratings_df = pd.read_csv(self._paths["ratings.csv"])
        merged_df = pd.merge(ratings_df, users_df, on="userID")
        merged_df = pd.merge(merged_df, rests_df, on="placeID")
        return merged_df

    def _load_and_prepare_rest_cuisine_df(self):
        df = pd.read_csv(self._paths["chefmozcuisine.csv"])
        df = df.drop_duplicates()
        df = df.join(pd.get_dummies(df["Rcuisine"]))
        df = df.drop("Rcuisine", axis=1)
        df = df.groupby("placeID").sum()
        return df

    def _load_and_clean_users_df(self):
        user_profile_df = pd.read_csv(self._paths["userprofile.csv"])
        user_cuisine_df = pd.read_csv(self._paths["usercuisine.csv"])
        user_profile_df = self._prepare_user_profile_df(user_profile_df)
        user_cuisine_df = self._prepare_user_cuisine_df(user_cuisine_df)

        users_df = pd.merge(user_profile_df, user_cuisine_df, on="userID")
        return users_df

    @staticmethod
    def _prepare_user_profile_df(df):
        df = df.drop(["latitude", "longitude"], axis=1)
        df = df.replace("?", pd.NA)
        df = df.fillna(method="bfill")
        return df

    @staticmethod
    def _prepare_user_cuisine_df(df):
        df.drop_duplicates()
        df = df.join(pd.get_dummies(df["Rcuisine"]))
        df = df.drop("Rcuisine", axis=1)
        df = df.groupby("userID").sum()
        return df



