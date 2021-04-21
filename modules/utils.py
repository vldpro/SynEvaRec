import typing as t

import pandas as pd


def reformat_evaluation_results_to_single_dataframe(eval_results_list: t.List[t.Dict]) -> pd.DataFrame:
    df = pd.DataFrame() 
    for eval_result_meta in eval_results_list:
        eval_results = [
            (r.a1, r.a2, r.test_error.rmse, r.test_error.mae, eval_result_meta["model_name"])
            for r in eval_result_meta["results"] 
        ]
        eval_results_df = pd.DataFrame(eval_results, columns=["a1", "a2", "rmse", "mae", "model"])
        df = pd.concat([df, eval_results_df])
    return df 


def group_points_by_minimum_error(points_df) -> pd.DataFrame:
    groupped = points_df[["a1", "a2", "rmse"]].groupby(
        by=["a1", "a2"], as_index=False
    ).min().join(
        points_df.set_index(["a1", "a2", "rmse"]), 
        on=["a1", "a2", "rmse"]
    )
    return groupped
