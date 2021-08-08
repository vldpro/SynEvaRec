import abc
import datetime
import typing as t
import itertools
import functools
import logging
import multiprocessing

import numpy as np
import pandas as pd


class TestSetError(t.NamedTuple):
    rmse: float
    mae: float


class ResponseFunctionABC:
    @abc.abstractmethod
    def __call__(self, a1: float, a2: float):
        pass


class TrainTestExecutorABC(abc.ABC):
    @abc.abstractproperty
    def model_name(self) -> str:
        pass

    @abc.abstractproperty
    def config(self) -> dict:
        pass

    @abc.abstractmethod
    def __call__(self, rating_matrix: pd.DataFrame, test_size: float, sample_size: float) -> TestSetError:
        pass


class EvaluationResult(t.NamedTuple):
    a1: float
    a2: float
    model_name: str
    sample_size: float
    rmse: float
    mae: float
    test_size: float 
    calculation_time: datetime.datetime


class ResponseFunctionConfig(t.NamedTuple):
    factory: t.Type[ResponseFunctionABC] 
    args: list


class TrainTestExecutorConfig(t.NamedTuple):
    factory: t.Type[TrainTestExecutorABC]
    model_name: str
    args: dict 


class _SubProcessArgs(t.NamedTuple):
    a1: float
    test_size: float
    a_sample_rate: int
    sample_sizes: t.List[float]
    resp_fn_config: ResponseFunctionConfig
    train_test_executor_config: TrainTestExecutorConfig 


def _iterate_over_a2_job(args: _SubProcessArgs) -> t.List[EvaluationResult]:
    print(f"Subprocess started.")
    train_fn = args.train_test_executor_config.factory(**args.train_test_executor_config.args)
    response_function = args.resp_fn_config.factory(*args.resp_fn_config.args)
    sample_rate = args.a_sample_rate

    results: t.List[EvaluationResult] = []
    a1_normalized = args.a1 / sample_rate 
    for a2 in range(0, sample_rate - args.a1):
        a2_normalized = a2 / sample_rate 
        ground_truth_matrix = response_function(a1_normalized, a2_normalized)
        for sample_size in args.sample_sizes:
            for _ in range(1):
                start = datetime.datetime.utcnow()
                test_error = train_fn(rating_matrix=ground_truth_matrix, test_size=args.test_size, sample_size=sample_size)
                duration = datetime.datetime.utcnow() - start
                result = EvaluationResult(
                    a1=a1_normalized, 
                    a2=a2_normalized,
                    rmse=test_error.rmse,
                    mae=test_error.mae,
                    sample_size=sample_size,
                    model_name=train_fn.model_name,
                    test_size=args.test_size,
                    calculation_time=duration
                )
                logging.info(f"{train_fn.model_name} - {result}")
                results.append(result)
    return results


class Evaluator:
    def __init__(self, resp_fn_config: ResponseFunctionConfig, n_proc: int = 4):
        assert n_proc > 0
        self._n_proc = n_proc
        self._resp_fn_config = resp_fn_config

    def evaluate(
        self,
        executor_configs: t.Union[t.List[TrainTestExecutorConfig], TrainTestExecutorConfig],
        a_sample_rate: int,
        test_size: float,
        sample_sizes: t.List[float]
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        assert a_sample_rate < 100
        assert 0.0 < test_size < 1.0
        if not isinstance(executor_configs, list):
            executor_configs = [executor_configs]
        results = [self._evaluate(c, a_sample_rate, test_size, sample_sizes) for c in executor_configs]
        results = list(itertools.chain.from_iterable(results))
        return pd.DataFrame(results)

    def _evaluate(self, executor_config: TrainTestExecutorConfig, a_sample_rate: int, test_size: float, sample_sizes: t.List[float]) -> t.Dict[str, t.Dict[str, t.Any]]:
        args = [
            _SubProcessArgs(
                a1=a1,
                test_size=test_size,
                resp_fn_config=self._resp_fn_config,
                train_test_executor_config=executor_config,
                a_sample_rate=a_sample_rate,
                sample_sizes=sample_sizes
            ) for a1 in range(0, a_sample_rate)
        ]
        start_time = datetime.datetime.utcnow()
        with multiprocessing.Pool(processes=self._n_proc) as p:
            results = p.map(_iterate_over_a2_job, args)
        duration = datetime.datetime.utcnow() - start_time
        return list(itertools.chain.from_iterable(results))