import multiprocessing as mp
import os
import time
from datetime import datetime
from functools import partial
from typing import Callable, Union

from .logging import get_logger

logger = get_logger("MP_Tool")

LOCK = mp.Lock()
MAX_LOCK = mp.Lock()
CPU_NUMBER = mp.cpu_count()


class MP_Tool:
    all_requests: int = 0

    def __init__(self, processes: int = CPU_NUMBER, max_processes: int = -1) -> None:
        if processes > CPU_NUMBER:
            assert ValueError(
                f"the number of processing must < cpu cores: {CPU_NUMBER}~"
            )
        self.processes = processes
        self.Pool = mp.Pool(processes=processes)
        if max_processes > 0:
            self.max_processes = max_processes
        else:
            self.max_processes = self.processes * 2

    @classmethod
    def activate_children_number(cls):
        return len(mp.active_children())

    @classmethod
    def request_add(cls, verbose=False, *args):
        LOCK.acquire()
        cls.all_requests += 1
        if verbose:
            logger.info(
                f"Activated processing: {cls.all_requests}/{cls.activate_children_number()}"
            )
        LOCK.release()

    @classmethod
    def request_sub(cls, verbose=False, *args):
        LOCK.acquire()
        cls.all_requests -= 1
        if verbose:
            logger.info(
                f"Activated processing: {cls.all_requests}/{cls.activate_children_number()}"
            )
        LOCK.release()

    def check_and_wait(self):
        while True:
            if self.all_requests < self.max_processes:
                return
            logger.info("Waiting for the pool...")
            time.sleep(5)

    def create(self, fun: Callable, verbose: bool = False, *args, **kwargs):
        """help user to create a processing for a task (function)

        Parameters
        ----------
        fun : Callable
            _description_
        verbose: bool
            if true, print information
        args:
            pass to fun, such as: fun(args[0], args[1], ..., args[n])
        kwargs:
            pass to fun, such as: fun(key_1=kwargs[key_1], ..., key_n=kwargs[key_n])
        """
        self.check_and_wait()
        callback_fn = kwargs.pop("callback_fn", None)
        if not callback_fn:
            callback_fn = self.request_sub
        else:
            callback_fn = partial(self.callback_fn, fun=callback_fn)
        result = self.Pool.apply_async(fun, args, kwargs, callback=callback_fn)
        self.request_add(verbose=verbose)
        return result

    def close(self):
        self.Pool.close()

    def join(self):
        self.Pool.join()

    def callback_fn(self, data, fun: Callable, verbose=False):
        self.request_sub(verbose=verbose)
        return fun(data)


# def f(x):
#     time.sleep(2)
#     print(x)
#     return x * x


# def main_(mp_tool):
#     results = []
#     for i in range(20):
#         results.append(mp_tool.create(f, False, i, callback_fn=f_callback))
#     for result in results:
#         result.wait()


# def f_callback(x):
#     print(f"end {x}")
#     return x
