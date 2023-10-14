import logging
import sys
import time
from itertools import product
from pprint import pprint

import jijmodeling as jm
import jijzept as jz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jijzept import JijSASampler, JijSQASampler
from pydantic import BaseModel

if __name__ == "__main__":
    df = pd.read_csv("result.csv")
    df = df.query("trotter_size == 60")

    df_sa = df.query("benchmark_device == 'SA'")
    df_g2 = df.query("benchmark_device == 'G2'")
    df_g3 = df.query("benchmark_device == 'G3'")

    for df_obj, c, label in zip(
        [df_sa, df_g2, df_g3],
        ["b", "r", "orange"],
        ["SA", "SQA (AWS Graviton 2)", "SQA (AWS Graviton 3)"],
    ):
        plt.plot(
            df_obj["sampling_time"],
            df_obj["objective_mean"],
            "--bo",
            label=label,
            color=c,
        )
        plt.fill_between(
            df_obj["sampling_time"],
            df_obj["objective_mean"] + df_obj["objective_std"],
            df_obj["objective_mean"] - df_obj["objective_std"],
            alpha=0.4,
            color=c,
        )
        plt.fill_between(
            df_obj["sampling_time"],
            df_obj["objective_max"],
            df_obj["objective_min"],
            alpha=0.1,
            color=c,
        )

    plt.legend()
    plt.xlabel("sampling time [s]")
    plt.ylabel("minimum cost function (traveling distance)")

    # plt.show()
    plt.savefig("result_benchmark.png")
