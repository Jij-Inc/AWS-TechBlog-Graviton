import time
from jijzept import JijSASampler, JijSQASampler
import jijzept as jz
import jijmodeling as jm
import logging
import sys
import numpy as np
import pandas as pd
from itertools import product
from pydantic import BaseModel
from pprint import pprint

# create JijModeling object
problem = jm.Problem("test")

# define variables
d = jm.Placeholder("d", dim=2)
N = d.shape[0].set_latex("N")
i = jm.Element("i", (0, N))
j = jm.Element("j", (0, N))
t = jm.Element("t", (0, N))
x = jm.Binary("x", shape=(N, N))

# set problem
problem = jm.Problem("TSP")
problem += jm.Sum([i, j], d[i, j] * jm.Sum(t, x[i, t] * x[j, (t + 1) % N]))
problem += jm.Constraint("one-city", x[:, t] == 1, forall=t)
problem += jm.Constraint("one-time", x[i, :] == 1, forall=i)

# generate 2d positions
city_range = 10
num_reads = 100
sa_one_city_multiplier = {20: 5, 40: 5}
sa_one_time_multiplier = {20: 5, 40: 5}
sqa_one_city_multiplier = {20: 5, 40: 5}
sqa_one_time_multiplier = {20: 5, 40: 5}

N_cities_list = [40]
trotter_sizes = [10, 20, 30, 40, 60]
num_sweepss = [1000, 2000, 4000, 6000, 10000, 20000]


class Data(BaseModel):
    N_cities: list[int]
    benchmark_device: list[str]
    num_sweeps: list[int]
    trotter_size: list[int]
    sample_no: list[int]
    objective: list[float]
    sampling_time: list[float]


data_obj = Data(
    N_cities=[],
    benchmark_device=[],
    num_sweeps=[],
    trotter_size=[],
    sample_no=[],
    objective=[],
    sampling_time=[],
)

for N_cities in N_cities_list:
    pprint(f"N_cities: {N_cities}")
    x = np.random.uniform(0, city_range, (N_cities))
    y = np.random.uniform(0, city_range, (N_cities))

    # generate points
    distance = np.zeros((N_cities, N_cities))
    for i in range(N_cities):
        for j in range(N_cities):
            distance[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

    instance_data = {"d": distance}
    benchmark_device = "SA"
    pprint(f"benchmark_device: {benchmark_device}")

    for num_sweeps in num_sweepss:
        pprint(f"num_sweeps: {num_sweeps}")
        # calculate SA algorithm
        # You need to specify JijZept credit toml file
        sampler = JijSASampler(config="config_prod.toml")
        response_orig: jm.SampleSet = sampler.sample_model(
            problem,
            instance_data,
            multipliers={
                "one-city": sa_one_city_multiplier[N_cities],
                "one-time": sa_one_city_multiplier[N_cities],
            },
            num_reads=num_reads,
            num_sweeps=num_sweeps,
        )
        response = response_orig.feasible().lowest()
        assert len(response.evaluation.objective) != 0
        calc_time = response_orig.get_backend_calculation_time()
        sa_objective = list(response.evaluation.objective)
        sa_calc_time = [
            calc_time["parameter_search"]["iteration"]["0"]["sample[0]"] / num_reads
        ] * num_reads

        # write data
        for trotter_size in trotter_sizes:
            data_obj.N_cities.append(N_cities)
            data_obj.benchmark_device.append(benchmark_device)
            data_obj.num_sweeps.append(num_sweeps)
            data_obj.trotter_size.append(trotter_size)
            data_obj.sample_no.append(0)
            data_obj.objective.append(sa_objective[0])
            data_obj.sampling_time.append(sa_calc_time[0])

    for benchmark_device, num_sweeps, trotter_size in product(
        ["G2", "G3"], num_sweepss, trotter_sizes
    ):
        pprint(
            f"benchmark_device: {benchmark_device}, num_sweeps: {num_sweeps} , trotter_size: {trotter_size}"
        )
        if benchmark_device == "G2":
            queue_name = "testsolver2"
        elif benchmark_device == "G3":
            queue_name = "testsolver3"

        # SQA algorithm (using AWS Graviton 2 or 3)
        # You need to specify JijZept credit toml file
        sampler = JijSQASampler(config="config_prod.toml")
        response_orig: jm.SampleSet = sampler.sample_model(
            problem,
            instance_data,
            queue_name=queue_name,
            multipliers={
                "one-city": sa_one_city_multiplier[N_cities],
                "one-time": sa_one_city_multiplier[N_cities],
            },
            num_reads=num_reads,
            trotter=trotter_size,
            beta=trotter_size,
        )
        response = response_orig.feasible().lowest()
        assert len(response.evaluation.objective) != 0
        calc_time = response_orig.get_backend_calculation_time()
        sqa_objective = list(response.evaluation.objective)
        sqa_calc_time = [
            calc_time["parameter_search"]["iteration"]["0"]["sample[0]"] / num_reads
        ] * num_reads

        data_obj.N_cities.append(N_cities)
        data_obj.benchmark_device.append(benchmark_device)
        data_obj.num_sweeps.append(num_sweeps)
        data_obj.trotter_size.append(trotter_size)
        data_obj.sample_no.append(0)
        data_obj.objective.append(sqa_objective[0])
        data_obj.sampling_time.append(sqa_calc_time[0])

df = pd.DataFrame(data_obj.dict())
print(df)
df.to_csv("result.csv")
