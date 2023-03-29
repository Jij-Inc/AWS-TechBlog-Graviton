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

# create JijModeling object here
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
num_sweepss_sa = [1000, 2000, 4000, 6000, 10000]
num_sweepss = [100, 200, 400, 600, 1000]


class Data(BaseModel):
    N_cities: list[int]
    benchmark_device: list[str]
    num_sweeps: list[int]
    trotter_size: list[int]
    sample_no: list[int]
    objective_mean: list[float]
    objective_std: list[float]
    objective_min: list[float]
    objective_max: list[float]
    sampling_time: list[float]


data_obj = Data(
    N_cities=[],
    benchmark_device=[],
    num_sweeps=[],
    trotter_size=[],
    sample_no=[],
    objective_mean=[],
    objective_std=[],
    objective_min=[],
    objective_max=[],
    sampling_time=[],
)

#np.random.seed(0x123456)

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

    for num_sweeps in num_sweepss_sa:
        pprint(f"num_sweeps: {num_sweeps}")
        # calculate SA
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
        response = response_orig.feasible()
        assert len(response.evaluation.objective) != 0
        pprint(f"feasibles={len(response.evaluation.objective)}")
        calc_time = response_orig.get_backend_calculation_time()
        sa_objective = list(response.evaluation.objective)
        print(np.mean(sa_objective))
        print(np.std(sa_objective))
        print(np.min(sa_objective))
        print(np.max(sa_objective))
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
            data_obj.objective_mean.append(np.mean(sa_objective))
            data_obj.objective_std.append(np.std(sa_objective))
            data_obj.objective_min.append(np.min(sa_objective))
            data_obj.objective_max.append(np.max(sa_objective))
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

        sampler = JijSQASampler(config="config.toml")
        response_orig: jm.SampleSet = sampler.sample_model(
            problem,
            instance_data,
            queue_name=queue_name,
            #multipliers={
            #    "one-city": 0,
            #    "one-time": 0,
            #},
            multipliers={
                "one-city": sa_one_city_multiplier[N_cities],
                "one-time": sa_one_city_multiplier[N_cities],
            },
            num_reads=num_reads,
            trotter=trotter_size,
            num_sweeps=num_sweeps,
            beta=trotter_size,
        )
        response = response_orig.feasible()
        #pprint(response)
        assert len(response.evaluation.objective) != 0
        pprint(f"feasibles={len(response.evaluation.objective)}")
        #pprint(response.evaluation.objective)
        calc_time = response_orig.get_backend_calculation_time()
        #pprint(calc_time)
        # {'parameter_search': {'compile': 0.080865, 'iteration': {'0': {'get_qubo[0]': 0.126019, 'sample[0]': 0.775543, 'decode[0]': 0.228371, 'update[0]': 0.026416}}, 'reformat': 7.9e-05}}
        sqa_objective = list(response.evaluation.objective)
        print(np.mean(sqa_objective))
        print(np.std(sqa_objective))
        print(np.min(sqa_objective))
        print(np.max(sqa_objective))
        sqa_calc_time = [
            calc_time["parameter_search"]["iteration"]["0"]["sample[0]"] / num_reads
        ] * num_reads

        data_obj.N_cities.append(N_cities)
        data_obj.benchmark_device.append(benchmark_device)
        data_obj.num_sweeps.append(num_sweeps)
        data_obj.trotter_size.append(trotter_size)
        data_obj.sample_no.append(0)
        data_obj.objective_mean.append(np.mean(sqa_objective))
        data_obj.objective_std.append(np.std(sqa_objective))
        data_obj.objective_min.append(np.min(sqa_objective))
        data_obj.objective_max.append(np.max(sqa_objective))
        data_obj.sampling_time.append(sqa_calc_time[0])

df = pd.DataFrame(data_obj.dict())
print(df)
df.to_csv("result.csv")
