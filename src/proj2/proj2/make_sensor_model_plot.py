#!/usr/bin/env python3
from __future__ import division

import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pdb

from proj2.sensor_model import SingleBeamSensorModel


def plot_sensor_model_for_obs(
    sensor_params=None, resolution=0.025, max_r=11.0, sim_r=7.0
):
  """Visualize the sensor model likelihood for a simulated observation.

  Args:
      sensor_params: Sensor model parameters
      resolution (float): Map pixel resolution (in meters/pixel)
      max_r (float): The maximum range (in meters)
      sim_r (float): The simulated range observation (in meters)
  """
  max_r_pixels = int(max_r / resolution)
  obs_r_pixels = int(sim_r / resolution)

  sensor_model = SingleBeamSensorModel(**sensor_params)
  prob_table = sensor_model.precompute_sensor_model(max_r_pixels)
  likelihood = prob_table[:, obs_r_pixels]
  distance = resolution * np.arange(likelihood.size, dtype=float)

  # pdb.set_trace()

  print("Sum of probabilities: {:.3f}".format(likelihood.sum()))

  plt.figure(figsize=(8, 4), dpi=100)
  plt.title("Sensor model likelihood")
  plt.xlabel("Observed distance measurement $z_t^k$ (m)")
  plt.ylabel("$P(z_t^k | {z_t^k}^*)$")

  xticks = list(range(int(max_r) + 1))
  labels = [str(x) for x in xticks]

  if sim_r in xticks:
    labels[xticks.index(sim_r)] = "${z_t^k}^*$"
  if max_r in xticks:
    labels[xticks.index(max_r)] = "$z_\\mathrm{max}$"

  plt.xticks(xticks, labels)
  plt.plot(distance, likelihood)
  plt.show()


if __name__ == "__main__":
  path_to_current = os.path.dirname(os.path.realpath(__file__))
  with open(os.path.join(path_to_current, "../config/parameters.yaml"), "r") as f:
    params = yaml.safe_load(f)
    sensor_params = params.get("sensor_params")
    print("Loaded parameters:", sensor_params)

  plot_sensor_model_for_obs(sensor_params)
