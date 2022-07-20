import csv
import json

import numpy as np


class Testclass:
    def __init__(self, config_file):
        config = json.load(open(config_file, "r"))

        self.weight_pv_availability = config["weights_input"]["pv_availability"]
        self.weight_demand = config["weights_input"]["demand"]
        self.attack_type = config["attack_settings"]["attack_type"]
        self.objective = config["attack_settings"]["objective"]
        self.target_capex = config["attack_settings"]["target_capex"]
        self.big_m = config["algorithm_settings"]["big_m"]
        self.warmstart = config["algorithm_settings"]["warmstart"]
        self.mip_gap = config["algorithm_settings"]["mip_gap"]
        self.no_negative_demand = config["additional_constraints"]["no_negative_demand"]
        self.no_negative_pv_availability = config["additional_constraints"]["no_negative_pv_availability"]
        self.pv_availability_smaller_one = config["additional_constraints"]["pv_availability_smaller_one"]
        self.sum_demand_zero = config["additional_constraints"]["sum_demand_zero"]
        self.sum_pv_availability_zero = config["additional_constraints"]["sum_pv_availability_zero"]

    def script(self):
        print("Test")


# test = Testclass("../configs/game_00.json")
# test.script()


def read_csv(csv_name):
    array = []

    with open(csv_name, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        for row in reader:
            array.append(float(row[0]))

    array = np.array(array)
    return array


bdew_demand = read_csv("../time_series/bdew_demand.csv")
bdew_demand_24h = read_csv("../time_series/bdew_demand_24h.csv")
original_demand = read_csv("../time_series/original_demand.csv")
original_demand_24h = read_csv("../time_series/original_demand_24h.csv")
original_pv_availability = read_csv("../time_series/original_pv_availability.csv")
original_pv_availability_24h = read_csv("../time_series/original_pv_availability_24h.csv")

np.savetxt("../time_series/bdew_demand.csv", bdew_demand, delimiter=",")
np.savetxt("../time_series/bdew_demand_24h.csv", bdew_demand_24h, delimiter=",")
np.savetxt("../time_series/original_demand.csv", original_demand, delimiter=",")
np.savetxt("../time_series/original_demand_24h.csv", original_demand_24h, delimiter=",")
np.savetxt("../time_series/original_pv_availability.csv", original_pv_availability, delimiter=",")
np.savetxt("../time_series/original_pv_availability_24h.csv", original_pv_availability_24h, delimiter=",")
