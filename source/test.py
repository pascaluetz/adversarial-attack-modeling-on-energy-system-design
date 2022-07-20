import json


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
        print(self.target_capex - 1e-4)
        print(1e-4)


test = Testclass("../configs/masterthesis.json")
test.script()
