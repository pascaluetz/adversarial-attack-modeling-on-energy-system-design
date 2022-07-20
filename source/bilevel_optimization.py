"""
Runs PV House Model solved with Karush-Kuhn-Tucker Conditions and finds a minimal delta.
- In function find_minimal_delta it is possible to set which variables should be manipulated and also the allowed
absolute mip gap to the optimal solution
- In function run it is possible to adjust the output data of the model
"""

import json
import time

import numpy as np
import pyomo.environ as pyo
import pypsa.opt as pypsa
from pyomo.opt import SolverFactory
from scipy.sparse import csr_matrix

from tools import convert_mps_to_array as converter


class Algorithm:
    """
    Class used to initialize, calculate, log and plot adversarial attack
    """

    def __init__(self, config_file):
        """
        Initialize bi-level optimization with config and model parameters
        """
        # initialize configuration
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

        # initialize model matrices of inner optimization problem
        model_parameters = open("source/model.mps", "r").readlines()

        all_variables, bounded_variables, c_row_names, c, A_row_names, A, H_row_names, H, b, d = converter.script(
            model_parameters, bounded_variables_as_equations=True
        )

        self.all_variables = all_variables
        self.c = c
        self.A_row_names = A_row_names
        self.A = A
        self.H_row_names = H_row_names
        self.H = H
        self.b = b
        self.d = d

    def calculate(self):
        start_time = time.time()

        model = self.calculate_iteration()
        print("calculation done")

        print(model.pprint())

        # iterative and stuff like this
        end_time = time.time()
        execution_time = end_time - start_time
        print(execution_time)

    def calculate_iteration(self):
        """
        finds minimal delta with PyPSA + indptr/indices
        """

        # get model data including warm binaries and Capacity_PV,0
        warm_binaries, Cap_PV_0 = self.get_model_data()

        # initialize instance of outer optimization model
        model = pyo.ConcreteModel()

        # define decision variables
        model.x = pyo.Var(range(len(self.all_variables)), within=pyo.NonNegativeReals)
        model.lambdas = pyo.Var(range(self.A.shape[0]), within=pyo.NonNegativeReals)
        model.ny = pyo.Var(range(self.H.shape[0]), within=pyo.Reals)
        model.binary = pyo.Var(range(self.A.shape[0]), within=pyo.Binary)

        # initialize attack variables
        model.deltab = pyo.Var(range(self.b.shape[0]), within=pyo.Reals)
        model.deltad = pyo.Var(range(self.d.shape[0]), within=pyo.Reals)

        # MIMIZE MAX
        # model.max_deltad = pyo.Var(within=pyo.NonNegativeReals)

        # ABSOLUTE VALUE OBJECTIVE
        # model.deltab_pos = pyo.Var(range(b.shape[0]), within=pyo.NonNegativeReals)
        # model.deltab_neg = pyo.Var(range(b.shape[0]), within=pyo.NonPositiveReals)

        # model.deltad_pos = pyo.Var(range(d.shape[0]), within=pyo.NonNegativeReals)
        # model.deltad_neg = pyo.Var(range(d.shape[0]), within=pyo.NonPositiveReals)

        # save the amount of changed variables
        changed_deltas = 0

        # define vectors to add manipulation (manipulate_vector * attack_variable)
        # manipulation of b in primal feasibility and complementary slackness
        manipulate_b = np.zeros(self.b.shape[0])
        start_range = self.A_row_names["c_u_limEQpv(1)_"]
        end_range = self.A_row_names["c_u_limEQpv(8760)_"] + 1
        self.range_limeqpv_b = range(start_range, end_range)

        if self.weight_pv_availability != 0:
            changed_deltas += end_range - start_range
            col = self.all_variables["CapacityPV"]
            if self.attack_type == "absolute":
                for row in self.range_limeqpv_b:
                    manipulate_b[row] = Cap_PV_0
            elif self.attack_type == "relative":
                for row in self.range_limeqpv_b:
                    manipulate_b[row] = Cap_PV_0 * -self.A[row, col]

        manipulate_b = csr_matrix(manipulate_b).transpose()

        # manipulation of d in primal feasibility
        manipulate_d = np.zeros(self.d.shape[0])
        start_range = self.H_row_names["c_e_EnergyEQ(1)_"]
        end_range = self.H_row_names["c_e_EnergyEQ(8760)_"] + 1
        self.range_energyeq_d = range(start_range, end_range)
        if self.weight_demand != 0:
            changed_deltas += end_range - start_range

            if self.attack_type == "absolute":
                for index in self.range_energyeq_d:
                    manipulate_d[index] = 1
            elif self.attack_type == "relative":
                for index in self.range_energyeq_d:
                    manipulate_d[index] = self.d[index]

        manipulate_d = csr_matrix(manipulate_d).transpose()

        # define objective

        # ABSOLUTE VALUE OBJECTIVE
        # factor = 1 / changed_deltas
        # deltab_tuples = []
        # for k in self.range_limeqpv_b:
        #     deltab_tuples.append((factor, model.deltab_pos[k]))
        #     deltab_tuples.append((-factor, model.deltab_neg[k]))
        #
        # deltad_tuples = []
        # for k in self.range_energyeq_d:
        #     deltad_tuples.append((factor, model.deltad_pos[k]))
        #     deltad_tuples.append((-factor, model.deltad_neg[k]))
        #
        # objective = pypsa.LExpression(
        #     (1 - self.weight_pv_availability) * deltab_tuples + (1 - self.weight_demand) * deltad_tuples
        # )
        # pypsa.l_objective(model, objective, pypsa.minimize)

        # QUADRATIC OBJECTIVE
        # calculating scaling coefficients
        b_numpy = []
        col = self.all_variables["CapacityPV"]
        for index in self.range_limeqpv_b:
            b_numpy.append(-self.A[index, col])
        b_numpy = np.array(b_numpy)

        d_numpy = []
        for index in self.range_energyeq_d:
            d_numpy.append(self.d[index])
        d_numpy = np.array(d_numpy)

        # initialize objective
        model.size_delta = pyo.Objective(
            expr=(1 - self.weight_pv_availability)
            * sum(
                ((model.deltab[i] - b_numpy.min()) / (b_numpy.max() - b_numpy.min())) ** 2 for i in self.range_limeqpv_b
            )
            + (1 - self.weight_demand)
            * sum(
                ((model.deltad[i] - d_numpy.min()) / (d_numpy.max() - d_numpy.min())) ** 2
                for i in self.range_energyeq_d
            ),
            sense=pyo.minimize,
        )

        # # MINIMIZE MAX FUNCTION
        # objective = pypsa.LExpression([(1, model.max_deltad)])
        # pypsa.l_objective(model, objective, pypsa.minimize)

        # Constraints

        # # MINIMIZE MAX FUNCTION
        # constraints = {}
        # for index in self.range_energyeq_d:
        #     lhs = pypsa.LExpression([(manipulate_d[index, 0], model.deltad_pos[index]),
        #                              (-manipulate_d[index, 0], model.deltad_neg[index])])
        #     rhs = pypsa.LExpression([(1, model.max_deltad)])
        #     constraints[index] = pypsa.LConstraint(lhs, "<=", rhs)
        # pypsa.l_constraint(model, "MaxDeltaD", constraints, self.range_energyeq_d)

        # # ABSOLUTE VALUE OBJECTIVE
        # constraints = {}
        # for index in self.range_limeqpv_b:
        #     lhs = pypsa.LExpression([(1, model.deltab[index])])
        #     rhs = pypsa.LExpression([(1, model.deltab_pos[index]), (1, model.deltab_neg[index])])
        #     constraints[index] = pypsa.LConstraint(lhs, "==", rhs)
        # pypsa.l_constraint(model, "DeltaBEQ", constraints, self.range_limeqpv_b)
        #
        # constraints = {}
        # for index in self.range_energyeq_d:
        #     lhs = pypsa.LExpression([(1, model.deltad[index])])
        #     rhs = pypsa.LExpression([(1, model.deltad_pos[index]), (1, model.deltad_neg[index])])
        #     constraints[index] = pypsa.LConstraint(lhs, "==", rhs)
        # pypsa.l_constraint(model, "DeltaDEQ", constraints, self.range_energyeq_d)

        # Primal feasibility
        # Bounded constraints
        constraints = {}
        for row in range(self.A.shape[0]):
            list_of_tuples = [
                (self.A.data[k], model.x[self.A.indices[k]]) for k in range(self.A.indptr[row], self.A.indptr[row + 1])
            ]
            list_of_tuples.append((-manipulate_b[row, 0], model.deltab[row]))
            lhs = pypsa.LExpression(list_of_tuples, -self.b[row])
            constraints[row] = pypsa.LConstraint(lhs, "<=")

        pypsa.l_constraint(model, "AEQS", constraints, range(self.A.shape[0]))

        # Equal constraints
        constraints = {}
        for row in range(self.H.shape[0]):
            list_of_tuples = [
                (self.H.data[k], model.x[self.H.indices[k]]) for k in range(self.H.indptr[row], self.H.indptr[row + 1])
            ]
            list_of_tuples.append((-manipulate_d[row, 0], model.deltad[row]))
            lhs = pypsa.LExpression(list_of_tuples, -self.d[row])
            constraints[row] = pypsa.LConstraint(lhs, "==")

        pypsa.l_constraint(model, "HEQS", constraints, range(self.H.shape[0]))

        # Dual feasibility -> through lambdas within NonNegativeReals

        # Complementary slackness
        constraints = {}
        for row in range(self.A.shape[0]):
            lhs = pypsa.LExpression([(-self.big_m, model.binary[row]), (1, model.lambdas[row])])
            constraints[row] = pypsa.LConstraint(lhs, "<=")
        pypsa.l_constraint(model, "SlackEQ1", constraints, range(self.A.shape[0]))

        constraints = {}
        for row in range(self.A.shape[0]):
            list_of_tuples = [
                (self.A.data[k], model.x[self.A.indices[k]]) for k in range(self.A.indptr[row], self.A.indptr[row + 1])
            ]
            list_of_tuples.append((-manipulate_b[row, 0], model.deltab[row]))
            rhs = pypsa.LExpression(list_of_tuples, -self.b[row])
            lhs = pypsa.LExpression([(self.big_m, model.binary[row])], -self.big_m)
            constraints[row] = pypsa.LConstraint(lhs, "<=", rhs)
        pypsa.l_constraint(model, "SlackEQ2", constraints, range(self.A.shape[0]))

        # Stationarity / Derivation of the Lagrange function
        constraints = {}
        AT = self.A.transpose().tocsr()
        HT = self.H.transpose().tocsr()

        for col in range(len(self.all_variables)):
            a_tuples = [(AT.data[k], model.lambdas[AT.indices[k]]) for k in range(AT.indptr[col], AT.indptr[col + 1])]
            h_tuples = [(HT.data[k], model.ny[HT.indices[k]]) for k in range(HT.indptr[col], HT.indptr[col + 1])]
            list_of_tuples = a_tuples + h_tuples
            lhs = pypsa.LExpression(list_of_tuples, self.c[0, col])
            constraints[col] = pypsa.LConstraint(lhs, "==")
        pypsa.l_constraint(model, "LagrangianEQ", constraints, range(len(self.all_variables)))

        # add additional constraints
        if self.no_negative_demand:
            constraints = {}
            for index in self.range_energyeq_d:
                lhs = pypsa.LExpression([(manipulate_d[index, 0], model.deltad[index])], self.d[index])
                constraints[index] = pypsa.LConstraint(lhs, ">=")
            pypsa.l_constraint(model, "NoNegativeDemand", constraints, self.range_energyeq_d)

        if self.no_negative_pv_availability:
            constraints = {}
            col = self.all_variables["CapacityPV"]
            for index in self.range_limeqpv_b:
                lhs = pypsa.LExpression([(manipulate_b[index, 0] / Cap_PV_0, model.deltab[index])], -self.A[index, col])
                constraints[index] = pypsa.LConstraint(lhs, ">=")
            pypsa.l_constraint(model, "NoNegativAvail", constraints, self.range_limeqpv_b)

        if self.pv_availability_smaller_one:
            constraints = {}
            col = self.all_variables["CapacityPV"]
            for index in self.range_limeqpv_b:
                lhs = pypsa.LExpression(
                    [(manipulate_b[index, 0] / Cap_PV_0, model.deltab[index])], -self.A[index, col] - 1
                )
                constraints[index] = pypsa.LConstraint(lhs, "<=")
            pypsa.l_constraint(model, "NoAvailBiggerOne", constraints, self.range_limeqpv_b)

        if self.sum_pv_availability_zero:
            model.Sum_0_pv_availability = pyo.Constraint(
                expr=sum((manipulate_b[index, 0] / Cap_PV_0) * model.deltab[index] for index in self.range_limeqpv_b)
                == 0
            )

        if self.sum_demand_zero:
            model.Sum_0_demand = pyo.Constraint(
                expr=sum(manipulate_d[index, 0] * model.deltad[index] for index in self.range_energyeq_d) == 0
            )

        # set target (CAPEX)
        model.target = pyo.Constraint(
            expr=self.c[0, self.all_variables["CapacityPV"]] * model.x[self.all_variables["CapacityPV"]]
            + self.c[0, self.all_variables["CapacityBattery"]] * model.x[self.all_variables["CapacityBattery"]]
            >= self.target_capex
        )  # CAPEX is 626.10 € per year in original model

        # implement binary variables for warmstart
        for index in range(self.A.shape[0]):
            model.binary[index] = warm_binaries[index]

        # solve problem
        solver = SolverFactory("cplex")
        solver.options["mip_tolerances_absmipgap"] = self.mip_gap
        solver.solve(model, warmstart=True, tee=True)

        return model

    def get_model_data(self):
        """
        Calculates Warm binaries !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # function returns warm binaries for warmstart of MILP solver + Cap0
        # PYPSA Constraints + using indptr/indices -> fastest
        returns !!!!!!!!!!!
        """
        # create an instance of the model
        model = pyo.ConcreteModel()

        # define decision variables
        model.x = pyo.Var(range(len(self.all_variables)), within=pyo.NonNegativeReals)

        # define objective
        objective = pypsa.LExpression(
            [(self.c.data[k], model.x[self.c.indices[k]]) for k in range(self.c.indptr[0], self.c.indptr[1])]
        )
        pypsa.l_objective(model, objective, pypsa.minimize)

        # bounded constraints
        constraints = {}
        for row in range(self.A.shape[0]):
            lhs = pypsa.LExpression(
                [
                    (self.A.data[k], model.x[self.A.indices[k]])
                    for k in range(self.A.indptr[row], self.A.indptr[row + 1])
                ],
                -self.b[row],
            )
            constraints[row] = pypsa.LConstraint(lhs, "<=")

        pypsa.l_constraint(model, "aeqs", constraints, range(self.A.shape[0]))

        # equal constraints
        constraints = {}
        for row in range(self.H.shape[0]):
            lhs = pypsa.LExpression(
                [
                    (self.H.data[k], model.x[self.H.indices[k]])
                    for k in range(self.H.indptr[row], self.H.indptr[row + 1])
                ],
                -self.d[row],
            )
            constraints[row] = pypsa.LConstraint(lhs, "==")

        pypsa.l_constraint(model, "heqs", constraints, range(self.H.shape[0]))

        # solve problem
        solver = SolverFactory("cplex")
        solver.solve(model)

        # calculate vector with warm binaries
        # save solution of x
        x = np.asarray(model.x)
        for i, xi in enumerate(x):
            x[i] = pyo.value(xi)
        x = x.astype(float)

        # calculate auxiliary vector g
        g = self.A @ x - self.b

        # set for every gi = 0 the binary variable to one (due to rounding, instead of 0, the value -1e-9 is used)
        for i, gi in enumerate(g):
            if gi >= -1e-9:
                g[i] = 1
            else:
                g[i] = 0
        binaries = g.astype(int)
        Cap_PV_0 = pyo.value(model.x[self.all_variables["CapacityPV"]])

        return binaries, Cap_PV_0

    # checks if target is hit
    def check_target(self):
        print("Check Target")

    # evtl. auch draußen wie bei Johnny
    def check_valid_params(self):
        print("Check Params")
        # Gewichtungen zwischen 0 und 1 und zusammen 1

    # Time Series berechnen (so wie original bei Demand teilen) und ausgeben/speichern für weiterverarbeitung
    def create_timeseries(self, model):
        # create new pv availability time series
        change_pv_availability = []
        for index in self.range_limeqpv_b:
            change_pv_availability.append(pyo.value(model.deltab[index]))
        change_pv_availability = np.array(change_pv_availability)

        attacked_pv_availability = []
        col = self.all_variables["CapacityPV"]
        if self.attack_type == "absolute":
            for count, index in enumerate(self.range_limeqpv_b):
                attacked_pv_availability.append(-self.A[index, col] + change_pv_availability[count])

        elif self.attack_type == "relativ":
            for count, index in enumerate(self.range_limeqpv_b):
                attacked_pv_availability.append(
                    -self.A[index, col] + -self.A[index, col] * change_pv_availability[count]
                )

        attacked_pv_availability = np.array(attacked_pv_availability)

        # !!!!!!!!!!!!!!!!!!!!! DEMAND TEILEN!!!!!!!!!!!!!!!!!!
        # create new demand time series
        change_demand = []
        for index in self.range_energyeq_d:
            change_demand.append(pyo.value(model.deltad[index]))
        change_demand = np.array(change_demand)

        attacked_demand = []
        if self.attack_type == "absolute":
            for count, index in enumerate(self.range_energyeq_d):
                attacked_demand.append(self.d[index] + change_demand[count])

        elif self.attack_type == "relativ":
            for count, index in enumerate(self.range_energyeq_d):
                attacked_demand.append(self.d[index] + self.d[index] * change_demand[count])

        attacked_demand = np.array(attacked_demand)

        return attacked_pv_availability, attacked_demand

    def create_statistics(self, model):
        # calculate battery usage
        start_range = self.all_variables["EnergyBattery(0)"]
        end_range = self.all_variables["EnergyBattery(8759)"]
        range_battery = range(start_range, end_range)

        battery_usage = []
        for index in range_battery:
            battery_usage.append(pyo.value(model.x[index]))

        battery_usage = np.array(battery_usage)
        np.savetxt("battery_usage.csv", battery_usage, delimiter=",")

        # calculate energy of solar cell
        start_range = self.all_variables["EnergyPV(0)"]
        end_range = self.all_variables["EnergyPV(8759)"]
        range_energy_pv = range(start_range, end_range)

        energy_pv = []
        for index in range_energy_pv:
            energy_pv.append(pyo.value(model.x[index]))

        energy_pv = np.array(energy_pv)
        np.savetxt("energy_pv.csv", energy_pv, delimiter=",")

    def print_output(self, model, execution_time):
        index_pv = self.all_variables["CapacityPV"]
        index_battery = self.all_variables["CapacityBattery"]

        capacity_pv = pyo.value(model.x[index_pv])
        capacity_battery = pyo.value(model.x[index_battery])
        capex = self.c[0, index_pv] * capacity_pv + self.c[0, index_battery] * capacity_battery

        print("\n\nCapacity for the PV module: " + str(capacity_pv) + " kW")
        print("Capacity for the battery module: " + str(capacity_battery) + " kWh")
        print("\nCost of PV module per day: " + str(self.c[0, index_pv]) + " €")
        print("Cost of Battery module per day: " + str(self.c[0, index_battery]) + " €")
        print("CAPEX per day: " + str(capex) + " €")
        print("\nExecution of the algorithm took " + str(round(execution_time, 2)) + " seconds")
