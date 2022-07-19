"""
Runs PV House Model solved with Karush-Kuhn-Tucker Conditions and finds a minimal delta.
- In function find_minimal_delta it is possible to set which variables should be manipulated and also the allowed
absolute mip gap to the optimal solution
- In function run it is possible to adjust the output data of the model
"""

import os
import sys
import time

import numpy as np
import pyomo.environ as pyo
import pypsa.opt as pypsa
from pyomo.opt import SolverFactory
from scipy.sparse import csr_matrix

from tools import convert_mps_to_array as converter

# open model.mps, read it and initialize model variables
model_data = open(
    sys.path[2] + os.path.normpath("/source/pv_house_year/pv_house_year_basic/model_sell.mps"), "r"
).readlines()
all_variables, bounded_variables, c_row_names, c, A_row_names, A, H_row_names, H, b, d = converter.script(
    model_data, bounded_variables_as_equations=True
)

all_variables_as_dict = dict(zip(all_variables, range(0, len(all_variables))))
H_row_names_as_dict = dict(zip(H_row_names, range(0, len(H_row_names))))
A_row_names_as_dict = dict(zip(A_row_names, range(0, len(A_row_names))))

b_array = b.toarray().reshape(b.shape[0])
d_array = d.toarray().reshape(d.shape[0])


# function returns warm binaries for warmstart of MILP solver
# PYPSA Constraints + using indptr/indices -> fastest
def get_warm_binaries():
    # Step 0: Create an instance of the model
    model = pyo.ConcreteModel()

    # Step 1: Define decisions
    model.x = pyo.Var(range(len(all_variables)), within=pyo.NonNegativeReals)

    # Step 2: Define objective
    objective = pypsa.LExpression([(c.data[k], model.x[c.indices[k]]) for k in range(c.indptr[0], c.indptr[1])])
    pypsa.l_objective(model, objective, pypsa.minimize)

    # Step 3: Bounded constraints
    constraints = {}
    for row in range(A.shape[0]):
        lhs = pypsa.LExpression(
            [(A.data[k], model.x[A.indices[k]]) for k in range(A.indptr[row], A.indptr[row + 1])], -b[row, 0]
        )
        constraints[row] = pypsa.LConstraint(lhs, "<=")

    pypsa.l_constraint(model, "aeqs", constraints, range(A.shape[0]))

    # Step 4: Equal constraints
    constraints = {}
    for row in range(H.shape[0]):
        lhs = pypsa.LExpression(
            [(H.data[k], model.x[H.indices[k]]) for k in range(H.indptr[row], H.indptr[row + 1])], -d[row, 0]
        )
        constraints[row] = pypsa.LConstraint(lhs, "==")

    pypsa.l_constraint(model, "heqs", constraints, range(H.shape[0]))

    # Step 5: Solve problem
    solver = SolverFactory("cplex")
    solver.solve(model)
    # results.write()

    # Step 6: Calculate vector with warm binaries
    x = np.asarray(model.x)
    for i, xi in enumerate(x):
        x[i] = pyo.value(xi)
    x = x.astype(float)

    values = A @ x - b_array
    for i, value in enumerate(values):
        if value >= -1e-9:
            values[i] = 1
        else:
            values[i] = 0
    values = values.astype(int)
    Cap_PV_0 = pyo.value(model.x[all_variables_as_dict["CapacityPV"]])

    # return model
    return values, Cap_PV_0


# finds minimal delta with PyPSA + indptr/indices
def find_minimal_delta(warm_binaries, cap_PV_0, mip_gap):
    # Step 0: Create an instance of the model
    model = pyo.ConcreteModel()

    # Step 1: Parameters
    big_m = 1e2

    # Step 2: Define decisions
    model.x = pyo.Var(range(len(all_variables)), within=pyo.NonNegativeReals)
    model.lambdas = pyo.Var(range(A.shape[0]), within=pyo.NonNegativeReals)
    model.ny = pyo.Var(range(H.shape[0]), within=pyo.Reals)
    model.binary = pyo.Var(range(A.shape[0]), within=pyo.Binary)

    # # for method minimize max delta
    # model.max_deltad = pyo.Var(within=pyo.NonNegativeReals)

    # # for absolute values
    model.deltac = pyo.Var(range(3), within=pyo.Reals)
    # model.deltac_pos = pyo.Var(range(3), within=pyo.NonNegativeReals)
    # model.deltac_neg = pyo.Var(range(3), within=pyo.NonPositiveReals)

    model.deltad = pyo.Var(range(d.shape[0]), within=pyo.Reals)
    # model.deltad_pos = pyo.Var(range(d.shape[0]), within=pyo.NonNegativeReals)
    # model.deltad_neg = pyo.Var(range(d.shape[0]), within=pyo.NonPositiveReals)

    model.deltab = pyo.Var(range(b.shape[0]), within=pyo.Reals)
    # model.deltab_pos = pyo.Var(range(b.shape[0]), within=pyo.NonNegativeReals)
    # model.deltab_neg = pyo.Var(range(b.shape[0]), within=pyo.NonPositiveReals)

    # Step 3: Define vectors to add manipulation (for absolute manipulation change nonzero value of manipulation to one)
    cbat = False
    cpv = False
    cbuy = False
    d_manipulate = True
    b_manipulate = True

    # save the amount of changed deltas for the calculation of the objective
    changed_deltas = 0

    # manipulation of c in lagrangian
    manipulate_c_cbat = np.zeros(c.shape[1])
    if cbat:
        index = all_variables_as_dict["CapacityBattery"]
        manipulate_c_cbat[index] = c[0, index]
        changed_deltas += 1
    manipulate_c_cbat = csr_matrix(manipulate_c_cbat)

    manipulate_c_cpv = np.zeros(c.shape[1])
    if cpv:
        index = all_variables_as_dict["CapacityPV"]
        manipulate_c_cpv[index] = c[0, index]
        changed_deltas += 1
    manipulate_c_cpv = csr_matrix(manipulate_c_cpv)

    manipulate_c_cbuy = np.zeros(c.shape[1])
    if cbuy:
        start_range = all_variables_as_dict["EnergyBuy(0)"]
        end_range = all_variables_as_dict["EnergyBuy(8759)"] + 1
        range_cbuy = range(start_range, end_range)
        changed_deltas += end_range - start_range
        for index in range_cbuy:
            manipulate_c_cbuy[index] = c[0, index]
    manipulate_c_cbuy = csr_matrix(manipulate_c_cbuy)

    # manipulation of d in primal feasibility
    manipulate_d = np.zeros(d.shape[0])
    start_range = H_row_names_as_dict["c_e_EnergyEQ(1)_"]
    end_range = H_row_names_as_dict["c_e_EnergyEQ(8760)_"] + 1
    range_energyeq_d = range(start_range, end_range)
    if d_manipulate:
        changed_deltas += end_range - start_range
        for index in range_energyeq_d:
            # manipulate_d[index] = d[index, 0]  # demand(t) relative change
            manipulate_d[index] = 1  # demand(t) absolute change
    manipulate_d = csr_matrix(manipulate_d).transpose()

    # manipulation of b in primal feasibility and complementary slackness
    manipulate_b = np.zeros(b.shape[0])
    start_range = A_row_names_as_dict["c_u_limEQpv(1)_"]
    end_range = A_row_names_as_dict["c_u_limEQpv(8760)_"] + 1
    range_limeqpv_b = range(start_range, end_range)
    # put Cap_PV_0 at the place to have a rough estimation
    if b_manipulate:
        changed_deltas += end_range - start_range
        col = all_variables_as_dict["CapacityPV"]
        for row in range_limeqpv_b:
            # manipulate_b[row] = cap_PV_0 * -A[row, col]  # availability(t) relative change
            manipulate_b[row] = cap_PV_0  # availability(t) absolute change
    manipulate_b = csr_matrix(manipulate_b).transpose()

    # Step 4: Define objective with absolute values
    # factor = 1 / changed_deltas
    # deltac_tuples = []
    # for k in range(len(model.deltac)):
    #     deltac_tuples.append((factor, model.deltac_pos[k]))
    #     deltac_tuples.append((-factor, model.deltac_neg[k]))
    #
    # deltad_tuples = []
    # for k in range_energyeq_d:
    #     deltad_tuples.append((factor, model.deltad_pos[k]))
    #     deltad_tuples.append((-factor, model.deltad_neg[k]))
    #
    # deltab_tuples = []
    # for k in range_limeqpv_b:
    #     deltab_tuples.append((factor, model.deltab_pos[k]))
    #     deltab_tuples.append((-factor, model.deltab_neg[k]))
    #
    # objective = pypsa.LExpression(deltac_tuples + deltad_tuples + deltab_tuples)
    # pypsa.l_objective(model, objective, pypsa.minimize)

    # Quadratic Objective
    # calculating scaling coefficients
    d_numpy = []
    for index in range_energyeq_d:
        d_numpy.append(d[index, 0])
    d_numpy = np.array(d_numpy)

    b_numpy = []
    col = all_variables_as_dict["CapacityPV"]
    for index in range_limeqpv_b:
        b_numpy.append(-A[index, col])
    b_numpy = np.array(b_numpy)

    model.size_delta = pyo.Objective(
        expr=sum(model.deltac[i] ** 2 for i in range(3))
        + 0.5
        * sum(((model.deltad[i] - d_numpy.min()) / (d_numpy.max() - d_numpy.min())) ** 2 for i in range_energyeq_d)
        + 0.5
        * sum(((model.deltab[i] - b_numpy.min()) / (b_numpy.max() - b_numpy.min())) ** 2 for i in range_limeqpv_b),
        sense=pyo.minimize,
    )

    # # minimize max function
    # objective = pypsa.LExpression([(1, model.max_deltad)])
    # pypsa.l_objective(model, objective, pypsa.minimize)

    # # Check if there is a solution constraint -> >17 min
    # # old objective
    # model.value = pyo.Objective(
    #     expr=sum(c[0, col] * model.x[col] for col in range(len(all_variables))), sense=pyo.minimize
    # )
    #
    # # checking condition (check if there is a solution)
    # check_value = 0.1
    # objective_constraint = {}
    # objective_constraint[0] = pypsa.LConstraint(
    #     pypsa.LExpression(deltac_tuples + deltad_tuples + deltab_tuples, -check_value), "<="
    # )
    # pypsa.l_constraint(model, "ObjectiveConstraint", objective_constraint, range(1))

    # Step 5: Constraints
    # # get maximum of deltad
    # constraints = {}
    # for index in range_energyeq_d:
    #     lhs = pypsa.LExpression([(manipulate_d[index, 0], model.deltad_pos[index]),
    #                              (-manipulate_d[index, 0], model.deltad_neg[index])])
    #     rhs = pypsa.LExpression([(1, model.max_deltad)])
    #     constraints[index] = pypsa.LConstraint(lhs, "<=", rhs)
    # pypsa.l_constraint(model, "MaxDeltaD", constraints, range_energyeq_d)

    # # absolute value constraints
    # constraints = {}
    # for index in range(len(model.deltac)):
    #     lhs = pypsa.LExpression([(1, model.deltac[index])])
    #     rhs = pypsa.LExpression([(1, model.deltac_pos[index]), (1, model.deltac_neg[index])])
    #     constraints[index] = pypsa.LConstraint(lhs, "==", rhs)
    # pypsa.l_constraint(model, "DeltaCEQ", constraints, range(len(model.deltac)))
    #
    # constraints = {}
    # for index in range_energyeq_d:
    #     lhs = pypsa.LExpression([(1, model.deltad[index])])
    #     rhs = pypsa.LExpression([(1, model.deltad_pos[index]), (1, model.deltad_neg[index])])
    #     constraints[index] = pypsa.LConstraint(lhs, "==", rhs)
    # pypsa.l_constraint(model, "DeltaDEQ", constraints, range_energyeq_d)
    #
    # constraints = {}
    # for index in range_limeqpv_b:
    #     lhs = pypsa.LExpression([(1, model.deltab[index])])
    #     rhs = pypsa.LExpression([(1, model.deltab_pos[index]), (1, model.deltab_neg[index])])
    #     constraints[index] = pypsa.LConstraint(lhs, "==", rhs)
    # pypsa.l_constraint(model, "DeltaBEQ", constraints, range_limeqpv_b)

    # no negative demand constraint
    constraints = {}
    for index in range_energyeq_d:
        lhs = pypsa.LExpression([(manipulate_d[index, 0], model.deltad[index])], d[index, 0])
        constraints[index] = pypsa.LConstraint(lhs, ">=")
    pypsa.l_constraint(model, "NoNegativeDemand", constraints, range_energyeq_d)

    # no negative availability constraints
    constraints = {}
    col = all_variables_as_dict["CapacityPV"]
    for index in range_limeqpv_b:
        lhs = pypsa.LExpression([(manipulate_b[index, 0] / cap_PV_0, model.deltab[index])], -A[index, col])
        constraints[index] = pypsa.LConstraint(lhs, ">=")
    pypsa.l_constraint(model, "NoNegativAvail", constraints, range_limeqpv_b)

    # no availability < 1 constraints
    constraints = {}
    col = all_variables_as_dict["CapacityPV"]
    for index in range_limeqpv_b:
        lhs = pypsa.LExpression([(manipulate_b[index, 0] / cap_PV_0, model.deltab[index])], -A[index, col] - 1)
        constraints[index] = pypsa.LConstraint(lhs, "<=")
    pypsa.l_constraint(model, "NoAvailBiggerOne", constraints, range_limeqpv_b)

    # sum of delta d and b should be 0 (just switching the time series)
    constraints = {}

    lhs = pypsa.LExpression([(manipulate_d[index, 0], model.deltad[index]) for index in range_energyeq_d])
    constraints[0] = pypsa.LConstraint(lhs, "==")

    lhs = pypsa.LExpression([(manipulate_b[index, 0] / cap_PV_0, model.deltab[index]) for index in range_limeqpv_b])
    constraints[1] = pypsa.LConstraint(lhs, "==")

    pypsa.l_constraint(model, "Sum_0", constraints, range(2))

    # # |delta availability| <= 0.05
    # constraints = {}
    # for index in range_limeqpv_b:
    #     lhs = pypsa.LExpression([(manipulate_b[index, 0] / cap_PV_0, model.deltab[index])], -0.05)
    #     constraints[index] = pypsa.LConstraint(lhs, "<=")
    # pypsa.l_constraint(model, "deltableq", constraints, range_limeqpv_b)
    #
    # constraints = {}
    # for index in range_limeqpv_b:
    #     lhs = pypsa.LExpression([(-manipulate_b[index, 0] / cap_PV_0, model.deltab[index])], -0.05)
    #     constraints[index] = pypsa.LConstraint(lhs, "<=")
    # pypsa.l_constraint(model, "deltabgeq", constraints, range_limeqpv_b)

    # # |delta demand| <= 0.18
    # constraints = {}
    # for index in range_energyeq_d:
    #     lhs = pypsa.LExpression([(manipulate_d[index, 0], model.deltad[index])], -0.06)
    #     constraints[index] = pypsa.LConstraint(lhs, "<=")
    # pypsa.l_constraint(model, "deltadleq", constraints, range_energyeq_d)
    #
    # constraints = {}
    # for index in range_energyeq_d:
    #     lhs = pypsa.LExpression([(-manipulate_d[index, 0], model.deltad[index])], -0.06)
    #     constraints[index] = pypsa.LConstraint(lhs, "<=")
    # pypsa.l_constraint(model, "deltdbgeq", constraints, range_energyeq_d)

    # Primal feasibility
    # Bounded constraints
    constraints = {}
    for row in range(A.shape[0]):
        list_of_tuples = [(A.data[k], model.x[A.indices[k]]) for k in range(A.indptr[row], A.indptr[row + 1])]
        list_of_tuples.append((-manipulate_b[row, 0], model.deltab[row]))
        lhs = pypsa.LExpression(list_of_tuples, -b[row, 0])
        constraints[row] = pypsa.LConstraint(lhs, "<=")

    pypsa.l_constraint(model, "AEQS", constraints, range(A.shape[0]))

    # Equal constraints
    constraints = {}
    for row in range(H.shape[0]):
        list_of_tuples = [(H.data[k], model.x[H.indices[k]]) for k in range(H.indptr[row], H.indptr[row + 1])]
        list_of_tuples.append((-manipulate_d[row, 0], model.deltad[row]))
        lhs = pypsa.LExpression(list_of_tuples, -d[row, 0])
        constraints[row] = pypsa.LConstraint(lhs, "==")

    pypsa.l_constraint(model, "HEQS", constraints, range(H.shape[0]))

    # Dual feasibility -> through lambdas within NonNegativeReals

    # Complementary slackness
    constraints = {}
    for row in range(A.shape[0]):
        lhs = pypsa.LExpression([(-big_m, model.binary[row]), (1, model.lambdas[row])])
        constraints[row] = pypsa.LConstraint(lhs, "<=")
    pypsa.l_constraint(model, "SlackEQ1", constraints, range(A.shape[0]))

    constraints = {}
    for row in range(A.shape[0]):
        list_of_tuples = [(A.data[k], model.x[A.indices[k]]) for k in range(A.indptr[row], A.indptr[row + 1])]
        list_of_tuples.append((-manipulate_b[row, 0], model.deltab[row]))
        rhs = pypsa.LExpression(list_of_tuples, -b[row, 0])
        lhs = pypsa.LExpression([(big_m, model.binary[row])], -big_m)
        constraints[row] = pypsa.LConstraint(lhs, "<=", rhs)
    pypsa.l_constraint(model, "SlackEQ2", constraints, range(A.shape[0]))

    # Stationarity / Derivation of the Lagrange function
    constraints = {}
    AT = A.transpose().tocsr()
    HT = H.transpose().tocsr()

    for col in range(len(all_variables)):
        a_tuples = [(AT.data[k], model.lambdas[AT.indices[k]]) for k in range(AT.indptr[col], AT.indptr[col + 1])]
        h_tuples = [(HT.data[k], model.ny[HT.indices[k]]) for k in range(HT.indptr[col], HT.indptr[col + 1])]
        list_of_tuples = a_tuples + h_tuples
        list_of_tuples.append((manipulate_c_cbat[0, col], model.deltac[0]))
        list_of_tuples.append((manipulate_c_cpv[0, col], model.deltac[1]))
        list_of_tuples.append((manipulate_c_cbuy[0, col], model.deltac[2]))
        lhs = pypsa.LExpression(list_of_tuples, c[0, col])
        constraints[col] = pypsa.LConstraint(lhs, "==")
    pypsa.l_constraint(model, "LagrangianEQ", constraints, range(len(all_variables)))

    # # Step 6: Set target (sum of PV capacity and battery capacity)
    # model.target = pyo.Constraint(
    #     expr=model.x[all_variables_as_dict["CapacityPV"]] + model.x[all_variables_as_dict["CapacityBattery"]] >= 11.08
    # )  # 9.23 kWh is sum of original

    # Step 6: Set target (CAPEX)
    model.target = pyo.Constraint(
        expr=c[0, all_variables_as_dict["CapacityPV"]] * model.x[all_variables_as_dict["CapacityPV"]]
        + c[0, all_variables_as_dict["CapacityBattery"]] * model.x[all_variables_as_dict["CapacityBattery"]]
        >= 751.32
    )  # CAPEX is 626.10 € per year in original model

    # Write files for investigations
    # model.write(filename="debugging_with_labels.lp", io_options={"symbolic_solver_labels": True})
    # model.write(filename="debugging_without_labels.lp")

    # Step 7: Implement binary variables for warmstart
    for index in range(A.shape[0]):
        model.binary[index] = warm_binaries[index]

    # # Step 8: Set branch priorities higher for limEQpv constraints
    # model.priority = pyo.Suffix(direction=pyo.Suffix.EXPORT, datatype=pyo.Suffix.INT)
    # model.direction = pyo.Suffix(direction=pyo.Suffix.EXPORT, datatype=pyo.Suffix.INT)
    # for index in range_limeqpv_b:
    #     model.priority.set_value(model.binary[index], 1)
    #     # direction should be passed as 1 (up), 0 (default - algorithm decides), or -1 (down). But 'default' does
    #     # not work -> probably a bug
    #     model.direction.set_value(model.binary[index], 1)

    # Step 9: Solve problem
    solver = SolverFactory("cplex")
    solver.options["mip_tolerances_absmipgap"] = mip_gap
    solver.solve(model, warmstart=True, tee=True)
    # solver.solve(model, warmstart=True, priorities=True, tee=True)
    # results.write()

    return model


# runs kkt approach and prints output
def run(*warm_binaries, mip_gap):
    start_time = time.time()
    if not warm_binaries:
        warm_binaries, cap_PV_0 = get_warm_binaries()
    else:
        warm_binaries = list(warm_binaries)[0]
    # Sum: cap_PV_0 = 5.7723541648595  # 4.990587586641461 from original model
    # CAPEX: cap_PV_0,1 IT = 6.132934037759659 # 4.990587586641461 from original model
    print("Cap0 ist: " + str(cap_PV_0))
    solution = find_minimal_delta(warm_binaries=warm_binaries, cap_PV_0=cap_PV_0, mip_gap=mip_gap)
    end_time = time.time()

    # output definition
    general = True
    capex = True
    check_complementary_slackness = False
    show_lambdas = False
    all_x = False
    nys = False
    battery_usage = False
    energy_pv = False
    deltas_not_zero = False
    deltac = False
    deltab = True
    deltad = True
    model_pprint = False

    # output
    capacity_pv = pyo.value(solution.x[all_variables_as_dict["CapacityPV"]])
    capacity_battery = pyo.value(solution.x[all_variables_as_dict["CapacityBattery"]])

    start_range = H_row_names_as_dict["c_e_EnergyEQ(1)_"]
    end_range = H_row_names_as_dict["c_e_EnergyEQ(8760)_"] + 1
    range_energyeq_d = range(start_range, end_range)

    start_range = A_row_names_as_dict["c_u_limEQpv(1)_"]
    end_range = A_row_names_as_dict["c_u_limEQpv(8760)_"] + 1
    range_limeqpv_b = range(start_range, end_range)

    if general:
        print("\n\nCapacity for the PV module: " + str(capacity_pv) + " kWh")
        print("Capacity for the Battery module: " + str(capacity_battery) + " kWh")
        print("Sum of PV capacity and battery capacity: " + str(capacity_pv + capacity_battery) + " kWh")
        print("\nExecution of Base Model took " + str(round(end_time - start_time, 2)) + " seconds")

    if capex:
        index = all_variables_as_dict["CapacityBattery"]
        if str(solution.deltac[0]) == "deltac[0]":
            cost_bat_day = c[0, index]
        else:
            cost_bat_day = c[0, index] + c[0, index] * solution.deltac[0]

        index = all_variables_as_dict["CapacityPV"]
        if str(solution.deltac[1]) == "deltac[1]":
            cost_pv_day = c[0, index]
        else:
            cost_pv_day = c[0, index] + c[0, index] * solution.deltac[1]

        capex = cost_pv_day * capacity_pv + cost_bat_day * capacity_battery
        print("\nCost of PV module per day: " + str(cost_pv_day) + " €")
        print("Cost of Battery module per day: " + str(cost_bat_day) + " €")
        print("CAPEX per day: " + str(capex) + " €")

    if check_complementary_slackness:
        for row in range(A.shape[0]):
            lambdas = pyo.value(solution.lambdas[row])
            gix = (
                sum(A.data[k] * pyo.value(solution.x[A.indices[k]]) for k in range(A.indptr[row], A.indptr[row + 1]))
                - b[row, 0]
                - cap_PV_0 * pyo.value(solution.deltab[row])
            )

            if lambdas != 0 and gix >= 1e-13:
                print("\nError in row: " + str(A_row_names[row]))
                print("Lambda is: " + str(lambdas) + " and g(x) is: " + str(gix))
                print("The respective binary variable is: " + str(pyo.value(solution.binary[row])))
        print("Check Complementary Slackness: Done")

    if show_lambdas:
        for i in range(A.shape[0]):
            value = pyo.value(solution.lambdas[i])
            print("Lambda " + str(i) + " has the value: " + str(value) + " belongs to the equation: " + A_row_names[i])

    if all_x:
        for i in range(len(all_variables)):
            value = pyo.value(solution.x[i])
            print("Variable " + all_variables[i] + " has the value: " + str(value))
            if value >= 1e2:
                print("Variable " + all_variables[i] + " is greater than 1e2 with the value: " + str(value))
            if value < 0:
                print("Variable " + all_variables[i] + " is less than 0 with the value: " + str(value))

    if nys:
        for i in range(H.shape[0]):
            value = pyo.value(solution.ny[i])
            print(
                "Variable ny "
                + str(i)
                + " has the value: "
                + str(value)
                + " and belongs to the equation: "
                + H_row_names[i]
            )
    if battery_usage:
        start_range = all_variables_as_dict["EnergyBattery(0)"]
        end_range = all_variables_as_dict["EnergyBattery(8759)"]
        range_battery = range(start_range, end_range)

        battery_usage_array = []
        for index in range_battery:
            battery_usage_array.append(pyo.value(solution.x[index]))

        battery_usage_array = np.array(battery_usage_array)
        np.savetxt("battery_usage.txt", battery_usage_array, delimiter=", ")

    if energy_pv:
        start_range = all_variables_as_dict["EnergyPV(0)"]
        end_range = all_variables_as_dict["EnergyPV(8759)"]
        range_energy_pv = range(start_range, end_range)

        energy_pv_array = []
        for index in range_energy_pv:
            energy_pv_array.append(pyo.value(solution.x[index]))

        energy_pv_array = np.array(energy_pv_array)
        np.savetxt("energy_pv.txt", energy_pv_array, delimiter=", ")

    if deltas_not_zero:
        for i in range(3):
            if pyo.value(solution.deltac[i]) != 0:
                print("DeltaC " + str(i) + " is: " + str(pyo.value(solution.deltac[i])))

        for i in range_energyeq_d:
            if pyo.value(solution.deltad[i]) != 0:
                print("DeltaD " + str(i) + " is: " + str(pyo.value(solution.deltad[i])))

        for i in range_limeqpv_b:
            if pyo.value(solution.deltab[i]) != 0:
                print("DeltaB " + str(i) + " is: " + str(pyo.value(solution.deltab[i])))

    if deltac:
        deltac_array = []
        for i in range(3):
            deltac_array.append(pyo.value(solution.deltac[i]))
        print("Delta C is: " + str(deltac_array))

    if deltab:
        deltab_array = []
        for i in range_limeqpv_b:
            deltab_array.append(pyo.value(solution.deltab[i]))
        deltab = np.array(deltab_array)
        np.savetxt("deltab.txt", deltab, delimiter=", ")

    if deltad:
        deltad_array = []
        for i in range_energyeq_d:
            deltad_array.append(pyo.value(solution.deltad[i]))
        deltad = np.array(deltad_array)
        np.savetxt("deltad.txt", deltad, delimiter=", ")

    if model_pprint:
        solution.pprint()


# standard mip_gap="1e-2"
# run(mip_gap="1")
# run(mip_gap="5")
# run(mip_gap="17.77")
run(mip_gap="1000000000")
