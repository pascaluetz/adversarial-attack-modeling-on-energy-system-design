"""
model.py from Jonas model in an adjusted version:
- function added to read time series
- removed class "HouseModel"
- removed function "__init__"
- enabled to implement time series into model
- removed fixing and scaling parameters
- removed variables Demand, CostBuy, CostPV and CostBat
- renamed limEQ constraints to limEQpv and limEQbat
- included initial battery constraint in battery constraint list
- included solver options in function
- removed function getKPI
- implementation of variable SellEnergy, parameter sell_price and updating objective function and EnergyEQ
"""

import time

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# configures the parameters of the original pv house model
def getSettings():
    settingsDict = {
        "lifetime": 10,  # Years
        "cost_PV": 1000,  # €/kW
        "cost_Battery": 300,  # €/kWh
        "cost_buy": 0.25,  # €/kWh
        "sell_price": 0.05,  # €/kWh
        "dem_tot": 3500,  # kWh/Year
    }
    return settingsDict


# create HouseModel
def HouseModel(
    settings_dict,
    pv_availability,
    demand,
):
    # Step 0: Create an instance of the model
    model = pyo.ConcreteModel()

    # Step 1.1: Define index sets
    time = range(8760)

    # Step 1.2: Parameters
    lifetime = settings_dict["lifetime"]  # lifetime in years
    cost_PV = settings_dict["cost_PV"] / lifetime  # € / (lifetime * kW)
    cost_Battery = settings_dict["cost_Battery"] / lifetime  # € / (lifetime * kWh)
    cost_buy_ele = settings_dict["cost_buy"]  # € / kWh
    dem_tot = settings_dict["dem_tot"]  # kWh
    sell_price = settings_dict["sell_price"]  # € / kWh

    # Step 2: Define the decision variables
    # Electricity sector
    model.EnergyPV = pyo.Var(time, within=pyo.NonNegativeReals)
    model.EnergyBattery = pyo.Var(time, within=pyo.NonNegativeReals)
    model.EnergyBattery_IN = pyo.Var(time, within=pyo.NonNegativeReals)
    model.EnergyBattery_OUT = pyo.Var(time, within=pyo.NonNegativeReals)
    model.EnergyBuy = pyo.Var(time, within=pyo.NonNegativeReals)
    model.CapacityPV = pyo.Var(within=pyo.NonNegativeReals)
    model.CapacityBattery = pyo.Var(within=pyo.NonNegativeReals)
    model.SellEnergy = pyo.Var(time, within=pyo.NonNegativeReals)

    # Step 3: Define objective
    model.cost = pyo.Objective(
        expr=cost_PV * model.CapacityPV
        + cost_buy_ele * sum(model.EnergyBuy[i] for i in time)
        + cost_Battery * model.CapacityBattery
        - sell_price * sum(model.SellEnergy[i] for i in time),
        sense=pyo.minimize,
    )

    # Step 4: Constraints
    model.limEQpv = pyo.ConstraintList()
    for i in time:
        model.limEQpv.add(model.EnergyPV[i] <= model.CapacityPV * pv_availability[i])  # PV Upper Limit

    model.limEQbat = pyo.ConstraintList()
    for i in time:
        model.limEQbat.add(model.EnergyBattery[i] <= model.CapacityBattery)  # Battery Upper Limit

    model.batteryEQ = pyo.ConstraintList()
    for i in time:
        model.batteryEQ.add(
            expr=model.EnergyBattery[i]
            == model.EnergyBattery[time[i - 1]] - model.EnergyBattery_OUT[i] + model.EnergyBattery_IN[i]
        )  # Battery Equation

    model.EnergyEQ = pyo.ConstraintList()
    for i in time:
        model.EnergyEQ.add(
            expr=dem_tot * demand[i]
            == model.EnergyBuy[i]
            + model.EnergyBattery_OUT[i]
            - model.EnergyBattery_IN[i]
            + model.EnergyPV[i]
            - model.SellEnergy[i]
        )  # Energy Equation

    # write model to mps file
    model.write(filename=r"output\model.mps", io_options={"symbolic_solver_labels": True})

    # Change lines below to use other solver
    solver = SolverFactory("cplex")
    solver.solve(model)

    return model


# runs model
def run():
    start = time.time()
    settings = getSettings()
    pv_availability = np.loadtxt("../time_series/original_pv_availability.csv")
    demand = np.loadtxt("../time_series/bdew_demand.csv")
    solution = HouseModel(settings, pv_availability, demand)
    end = time.time()

    # print output
    print("Capacity for the PV module: " + str(pyo.value(solution.CapacityPV)) + " kW")
    print("Capacity for the Battery module: " + str(pyo.value(solution.CapacityBattery)) + " kWh")

    cost_PV_year = settings["cost_PV"] / settings["lifetime"]
    cost_BAT_year = settings["cost_Battery"] / settings["lifetime"]
    print("\nCost of PV module per year: " + str(cost_PV_year) + " €")
    print("Cost of Battery module per year: " + str(cost_BAT_year) + " €")
    print(
        "Capital expenditures: "
        + str(cost_PV_year * pyo.value(solution.CapacityPV) + cost_BAT_year * pyo.value(solution.CapacityBattery))
        + " €"
    )

    print("\nThe execution of the model took " + str(round(end - start, 2)) + " seconds")


run()
