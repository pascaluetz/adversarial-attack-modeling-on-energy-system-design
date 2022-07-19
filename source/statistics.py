import sys

import matplotlib.pyplot as plt
import numpy as np

from tools import convert_mps_to_array as converter

# import statsmodels.api as sm
#
# import tikzplotlib


# open model.mps, read it and initialize model variables
model_data = open(sys.path[2] + "\\source\\pv_house_year\\pv_house_year_basic\\model_sell.mps", "r").readlines()
all_variables, bounded_variables, c_row_names, c, A_row_names, A, H_row_names, H, b, d = converter.script(
    model_data, bounded_variables_as_equations=True
)

all_variables_as_dict = dict(zip(all_variables, range(0, len(all_variables))))
H_row_names_as_dict = dict(zip(H_row_names, range(0, len(H_row_names))))
A_row_names_as_dict = dict(zip(A_row_names, range(0, len(A_row_names))))

b_array = b.toarray().reshape(b.shape[0])
d_array = d.toarray().reshape(d.shape[0])

# get data from files
deltad = np.loadtxt("deltad.txt")
deltab = np.loadtxt("deltab.txt")

# get ranges
start_range = H_row_names_as_dict["c_e_EnergyEQ(1)_"]
end_range = H_row_names_as_dict["c_e_EnergyEQ(8760)_"] + 1
range_energyeq_d = range(start_range, end_range)

start_range = A_row_names_as_dict["c_u_limEQpv(1)_"]
end_range = A_row_names_as_dict["c_u_limEQpv(8760)_"] + 1
range_limeqpv_b = range(start_range, end_range)

# calc deltad
d_value = []
d_changed = []
for count, index in enumerate(range_energyeq_d):
    d_value.append(d[index, 0])

    # d_changed.append(d[index, 0] + d[index, 0] * deltad[index])  # relative changes
    d_changed.append(d[index, 0] + deltad[count])  # absolute changes

new_demand = np.array(d_changed)
np.savetxt("new_demand.csv", new_demand, delimiter=",")


plt.figure(1)
plt.plot(d_changed, label="Changed demand per hour", linestyle="solid")
plt.plot(d_value, label="Demand per hour", linestyle="solid")
plt.xlabel("Time [h]")
plt.ylabel("Demand [kWh]")
plt.title("Base Demand vs. Changed Demand per hour")
plt.legend(loc=1)
plt.show()

# calc deltab
b_value = []
b_changed = []
col = all_variables_as_dict["CapacityPV"]
for count, index in enumerate(range_limeqpv_b):
    b_value.append(-A[index, col])

    b_changed.append(-A[index, col] + deltab[count])  # absolute changes

new_solar = np.array(b_changed)
np.savetxt("new_solar.csv", new_solar, delimiter=",")

plt.figure(2)
plt.plot(b_changed, label="Changed availability per hour", linestyle="solid")
plt.plot(b_value, label="Availability per hour", linestyle="solid")
plt.xlabel("Time [h]")
plt.ylabel("Availability [%]")
plt.title("Base Availability vs. Changed Availability per hour")
plt.legend(loc=1)
plt.show()


# # smoothing b_changed
# kernel_size = 2
# kernel = np.ones(kernel_size) / kernel_size
# b_changed_smooth = np.convolve(b_changed, kernel, mode="same")
#
# plt.figure(3)
# plt.plot(b_changed_smooth, label="Changed availability per hour", linestyle="solid")
# plt.plot(b_value, label="Availability per hour", linestyle="solid")
# plt.xlabel("Time [h]")
# plt.ylabel("Availability [%]")
# plt.title("Base Availability vs. Changed Smooth Availability per hour")
# plt.legend(loc=1)
# plt.show()

# # plot battery usage
# battery_usage = np.loadtxt("battery_usage.txt")
#
# plt.figure(4)
# plt.plot(battery_usage, label="Battery usage per hour", linestyle="solid")
# plt.xlabel("Time [h]")
# plt.ylabel("Battery Usage [kWh]")
# plt.title("Battery usage per hour")
# plt.legend(loc=1)
# plt.show()

# # plot energy PV
# energy_pv = np.loadtxt("energy_pv.txt")
#
# plt.figure(5)
# plt.plot(energy_pv, label="PV Energy per hour", linestyle="solid")
# plt.xlabel("Time [h]")
# plt.ylabel("PV Energy [kW]")
# plt.title("PV Energy per hour")
# plt.legend(loc=1)
# plt.show()


# plt.savefig -> Pfad
# tikzplotlib.save("base_demand_vs_changed_demand_over_time.tex")
# save as pgf in matplotlib menu?

# \begin{figure}[h!]
# 	\centering
# 	\input{pics/payoff_over_iterations.tex}
# 	\caption{Veränderung des Payoffs der Microgrids über die Iterationen.}
# 	\label{fig:payoff_der_grids}
# \end{figure}

# https://blog.martisak.se/2019/09/29/publication_ready_figures/


# # plot from original csv data
# demand_old = np.loadtxt("demand_bdew.csv")
# demand_new = np.loadtxt("new_demand.csv")
# demand_old_array = []
#
# for entry in demand_old:
#     demand_old_array.append(entry * 3500)
#
# demand_old = np.array(demand_old_array)
#
# plt.figure(1)
# plt.plot(demand_new, label="Changed Demand per hour", linestyle="solid")
# plt.plot(demand_old, label="Original Demand per hour", linestyle="solid")
# plt.xlabel("Time [h]")
# plt.ylabel("Demand [kWh]")
# plt.legend(loc=4)
# # plt.xlim(1296, 1464)
# plt.show()
# # tikzplotlib.save("../../../../../05 Ausarbeitung/figures/15 case_study/deman_week.tex")
# # xtick={1300, 1340, 1380, 1420, 1460}, -> NOT in Tex-File for week
#
#
# # plot deltab
# availability_old = np.loadtxt("TS_PVAvail.csv")
# availability_new = np.loadtxt("new_solar.csv")
#
# plt.figure(2)
# plt.plot(availability_new, label="Changed Solar Availability per hour", linestyle="solid")
# plt.plot(availability_old, label="Original Solar Availability per hour", linestyle="solid")
# plt.xlabel("Time [h]")
# plt.ylabel("Solar Availability [%]")
# # plt.legend(loc=4)
# # plt.xlim(1296, 1464)
# plt.show()
# # tikzplotlib.save("../../../../../05 Ausarbeitung/figures/15 case_study/solar_week.tex")
# # xtick={1300, 1340, 1380, 1420, 1460}, -> NOT in Tex-File for week
# # to add to file:
# # width=\textwidth,
# # height=7cm,


# # Regression
# demand_change = demand_new - demand_old
# availability_change = availability_new - availability_old
#
# regression = sm.OLS(demand_change, sm.add_constant(availability_new)).fit()
# print(regression.summary())
#
# plt.figure(3)
# plt.scatter(availability_new, demand_change)
# plt.plot(availability_new, regression.params[1]*availability_new + regression.params[0], color='orange')
# plt.xlabel("Changed Solar Availability [%]")
# plt.ylabel("$\Delta \mathbf{demand}$ [kWh]")
# tikzplotlib.save("../../../../../05 Ausarbeitung/figures/15 case_study/regression.tex")
