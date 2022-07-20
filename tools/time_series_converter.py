"""
Converts PV availability and demand year times series to the average of each hour over 365 days. The result is are 24h
time series for a smaller model.
"""

import numpy as np

# load year time series
availability_pv = np.loadtxt("../time_series/original_pv_availability.csv")
DemandVal = np.loadtxt("../time_series/original_demand.csv")

# set ranges
hours = range(24)
days = range(365)

# create empty lists for 24h time series
availability_pv_24h = []
DemandVal_24h = []

# loop over all days hourly and add the average to array
for hour in hours:
    value_availability = 0
    value_demand = 0
    # sum up all values of each hour for 365 days
    for day in days:
        value_availability += availability_pv[hour + day * 24]
        value_demand += DemandVal[hour + day * 24]

    # calculate average
    value_availability = value_availability / 365
    value_demand = value_demand / 365

    availability_pv_24h.append(value_availability)
    DemandVal_24h.append(value_demand)

availability_pv_24h = np.array(availability_pv_24h)
DemandVal_24h = np.array(DemandVal_24h)

# write both datasets in csv
np.savetxt("../time_series/original_pv_availability_24h.csv", availability_pv_24h, delimiter=",")
np.savetxt("../time_series/original_demand_24h.csv", DemandVal_24h, delimiter=",")
