"""
Converts PV availability and demand year times series to the average of each hour over 365 days. The result is are 24h
time series for a smaller model.
"""

import csv


# extract digits from list
def extractDigits(lst):
    return [[el] for el in lst]


# create empty arrays
availability_pv = []
DemandVal = []

# open time series and add them to array
with open("../time_series/original_pv_availability.csv", "r") as file:
    reader = csv.reader(file, delimiter="\n")
    for row in reader:
        availability_pv.append(float(row[0]))

with open("../time_series/original_demand.csv", "r") as file:
    reader = csv.reader(file, delimiter="\n")
    for row in reader:
        DemandVal.append(float(row[0]))

# set ranges
hours = range(24)
days = range(365)

# create empty arrays for 24h time series
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

# write both datasets in csv
with open("../time_series/original_pv_availability_24h.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(extractDigits(availability_pv_24h))

with open("../time_series/original_demand_24h.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(extractDigits(DemandVal_24h))
