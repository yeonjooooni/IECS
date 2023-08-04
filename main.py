import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm
import numpy as np
from itertools import cycle
from matplotlib import pyplot as plt
from pyVRP import *
from output import *
od_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (od_matrix) 수정완료.csv')

pivot_table = pd.read_csv("./pivot_table_filled.csv", encoding='cp949', index_col=[0])

real_distance_matrix = pd.read_csv("./distance_matrix.csv", index_col=0)

demand_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv', encoding='cp949')
veh_table = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (veh_table).csv', encoding='cp949')
tmp_veh = veh_table[veh_table['StartCenter'] == 'O_179']

# for i in range(1, 6):
#     for j in range(4):
tmp_df = demand_df[demand_df['date']=='2023-05-02']
tmp_df = tmp_df[tmp_df['Group'].isin([0])]
tmp_df = tmp_df[tmp_df['터미널ID']=='O_179']

id_list_only_in_tmp_df = list(set(tmp_df['터미널ID'].values.tolist() + tmp_df['착지ID'].values.tolist()))
pivot_table = pivot_table.loc[id_list_only_in_tmp_df,id_list_only_in_tmp_df]
pivot_table = pivot_table.sort_index(axis=1, ascending=False)
pivot_table = pivot_table.sort_index(axis=0,  ascending=False)

# '착지_ID'열의 각 값의 인덱스를 담을 리스트 초기화
index_positions = [list(pivot_table.index).index("O_179")]
cbm_list = [0]

# tmp_df의 '착지_ID'열의 각 값에 대해 pivot_table.index 리스트의 인덱스를 찾습니다.
for i in range(len(tmp_df)):
    value = tmp_df['착지ID'].values.tolist()[i]
    index_positions.append(list(pivot_table.index).index(value))
    cbm_list.append(float(tmp_df['CBM'].values[i]))

departure_coordinates = demand_df.drop_duplicates(['착지ID'])[['착지ID', '하차지_위도', '하차지_경도']]
departure_coordinates.columns = ['ID', 'y', 'x']
origin_coordinates = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (Terminals).csv", encoding='cp949', usecols = [0,1,2])
origin_coordinates.columns = departure_coordinates.columns
coordinates = pd.concat([departure_coordinates, origin_coordinates], ignore_index=True)
coordinates = coordinates.set_index(['ID'])
coordinates = coordinates.reindex(index=pivot_table.index)
coordinates = coordinates.loc[id_list_only_in_tmp_df].sort_index(ascending=False).reset_index(drop=True)

# 시간을 분 단위로 변환하는 함수
def time_to_minutes(time_str):
    hour, minute = map(int, time_str.split(':'))
    return hour * 60 + minute

# 3일간의 하차 가능 시작과 끝 시간 리스트를 구하는 함수
# 여기서 이미 time_window와 무관하게 3일차(4320분)에 딱 cut하도록 만들어 놓음
def get_trip_time_lists(start_time, end_time, num_days=3):
    start_time_minutes = time_to_minutes(start_time)
    end_time_minutes = time_to_minutes(end_time)

    start_list = []
    end_list = []
    
    for day in range(num_days):
        if start_time_minutes + day * 24 * 60 > 4320:
            start_list.append(4320)
        else:
            start_list.append((start_time_minutes + day * 24 * 60) )

        if start_time_minutes > end_time_minutes:
            if end_time_minutes + (day+1) * 24 * 60 > 4320:
                end_list.append(4320)
            else:
                end_list.append((end_time_minutes + (day+1) * 24 * 60))
        else:
            if end_time_minutes + day * 24 * 60 > 4320:
                end_list.append(4320)
            else:
                end_list.append((end_time_minutes + day * 24 * 60) )

    return start_list, end_list

# 3일간의 하차 가능 시작과 끝 시간 리스트 계산
trip_start_times = [[0,0,0]]
trip_end_times = [[4320,4320,4320]]

for idx, row in tmp_df.iterrows():
    start_time = row['하차가능시간_시작']
    end_time = row['하차가능시간_종료']
    start_list, end_list = get_trip_time_lists(start_time, end_time)
    trip_start_times.append(start_list)
    trip_end_times.append(end_list)

parameters = pd.DataFrame({
    'arrive_station': index_positions,
    'TW_early':trip_start_times,
    'TW_late':trip_end_times,
    'TW_service_time':60,
    'TW_wait_cost':0,
    'cbm':cbm_list
})

# Tranform to Numpy Array
coordinates = coordinates.values
parameters  = parameters.values
distance_matrix = pivot_table.values
real_distance_matrix = real_distance_matrix.values

# Parameters - Model
n_depots    =  1          # The First n Rows of the 'distance_matrix' or 'coordinates' are Considered as Depots
time_window = 'with'    # 'with', 'without'
route       = 'closed'     # 'open', 'closed'
model       = 'vrp'        # 'tsp', 'mtsp', 'vrp'
graph       = True        # True, False

# Parameters - Vehicle
vehicle_types = tmp_veh.shape[0] # 해당 출발지에 속한 차량 수 전부
fixed_cost    = tmp_veh['FixedCost'].values.tolist()
variable_cost = tmp_veh['VariableCost'].values.tolist()
capacity      = tmp_veh['MaxCapaCBM'].values.tolist()
velocity      = [1] * vehicle_types # 전부 1
fleet_size    = [1] * vehicle_types # 전부 1

# Parameters - GA
penalty_value   = 1000000    # GA Target Function Penalty Value for Violating the Problem Constraints
population_size = 10      # GA Population Size
mutation_rate   = 0.2     # GA Mutation Rate
elite           = 1        # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained 
generations     = 2     # GA Number of Generations

fleet_used = [0]*vehicle_types
# Run GA Function
ga_report, ga_vrp, fleet_used = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, real_distance_matrix, population_size, vehicle_types, n_depots, route, model, time_window, fleet_size, mutation_rate, elite, generations, penalty_value, graph, fleet_used)
output = output_report(ga_vrp, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window)
ga_report.to_csv('VRP-04-Report.csv', encoding= 'cp949', index = False)
output.to_csv('output-01-Report.csv', encoding= 'cp949', index = False)