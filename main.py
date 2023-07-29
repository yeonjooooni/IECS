import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm

from itertools import cycle
from matplotlib import pyplot as plt
from pyVRP import *
od_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (od_matrix) 수정완료.csv')

pivot_table = pd.read_csv("./pivot_table_filled.csv", encoding='cp949', index_col=[0])

demand_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv', encoding='cp949')
tmp_df = demand_df[demand_df['date']=='2023-05-04']
tmp_df = tmp_df[tmp_df['Group']==3]
tmp_df = tmp_df[tmp_df['터미널ID']=='O_9']

# orders_table 시작(터미널 id)-도착지(착지 id) symmetric한 2d matrix로 나타낸 것.
# index는 symmetric하나 value(시간)은 다름, OD matrix 이용.
# 이 부분은 0_179, 터미널에 대해서 만든 (sub) pivot table
id_list_only_in_tmp_df = list(set(tmp_df['터미널ID'].values.tolist() + tmp_df['착지ID'].values.tolist()))
pivot_table = pivot_table.loc[id_list_only_in_tmp_df,id_list_only_in_tmp_df]
pivot_table = pivot_table.sort_index(axis=1, ascending=False)
pivot_table = pivot_table.sort_index(axis=0,  ascending=False)

import numpy as np
# '착지_ID'열의 각 값의 인덱스를 담을 리스트 초기화
index_positions = [list(pivot_table.index).index("O_9")]
cbm_list = [0]

# 왜 이런식으로 index 바꾼 것?
# tmp_df의 '착지_ID'열의 각 값에 대해 pivot_table.index 리스트의 인덱스를 찾습니다.
# (시작지, 착지, CBM 형태의 주문 triplet)
for i in range(len(tmp_df)):
    value = tmp_df['착지ID'].values.tolist()[i]
    index_positions.append(list(pivot_table.index).index(value))
    cbm_list.append(float(tmp_df['CBM'].values[i]))

# origin 및 departure 모든 지점을 coordinate 형태로 넣어준 것
departure_coordinates = demand_df.drop_duplicates(['착지ID'])[['착지ID', '하차지_위도', '하차지_경도']]
departure_coordinates.columns = ['ID', 'y', 'x']
origin_coordinates = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (Terminals).csv", encoding='cp949', usecols = [0,1,2])
origin_coordinates.columns = departure_coordinates.columns
coordinates = pd.concat([departure_coordinates, origin_coordinates], ignore_index=True)
coordinates = coordinates.set_index(['ID'])
coordinates = coordinates.reindex(index=pivot_table.index)
coordinates = coordinates.loc[id_list_only_in_tmp_df].sort_index(ascending=False).reset_index(drop=True)

parameters = pd.DataFrame({
    'arrive_station': index_positions,
    'TW_early':[[1080,2520,3960] for _ in range(len(index_positions))],#18-00, 2일차(42-48:00)
    'TW_late':[[1560,3000,4320] for _ in range(len(index_positions))],
    'TW_service_time':60,
    'TW_wait_cost':0,
    'cbm':cbm_list
})
# print(parameters.head(10))
# Tranform to Numpy Array
coordinates = coordinates.values
parameters  = parameters.values
distance_matrix = pivot_table.values

# Parameters - Model
n_depots    =  1          # The First n Rows of the 'distance_matrix' or 'coordinates' are Considered as Depots
time_window = 'with'    # 'with', 'without'
route       = 'closed'     # 'open', 'closed'
model       = 'vrp'        # 'tsp', 'mtsp', 'vrp'
graph       = True        # True, False

# Parameters - Vehicle
vehicle_types = 5                           # Quantity of Vehicle Types
fixed_cost    = [ 80,110,150,200,250 ]      # Fixed Cost
variable_cost = [ 0.8,1,1.2,1.5,1.8 ]      # Variable Cost
capacity      = [ 27,33,42,51,55 ]      # Capacity of the Vehicle
velocity      = [ 1,1,1,1,1 ]      # The Average Velocity Value is Used as a Constant that Divides the Distance Matrix.
fleet_size    = [ 10,10,10,10,10 ]      # Available Vehicles

# Parameters - GA
penalty_value   = 1000000    # GA Target Function Penalty Value for Violating the Problem Constraints
population_size = 5      # GA Population Size
mutation_rate   = 0.2     # GA Mutation Rate
elite           = 1        # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained 
generations     = 10     # GA Number of Generations

# Run GA Function
ga_report, ga_vrp = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, population_size, vehicle_types, n_depots, route, model, time_window, fleet_size, mutation_rate, elite, generations, penalty_value, graph)