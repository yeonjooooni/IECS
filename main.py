import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm
import numpy as np
from itertools import cycle
from matplotlib import pyplot as plt
import copy
from pyVRP import *

# 시간을 분 단위로 변환하는 함수
def time_to_minutes(time_str):
    hour, minute = map(int, time_str.split(':'))
    return hour * 60 + minute

# 3일간의 하차 가능 시작과 끝 시간 리스트를 구하는 함수
# 여기서 이미 time_window와 무관하게 3일차(4320분)에 딱 cut하도록 만들어 놓음
def get_trip_time_lists(start_time, end_time, day, group, num_days=3): #수정필요
    start_time_minutes = time_to_minutes(start_time)
    end_time_minutes = time_to_minutes(end_time)

    if start_time_minutes > end_time_minutes:
        end_time_minutes += 1440

    start_time_minutes -= 1440 * day  +  360 * group
    end_time_minutes   -= 1440 * day  +  360 * group

    while start_time_minutes < 0:
        start_time_minutes += 1440
        end_time_minutes   += 1440

    start_list = []
    end_list   = []

    for day in range(num_days):
        if start_time_minutes + day * 24 * 60 > 4320 - max(0, day - 4) * 1440:
            start_list.append((4320 - max(0, day - 4) * 1440) - 360 * group - 1440 * (day-4) if day>=4 else (4320 - max(0, day - 4) * 1440))
        else:
            start_list.append((start_time_minutes + day * 24 * 60)- 360 * group - 1440 * (day-4) if day>=4 else (start_time_minutes + day * 24 * 60))

        if end_time_minutes + day * 24 * 60 > 4320 - max(0, day - 4) * 1440:
            end_list.append((4320 - max(0, day - 4) * 1440) - 360 * group - 1440 * (day-4) if day>=4 else (4320 - max(0, day - 4) * 1440))
        else:
            end_list.append((end_time_minutes + day * 24 * 60)- 360 * group - 1440 * (day-4) if day>=4 else (end_time_minutes + day * 24 * 60))

    return start_list, end_list


# 차량 대수 확인을 위한 함수
def get_checked_fleet_cnt(vehicles_within_intervals):
    checked_fleet_cnt = 0
    for i in vehicles_within_intervals.values():
        for j in i:
            checked_fleet_cnt += j
    return checked_fleet_cnt

fleet_size_dict = {}
fleet_size_no_fixed_cost = {}
for time_duration in range(0, 28 * 360, 360):
    fleet_size_dict[time_duration] = [10, 10, 10, 10, 10]
    fleet_size_no_fixed_cost[time_duration] = [0, 0, 0, 0, 0]
     
for day in range(6):
    for group in range(4):
        time_absolute = 1440 * day  +  360 * group

        od_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (od_matrix) 수정완료.csv')
        pivot_table = pd.read_csv("./pivot_table_filled.csv", encoding='cp949', index_col=[0])
        real_distance_matrix = pd.read_csv("./distance_matrix.csv", index_col=0)
        demand_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv', encoding='cp949')

        tmp_df = demand_df[demand_df['date']==f'2023-05-0{1+day}']
        tmp_df = tmp_df[tmp_df['Group'].isin([group])]
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

        # 3일간의 하차 가능 시작과 끝 시간 리스트 계산
        trip_start_times = [[0,0,0]]
        trip_end_times = [[(4320 - max(0, day - 4) * 1440) for _ in range(3)]]

        for idx, row in tmp_df.iterrows():
            start_time = row['하차가능시간_시작']
            end_time = row['하차가능시간_종료']
            start_list, end_list = get_trip_time_lists(start_time, end_time, day, group, num_days=3)
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
        n_depots    =  1           # The First n Rows of the 'distance_matrix' or 'coordinates' are Considered as Depots
        time_window = 'with'       # 'with', 'without'
        route       = 'closed'     # 'open', 'closed'
        model       = 'vrp'        # 'tsp', 'mtsp', 'vrp'
        graph       = True         # True, False

        # Parameters - Vehicle
        vehicle_types = 5                                   # Quantity of Vehicle Types
        fixed_cost    = [ 80,110,150,200,250 ]              # Fixed Cost
        variable_cost = [ 0.8,1,1.2,1.5,1.8 ]               # Variable Cost
        capacity      = [ 27,33,42,51,55 ]                  # Capacity of the Vehicle
        velocity      = [ 1,1,1,1,1 ]                       # The Average Velocity Value is Used as a Constant that Divides the Distance Matrix.
        fleet_size    = fleet_size_dict[time_absolute]      # Available Vehicles

        # Parameters - GA
        penalty_value   = 1000000    # GA Target Function Penalty Value for Violating the Problem Constraints
        population_size = 10         # GA Population Size
        mutation_rate   = 0.2        # GA Mutation Rate
        elite           = 1          # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained 
        generations     = 10         # GA Number of Generations

        # Run GA Function
        ga_report, ga_vrp, fleet_used_now = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, real_distance_matrix, population_size, vehicle_types, n_depots, route, model, time_window, fleet_size, mutation_rate, elite, generations, penalty_value, graph, 'rw', fleet_size_no_fixed_cost[time_absolute])

        print("현재 절대 시각")
        print(time_absolute)
        print("####################################")

        #ga_report.to_csv(f"./report/{date}_{group}.csv", index=False, encoding = 'cp949')

        # 사용한 차량의 복귀 시간대 파악
        clean_report = ga_report[ga_report['Route'].str.startswith('#')]

        vehicles_within_intervals = {}

        time_block = 6 * 60
        while get_checked_fleet_cnt(vehicles_within_intervals) < sum(fleet_used_now):
            vehicles_within_intervals_for_one_block = [0]*5

            for route, group in clean_report.groupby(['Route']):
                last_row = group.iloc[-1]
                if time_block - 6 * 60 < last_row['Arrive_Time'] <= time_block:
                    vehicle_type = last_row['Vehicle']
                    vehicles_within_intervals_for_one_block[vehicle_type] += 1

            vehicles_within_intervals[time_block + time_absolute] = vehicles_within_intervals_for_one_block
            time_block += 6 * 60

        print("사용한 차량의 복귀 시간대")
        print(vehicles_within_intervals, fleet_used_now)
        print("####################################")

        # 갔다 돌아와서 쓸 수 있는 차량 대수인 fleet size 없데이트
        ## 일단 전부 빼고
        for time_duration in fleet_size_dict:
            if time_duration >= 360 + time_absolute:
                fleet_size_dict[time_duration] = [size - used for size, used in zip(fleet_size_dict[time_duration], fleet_used_now)]
        ## 돌아온거 더함
        for return_time, vehicles_dict in vehicles_within_intervals.items(): # time criteria: 차량이 돌아오는 시간들
            for time_in_terminal in fleet_size_dict:
                if time_in_terminal >= return_time:
                    for vehicle_type in range(5):
                        count = vehicles_dict[vehicle_type]
                        fleet_size_dict[time_in_terminal][vehicle_type] += count

        print("사용 가능한 차량 현황")
        print(fleet_size_dict)
        print("####################################")

        # 쓸 수 있는 고정비 없는 차량 대수인 fleet_size_no_fixed_cost 없데이트
        ## 일단 이번에 사용한 고정비 없는 차량 대수 구하고
        clean_report = ga_report[ga_report['Route'].str.startswith('#')]

        fleet_size_no_fixed_cost_now = [0,0,0,0,0]

        for route, group in clean_report.groupby(['Route']):
            first_row = group.iloc[0]
            if first_row['Costs'] == 0:
                fleet_size_no_fixed_cost_now[first_row['Vehicle']] += 1

        ## 일단 빼고
        for time_duration in fleet_size_no_fixed_cost:
            if time_duration >= 360 + time_absolute:
                fleet_size_no_fixed_cost[time_duration] = [size - used for size, used in zip(fleet_size_no_fixed_cost[time_duration], fleet_size_no_fixed_cost_now)]
        ## 돌아온거 더함
        for return_time, vehicles_dict in vehicles_within_intervals.items():
            for time_in_terminal in fleet_size_no_fixed_cost:
                if time_in_terminal >= return_time:
                    for vehicle_type in range(5):
                        count = vehicles_dict[vehicle_type]
                        fleet_size_no_fixed_cost[time_in_terminal][vehicle_type] = min(fleet_size_no_fixed_cost[time_in_terminal][vehicle_type] + count, 10)

        print("사용 가능한 고정비 없는 차량 현황")
        print(fleet_size_no_fixed_cost)
        print("####################################")
        print()
