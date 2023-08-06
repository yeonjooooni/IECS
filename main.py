import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm
from collections import defaultdict
from itertools import cycle
from matplotlib import pyplot as plt
import copy
from pyVRP import *
from utils import *

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

def run_ga(terminal_id):
    # dict 대신 defaultdict를 사용하면, key가 없을 때 자동으로 0을 할당해줌
    fleet_size_dict = defaultdict(lambda : 0)
    fleet_size_no_fixed_cost = defaultdict(lambda : 0)
    od_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (od_matrix) 수정완료.csv')
    # pivot_table = pd.read_csv("./pivot_table_filled.csv", encoding='cp949', index_col=[0])
    real_distance_matrix = pd.read_csv("./distance_matrix.csv", index_col=0)
    real_distance_matrix = real_distance_matrix.values

    demand_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv', encoding='cp949')
    veh_table = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (veh_table).csv', encoding='cp949')
    # veh_table에 'CurrentCenter', 'CenterArriveTime', 'IsUsed' 열 추가
    # cueerntcenter : 이 veh이 도착할 center, centerarrivetime : 이 veh이 center에 도착할 시간, isused : 이 veh이 사용한 적 있는지 여부
    veh_table['CurrentCenter'] = veh_table['StartCenter']
    veh_table['CenterArriveTime'] = 0
    veh_table['IsUsed'] = 0
    
    ga_column_names = ['Route', 'Vehicle', 'Activity', 'Job_도착지점의 index', 'Arrive_Load', 'Leave_Load', 'Wait_Time', 'Arrive_Time','Leave_Time', 'Distance', 'Costs']
    total_ga_report = pd.DataFrame([], columns = ga_column_names)
    output_column_names = ['ORD_NO', 'VehicleID', 'Sequence', 'SiteCode', 'ArrivalTime', 'WaitingTime', 'ServiceTime', 'DepartureTime', 'Delivered']
    total_output_report = pd.DataFrame([], columns=output_column_names)

    # range 기본은 (0, 6)
    for day in range(1):
        for group in range(2):
            time_absolute = 1440 * day  +  360 * group

            tmp_df = demand_df[demand_df['date']==f'2023-05-0{1+day}']
            tmp_df = tmp_df[tmp_df['Group'].isin([group])]
            tmp_df = tmp_df[tmp_df['터미널ID']==terminal_id]
            
            id_list_only_in_tmp_df = list(set(tmp_df['터미널ID'].values.tolist() + tmp_df['착지ID'].values.tolist()))
            pivot_table = pd.read_csv("./pivot_table_filled.csv", encoding='cp949', index_col=[0])
            pivot_table = pivot_table.loc[id_list_only_in_tmp_df,id_list_only_in_tmp_df]
            pivot_table = pivot_table.sort_index(axis=1, ascending=False)
            pivot_table = pivot_table.sort_index(axis=0,  ascending=False)

            # '착지_ID'열의 각 값의 인덱스를 담을 리스트 초기화
            index_positions = [list(pivot_table.index).index(terminal_id)]
            cbm_list = [0]
            ORD_NO_list = []

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
                'cbm':cbm_list,
            })

            # Tranform to Numpy Array
            coordinates = coordinates.values
            parameters  = parameters.values
            distance_matrix = pivot_table.values

            # Parameters - Model
            n_depots    =  1           # The First n Rows of the 'distance_matrix' or 'coordinates' are Considered as Depots
            time_window = 'with'       # 'with', 'without'
            route       = 'closed'     # 'open', 'closed'
            model       = 'vrp'        # 'tsp', 'mtsp', 'vrp'
            graph       = True         # True, False

            # Parameters - Vehicle
            tmp_veh = veh_table[veh_table['CurrentCenter'] == terminal_id]
            vehicle_index = tmp_veh.index.to_list()
            vehicle_types = tmp_veh.shape[0] # 해당 출발지에 속한 차량 수
            fixed_cost    = tmp_veh['FixedCost'].values.tolist()
            variable_cost = tmp_veh['VariableCost'].values.tolist()
            capacity      = tmp_veh['MaxCapaCBM'].values.tolist()
            velocity      = [1] * vehicle_types # 전부 1
            fleet_size    = [1] * vehicle_types # 전부 1

            for time_duration in range(0, 28 * 360, 360):
                fleet_size_dict[time_duration] = [1]*vehicle_types
                fleet_size_no_fixed_cost[time_duration] = [0]*vehicle_types

            # Parameters - GA
            penalty_value   = 1000000    # GA Target Function Penalty Value for Violating the Problem Constraints
            population_size = 10      # GA Population Size
            mutation_rate   = 0.2     # GA Mutation Rate
            elite           = 1        # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained 
            generations     = 2     # GA Number of Generations

            # Run GA Function
            ga_report, output_report, ga_vrp, fleet_used_now = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, real_distance_matrix, population_size, vehicle_types, n_depots, route, model, time_window, fleet_size, mutation_rate, elite, generations, penalty_value, graph, 'rw', fleet_size_no_fixed_cost[time_absolute], time_absolute)
            total_ga_report = pd.concat([total_ga_report, ga_report])
            total_output_report = pd.concat([total_output_report, output_report])

            print("현재 절대 시각")
            print(time_absolute)
            print("####################################")

            #ga_report.to_csv(f"./report/{date}_{group}.csv", index=False, encoding = 'cp949')

            # 사용한 차량의 복귀 시간대 파악
            clean_report = ga_report[ga_report['Route'].str.startswith('#')]
            vehicles_within_intervals, fleet_used_now = vehicle_return_time(clean_report, fleet_used_now, vehicle_types, time_absolute)

            # 갔다 돌아와서 쓸 수 있는 차량 대수인 fleet size 없데이트
            fleet_size_dict = usable_vehicle(fleet_size_dict, fleet_used_now, vehicle_types, vehicles_within_intervals, time_absolute)

            # 다시 쓸 수 있어 고정비용을 재계산하지 않아도 되는 vehicle 계산
            fleet_size_no_fixed_cost = reusable_vehicle(clean_report, fleet_size_no_fixed_cost, vehicle_types, vehicles_within_intervals, time_absolute)
            
            # 차량의 현재 위치와 도착 시간 업데이트
            update_veh_table(veh_table, vehicle_index, vehicles_within_intervals, vehicle_types, terminal_id, time_duration)
            
    return total_ga_report, total_output_report

# 일종의 main
# terminal 지정해줘야 됨
terminal_id = 'O_179'
ga_report, output_report = run_ga(terminal_id)
ga_report.to_csv('VRP-04-Report.csv', encoding= 'cp949', index = False)
output_report.to_csv('output-01-Report.csv', encoding= 'cp949', index = False)