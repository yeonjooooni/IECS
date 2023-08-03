#필요 Library import
import pandas as pd
import numpy as np
import time as tm
from itertools import cycle
from matplotlib import pyplot as plt
from pyVRP import *

demand_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv', encoding='cp949')

pivot_table = pd.read_csv("./pivot_table_filled.csv", encoding='cp949', index_col=[0])

real_distance_matrix = pd.read_csv("./distance_matrix.csv", index_col=0)


def preprocess(terminal_ID, day, group_num): #날짜와 그룹을 주면 필요 df뽑아오기 
    terminal_demand = demand_df[demand_df['터미널ID']==terminal_ID]
    target_df = terminal_demand[terminal_demand['date']==f'2023-05-0{2+day}']
    target_df = target_df[target_df['Group'].isin([group_num])]

    #필요한 node들의 matrix만 가져오기
    id_list = list(set(target_df['터미널ID'].values.tolist() + target_df['착지ID'].values.tolist()))
    pivot_table = pivot_table.loc[id_list]
    pivot_table = pivot_table.sort_index(axis=1, ascending=False)
    pivot_table = pivot_table.sort_index(axis=0, ascending=False)

    # 각 물건 적재용량 list로 만들기
    index_positions = [list(pivot_table.index).index(terminal_ID)]
    cbm_list = [0]
    
    
    for i in range(len(target_df)):
        value = target_df.iloc[i]['착지ID']
        index_positions.append(list(pivot_table.index).index(value))
        cbm_list.append(float(target_df.iloc[i]['CBM']))

    return target_df, pivot_table, index_positions, cbm_list


#좌표 설정 함수
def set_coordinates(pivot_table, id_list,df=demand_df):
    departure_coordinates = df.drop_duplicates(['착지ID'])[['착지ID', '하차지_위도', '하차지_경도']]
    departure_coordinates.columns = ['ID','y','x']
    
    origin_coordinates = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (Terminals).csv", encoding='cp949', usecols = [0,1,2])
    origin_coordinates.columns = departure_coordinates.columns
    coordinates = pd.concat([departure_coordinates, origin_coordinates], ignore_index=True)
    coordinates = coordinates.set_index(['ID'])
    coordinates = coordinates.reindex(index=pivot_table.index)
    coordinates = coordinates.loc[id_list].sort_index(ascending=False).reset_index(drop=True)

    return coordinates


def time_to_minutes(time_str):
    hour,minute = map(int, time_str.split(':'))
    return hour*60 + minute

def get_trip_time_lists(start_time, end_time, day, group, num_days=3)
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
        if start_time_minutes + day * 24 * 60 > 4320:
            start_list.append(4320)
        else:
            start_list.append((start_time_minutes + day * 24 * 60))
        
        if end_time_minutes + day * 24 * 60 > 4320:
            end_list.append(4320)
        else:
            end_list.append((end_time_minutes + day * 24 * 60))
    
    return start_list, end_list

def make_parameters(terminal_ID, day, group_num):

    target_df, pivot_table, index_positions, cbm_list = preprocess(terminal_ID, day, group_num)

    coordinates = set_coordinates(pivot_table, index_positions)

    trip_start_times = [[0,0,0]]
    trip_end_times = [[4320,4320,4320]]

    for idx, row in target_df.itterrows():
        start_time = row['하차가능시간_시작']
        end_time = row['하차가능시간_종료']

        start_list, end_list = get_trip_time_lists(start_time, end_time, day, group_num)

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
    return parameters, coordinates.values, pivot_table.values

for day in range(7):
    for group in range(4):
        parameters, coordinates, distance_matrix = make_parameters("O_179",day,group)
