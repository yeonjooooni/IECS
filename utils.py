import folium
import folium.plugins
import pandas as pd
import numpy as np

from itertools import cycle
from matplotlib import pyplot as plt

import random
from collections import defaultdict

# Function: Tour Plot
def plot_tour_coordinates (coordinates, solution, n_depots, route, size_x = 10, size_y = 10):
    depot     = solution[0]
    city_tour = solution[1]  # [[3,2,6,2],[15,2,7,6]]
    cycol     = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', 
                       '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193', 
                       '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
    plt.figure(figsize = [size_x, size_y])
    for j in range(0, len(city_tour)):
        if (route == 'closed'):
            xy = np.zeros((len(city_tour[j]) + 2, 2))
        else:
            xy = np.zeros((len(city_tour[j]) + 1, 2))
        for i in range(0, xy.shape[0]):
            if (i == 0):
                xy[ i, 0] = coordinates[depot[j][i], 0]
                xy[ i, 1] = coordinates[depot[j][i], 1]
                if (route == 'closed'):
                    xy[-1, 0] = coordinates[depot[j][i], 0]
                    xy[-1, 1] = coordinates[depot[j][i], 1]
            if (i > 0 and i < len(city_tour[j])+1):
                xy[i, 0] = coordinates[city_tour[j][i-1], 0]
                xy[i, 1] = coordinates[city_tour[j][i-1], 1]
        plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 0.5, markersize = 5, color = next(cycol))
    for i in range(0, coordinates.shape[0]):
        if (i < n_depots):
            plt.plot(coordinates[i,0], coordinates[i,1], marker = 's', alpha = 1.0, markersize = 7, color = 'k')[0]
            plt.text(coordinates[i,0], coordinates[i,1], i, ha = 'center', va = 'bottom', color = 'k', fontsize = 7)
        else:
            plt.text(coordinates[i,0],  coordinates[i,1], i, ha = 'center', va = 'bottom', color = 'k', fontsize = 7)
    return

# Function: Tour Plot - Lat Long
def plot_tour_latlong (lat_long, solution, n_depots, route):
    m       = folium.Map(location = (lat_long.iloc[0][0], lat_long.iloc[0][1]), zoom_start = 14)
    clients = folium.plugins.MarkerCluster(name = 'Clients').add_to(m)
    depots  = folium.plugins.MarkerCluster(name = 'Depots').add_to(m)
    for i in range(0, lat_long.shape[0]):
        if (i < n_depots):
            folium.Marker(location = [lat_long.iloc[i][0], lat_long.iloc[i][1]], popup = '<b>Client: </b>%s</br> <b>Adress: </b>%s</br>'%(int(i), 'D'), icon = folium.Icon(color = 'black', icon = 'home')).add_to(depots)
        else:
            folium.Marker(location = [lat_long.iloc[i][0], lat_long.iloc[i][1]], popup = '<b>Client: </b>%s</br> <b>Adress: </b>%s</br>'%(int(i), 'C'), icon = folium.Icon(color = 'blue')).add_to(clients)
    depot     = solution[0]
    city_tour = solution[1]
    cycol     = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', 
                       '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193', 
                       '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
    for j in range(0, len(city_tour)):
        if (route == 'closed'):
            ltlng = np.zeros((len(city_tour[j]) + 2, 2))
        else:
            ltlng = np.zeros((len(city_tour[j]) + 1, 2))
        for i in range(0, ltlng.shape[0]):
            if (i == 0):
                ltlng[ i, 0] = lat_long.iloc[depot[j][i], 0]
                ltlng[ i, 1] = lat_long.iloc[depot[j][i], 1]
                if (route == 'closed'):
                    ltlng[-1, 0] = lat_long.iloc[depot[j][i], 0]
                    ltlng[-1, 1] = lat_long.iloc[depot[j][i], 1]
            if (i > 0 and i < len(city_tour[j])+1):
                ltlng[i, 0] = lat_long.iloc[city_tour[j][i-1], 0]
                ltlng[i, 1] = lat_long.iloc[city_tour[j][i-1], 1]
        c = next(cycol)
        for i in range(0, ltlng.shape[0]-1):
          locations = [ (ltlng[i,0], ltlng[i,1]), (ltlng[i+1,0], ltlng[i+1,1])]
          folium.PolyLine(locations , color = c, weight = 1.5, opacity = 1).add_to(m)
    return m

def preprocess_demand_df():
    demand_df = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv', encoding='cp949')
    # 3일간의 하차 가능 시작과 끝 시간 리스트 계산
    landing_start_times = []
    landing_end_times = []
    for idx, row in demand_df.iterrows():
        start_time = row['하차가능시간_시작']
        end_time = row['하차가능시간_종료']
        day = int(row['date'][-1])-1
        group = row['Group']
        start_list, end_list = get_trip_time_lists(start_time, end_time, day, group, num_days=3)
        landing_start_times.append(start_list)
        landing_end_times.append(end_list)
    demand_df['landing_start_times'] = landing_start_times
    demand_df['landing_end_times'] = landing_end_times 
    return demand_df

def update_landing_available_time_zone(unassigned_rows):
    def update_times(row):
        landing_start_times = row['landing_start_times']
        landing_end_times = row['landing_end_times']
        
        updated_start_times = [max(time - 360, 0) for time in landing_start_times]
        updated_end_times = [max(time - 360, 0) for time in landing_end_times]
        
        row['landing_start_times'] = updated_start_times
        row['landing_end_times'] = updated_end_times
        return row
    
    return unassigned_rows.apply(update_times, axis=1)

def preprocess_coordinates(demand_df, pivot_table, id_list_only_in_tmp_df):
    departure_coordinates = demand_df.drop_duplicates(['착지ID'])[['착지ID', '하차지_위도', '하차지_경도']]
    departure_coordinates.columns = ['ID', 'y', 'x']
    origin_coordinates = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (Terminals).csv", encoding='cp949', usecols = [0,1,2])
    origin_coordinates.columns = departure_coordinates.columns
    coordinates = pd.concat([departure_coordinates, origin_coordinates], ignore_index=True)
    coordinates = coordinates.set_index(['ID'])
    coordinates = coordinates.reindex(index=pivot_table.index)
    coordinates = coordinates.loc[id_list_only_in_tmp_df].sort_index(ascending=False).reset_index(drop=True)
    return coordinates

# 차량 대수 확인을 위한 함수
def get_checked_fleet_cnt(vehicles_within_intervals):
    checked_fleet_cnt = 0
    for i in vehicles_within_intervals.values():
        for j in i:
            checked_fleet_cnt += j
    return checked_fleet_cnt

def vehicle_return_time(clean_report, fleet_used_now, vehicle_types):
    return_time = [0 for _ in range(vehicle_types)]

    for route, group in clean_report.groupby(['Route']):
        last_row = group.iloc[-1]
        vehicle_type = last_row['Vehicle']
        return_time[vehicle_type] = last_row['Arrive_Time']

    return return_time

def min_to_day(minute):
    #minute으로 받은 거 해당 날짜로 바꿔주는 format
    minute = int(round(minute, 0))
    hr = minute // 60
    minute = str(minute % 60)
    day = "2023-05-0{}".format(1+hr//24)
    hr = str(hr % 24)
    return day+" "+hr.zfill(2)+":"+minute.zfill(2)

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

def update_veh_table(veh_table, vehicle_index, return_time, vehicle_types, terminal_id):
    for idx in range(vehicle_types):
        if return_time[idx] != 0:
            veh_table.loc[vehicle_index[idx], 'CenterArriveTime'] = return_time[idx]
            veh_table.loc[vehicle_index[idx], 'IsUsed'] = 1

# terminal별로 각 terminal 까지의 (거리,시간,도착터미널)을 value로 만들고, 거리 기준 정렬해주는 함수
def time_distance_in_order(time_matrix, distance_matrix, terminals):
    asc_dist_dict = {}
    for T1 in terminals:
        for T2 in terminals:
            if T1!=T2:
                if T1 in asc_dist_dict.keys():
                    asc_dist_dict[T1].append((distance_matrix[T1][T2],time_matrix[T1][T2],T2))
                else:
                    asc_dist_dict[T1] = [(distance_matrix[T1][T2],time_matrix[T1][T2],T2)]
    for terminal in terminals:
        asc_dist_dict[terminal].sort(key = lambda x: x[0])
    return asc_dist_dict 

# total_dict: terminal ID를 key로 받으며 value로는 해당 시간대의 해당 terminal의 fleet_size, fleet_size_used, fleet_used_now, veh_interval, veh_table에서의 idx를 반환하는 dictionary
def get_total_dict(veh_table):
    total_dict = defaultdict(lambda : [[None],[None],[None]])
    for center in veh_table['StartCenter'].unique():
        center_data = veh_table[veh_table['CurrentCenter'] == center]
        # 해당 센터 차량의 출발 가능 여부
        fleet_available = [0 if x > 0 else 1 for x in center_data['CenterArriveTime']]
        # 출발 가능 차량 중 이전에 사용해서 고정비가 발생하지 않는 차량의 수
        fleet_available_no_fixed_cost = center_data['IsUsed'].tolist()    
        # 해당 센터의 보유 차량들의 인덱스 리스트
        fleet_idx = center_data.index.tolist()
        total_dict[center] = [fleet_available, fleet_available_no_fixed_cost, fleet_idx]
    return total_dict

# 현재까지 사용한 차 수, 각 터미널별 현재 차 수(veh_table), 터미널 별 가장 가까운 터미널들, 부족한 차 수(==미처리된 주문수)
def reallocate_veh(max_car, veh_table, asc_dist_dict, unassigned_orders, terminals, total_dict):
    # 터미널별 필요 차량 수 확인(==미처리된 주문 수)
    for terminal in terminals:
        if unassigned_orders[terminal]:
            car_taken = 0
            # 가까운 터미널 부터 돌면서 가져올 수 있는 차량의 수 확인
            for dist, time, arrival_terminal in asc_dist_dict[terminal]:
                if not unassigned_orders[arrival_terminal]:
                    available_cars = max(0, len(veh_table[(veh_table["CurrentCenter"] == arrival_terminal) & (veh_table["CenterArriveTime"] == 0)]) - max_car[arrival_terminal])
                    # 가져올 차량이 있는 경우 해당 수가 필요 차량보다 많으면 break, 아니면 더하고 continue
                    if available_cars:
                        # available_cars가 현재 필요한 차량보다 많아서 random하게 뽑는 경우
                        if available_cars>=unassigned_orders[terminal]-car_taken:
                            cur_car_taken = unassigned_orders[terminal]-car_taken
                            # 현재 터미널에 있는 차량들의 idx lst 생성
                            lst = [i for i in range(len(total_dict[arrival_terminal][0])) if total_dict[arrival_terminal][0][i] != 0]
                            #car_idx = random.sample(lst, cur_car_taken)
                            if cur_car_taken <= len(lst):
                                car_idx = random.sample(lst, cur_car_taken)
                            else:
                                car_idx = lst
                            print("########차량 이동########")
                            print("car_idx", car_idx)
                            print("from", arrival_terminal, "to", terminal)
                            print("#########################")
                            # 각 터미널의 차량 증감 처리 + 비용처리도 필요함! -> history를 만드는게 좋을듯
                            for idx in car_idx:
                                veh_table.loc[total_dict[arrival_terminal][2][idx], 'CurrentCenter'] = terminal
                                veh_table.loc[total_dict[arrival_terminal][2][idx], 'CenterArriveTime'] = time
                                veh_table.loc[total_dict[arrival_terminal][2][idx], 'IsUsed'] = 1 # 일단 사용한거로 처리. 고정비 + 가변비로 비교..?

                            # total_dict update
                            for i in range(3):
                                for j in car_idx:
                                    total_dict[terminal][i].append(total_dict[arrival_terminal][i][j])
                                total_dict[arrival_terminal][i] = [total_dict[arrival_terminal][i][k] for k in range(len(total_dict[arrival_terminal][i])) if k not in car_idx]
                            #car_taken = unassigned_orders[terminal]
                            break

                        else:
                            # available_cars를 모두 가져오는 경우
                            cur_car_taken = available_cars
                            # 모든 차량을 가져오기 때문에 그냥 총 idx lst생성
                            car_idx = [i for i in range(len(total_dict[arrival_terminal][0])) if total_dict[arrival_terminal][0][i] != 0]
                            car_idx = random.sample(car_idx, available_cars)
                            print("########차량 이동########")
                            print("car_idx", car_idx)
                            print("from", arrival_terminal, "to", terminal)
                            print("#########################")
                            # 각 터미널의 차량 증감 처리 + 비용처리도 필요함! -> history를 만드는게 좋을듯
                            for idx in car_idx:
                                veh_table.loc[total_dict[arrival_terminal][2][idx], 'CurrentCenter'] = terminal
                                veh_table.loc[total_dict[arrival_terminal][2][idx], 'CenterArriveTime'] = time
                                veh_table.loc[total_dict[arrival_terminal][2][idx], 'IsUsed'] = 1
                            # total_dict update
                            for i in range(3):
                                for j in car_idx:
                                    total_dict[terminal][i].append(total_dict[arrival_terminal][i][j])
                                total_dict[arrival_terminal][i] = [total_dict[arrival_terminal][i][k] for k in range(len(total_dict[arrival_terminal][i])) if k not in car_idx]
                            car_taken += cur_car_taken
                            if car_taken == unassigned_orders[terminal]:
                                break
                            continue

def check_max_car(terminal, max_car, fleet_used_now):
    max_car[terminal] = max(max_car[terminal], sum(fleet_used_now))
    return max_car

def set_max_car(terminals):
    max_car = {}
    for terminal in terminals:
        max_car.update({terminal:5})
    return max_car
