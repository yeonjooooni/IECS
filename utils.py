import folium
import folium.plugins
import pandas as pd
import numpy as np

from itertools import cycle
from matplotlib import pyplot as plt

# 차량 대수 확인을 위한 함수
def get_checked_fleet_cnt(vehicles_within_intervals):
    checked_fleet_cnt = 0
    for i in vehicles_within_intervals.values():
        for j in i:
            checked_fleet_cnt += j
    return checked_fleet_cnt

def vehicle_return_time(clean_report, fleet_used_now, vehicle_types, time_absolute):
    # 사용한 차량의 복귀 시간대 파악
    vehicles_within_intervals = {}

    time_block = 6 * 60
    while get_checked_fleet_cnt(vehicles_within_intervals) < sum(fleet_used_now):
        vehicles_within_intervals_for_one_block = [0]*vehicle_types

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
    return vehicles_within_intervals, fleet_used_now

    # 갔다 돌아와서 쓸 수 있는 차량 대수인 fleet size 없데이트
    ## 일단 전부 빼고
def usable_vehicle(fleet_size_dict, fleet_used_now, vehicle_types, vehicles_within_intervals, time_absolute):
    for time_duration in fleet_size_dict:
        if time_duration >= 360 + time_absolute:
            fleet_size_dict[time_duration] = [size - used for size, used in zip(fleet_size_dict[time_duration], fleet_used_now)]
    ## 돌아온거 더함
    for return_time, vehicles_dict in vehicles_within_intervals.items(): # time criteria: 차량이 돌아오는 시간들
        for time_in_terminal in fleet_size_dict:
            if time_in_terminal >= return_time:
                for vehicle_type in range(vehicle_types):
                    count = vehicles_dict[vehicle_type]
                    fleet_size_dict[time_in_terminal][vehicle_type] += count

    print("사용 가능한 차량 현황")
    print(fleet_size_dict)
    print("####################################")
    return fleet_size_dict

def reusable_vehicle(clean_report, fleet_size_no_fixed_cost, vehicle_types, vehicles_within_intervals, time_absolute):
    fleet_size_no_fixed_cost_now = [0]*vehicle_types

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
                for vehicle_type in range(vehicle_types):
                    count = vehicles_dict[vehicle_type]
                    fleet_size_no_fixed_cost[time_in_terminal][vehicle_type] = min(fleet_size_no_fixed_cost[time_in_terminal][vehicle_type] + count, 1)

    print("사용 가능한 고정비 없는 차량 현황")
    print(fleet_size_no_fixed_cost)
    print("####################################")
    return fleet_size_no_fixed_cost

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
    plt.savefig('line_visualization.png')
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

def update_veh_table(veh_table, vehicle_index, vehicles_within_intervals, vehicle_types, terminal_id, time_duration):
    for time_duration in vehicles_within_intervals.keys():
        for idx in range(vehicle_types):
            if vehicles_within_intervals[time_duration][idx] == 1:
                veh_table.loc[vehicle_index[idx], 'CurrentCenter'] = terminal_id
                veh_table.loc[vehicle_index[idx], 'CenterArriveTime'] = time_duration
                veh_table.loc[vehicle_index[idx], 'IsUsed'] = 1
            # print(veh_table.loc[vehicle_index[idx]]["VehNum"])