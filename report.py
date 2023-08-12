import folium
import folium.plugins
import pandas as pd
import numpy as np
import copy

from itertools import cycle
from matplotlib import pyplot as plt
from eval import *
from utils import *
def output_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window, time_absolute, order_id, city_name_list):
    column_names = ['ORD_NO', 'VehicleID', 'Sequence', 'SiteCode', 'ArrivalTime', 'WaitingTime', 'ServiceTime', 'DepartureTime', 'Delivered']
    tt = 0
    td = 0 
    tc = 0
    tw_st = parameters[:, 3]
    report_lst = []
    # Create the Delivered table at specified times
    delivered_table = pd.DataFrame()
    for i in range(0, len(solution[1])):
        reversed_sol = copy.deepcopy(solution[1][i])
        reversed_sol.reverse()
        cap          = evaluate_capacity(parameters, solution[0][i], reversed_sol) 
        cap.reverse()
        #solution[1][i]  = evaluate_subroute(solution[1][i],parameters)
        dist = evaluate_distance(distance_matrix, solution[0][i], solution[1][i],parameters)
        wait, time = evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i], velocity = [velocity[solution[2][i][0]]])[0:2]

        leave_cap = copy.deepcopy(cap)
        for n in range(1, len(leave_cap)-1):
            leave_cap[n] = cap[n+1] 
        cost = evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i], fixed_cost = [fixed_cost[solution[2][i][0]]], variable_cost = [variable_cost[solution[2][i][0]]], time_window = time_window)
        if (route == 'closed'):
            subroute = [solution[0][i] + solution[1][i] + solution[0][i]]
        else: #elif (route == 'open'):
            subroute = [solution[0][i] + solution[1][i]]
        for j in range(0, len(subroute[0])):
            if (j == 0):
                activity = 'start'
                arrive_time = round(time[j], 2)
                delivered_status = 'Null'
                ORD_NO = 'Null'
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j], 2)
            if (j > 0 and j < len(subroute[0]) - 1):
                activity = 'service'  
                delivered_status = 'Yes'
                ORD_NO = order_id[solution[1][i][j-1]]
            if (j == len(subroute[0]) - 1):
                activity = 'finish'
                delivered_status = "temp"
                ORD_NO = None
                if (time[j] > tt):
                    tt = time[j]
                td = td + dist[j]
                tc = tc + cost[j]
                #continue
            city_name = city_name_list[subroute[0][j]]
                        
            # # Prepare data for the Delivered column
            # # 우리는 아직 service가 마무리 안된 건수가 없음. 그리고 finish는 출력할 필요 없음
            #activity = finish, 우리가 보려고 표시한 return한 차량
        
            report_lst.append([ORD_NO, 'VEH_' + str(solution[2][i][0]), j+1, city_name, 
                               min_to_day(arrive_time+time_absolute), round(wait[j], 2)+time_absolute, round(time[j], 2)+time_absolute if activity != 'start' else 'Null', min_to_day(round(time[j], 2)+time_absolute) if activity == 'service' else 'Null', delivered_status])
        
        report_lst.append(['-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-'])
    
    report_df = pd.DataFrame(report_lst, columns=column_names)
    
    return report_df

def show_report(solution, distance_matrix,  parameters, velocity, fixed_cost, variable_cost, route, time_window, real_distance_matrix, fleet_used, time_absolute):
    column_names = ['Route', 'Vehicle', 'Activity', 'Job_도착지점의 index', 'Arrive_Load', 'Leave_Load', 'Wait_Time', 'Arrive_Time','Leave_Time', 'Distance', 'Costs']
    tt           = 0
    td           = 0 
    tc           = 0
    tw_st        = parameters[:, 3]
    report_lst   = []
    no_fixed_cost_count = [0]*len(fleet_used)
    
    for i in range(0, len(solution[1])):
        reversed_sol = copy.deepcopy(solution[1][i])
        reversed_sol.reverse()
        cap          = evaluate_capacity(parameters, solution[0][i], reversed_sol) 
        cap.reverse()
        solution[1][i]  = evaluate_subroute(solution[1][i],parameters)
        dist         = evaluate_distance(real_distance_matrix, solution[0][i], solution[1][i], parameters)
        wait, time   = evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i], velocity = [velocity[solution[2][i][0]]])[0:2]
        
        leave_cap = copy.deepcopy(cap)
        for n in range(1, len(leave_cap)-1):
            leave_cap[n] = cap[n+1] 

        flag = True
        if fleet_used[solution[2][i][0]] > no_fixed_cost_count[solution[2][i][0]]:
            flag = False
            no_fixed_cost_count[solution[2][i][0]] += 1

        if flag:
            cost = evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i], fixed_cost = [fixed_cost[solution[2][i][0]]], variable_cost = [variable_cost[solution[2][i][0]]], time_window = time_window)
        else:
            cost = evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i], fixed_cost = [0],                               variable_cost = [variable_cost[solution[2][i][0]]], time_window = time_window)

        if (route == 'closed'):
            subroute = [solution[0][i] + solution[1][i] + solution[0][i] ]
        elif (route == 'open'):
            subroute = [solution[0][i] + solution[1][i] ]
        for j in range(0, len(subroute[0])):
            if (j == 0):
                activity    = 'start'
                arrive_time = round(time[j],2)
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j],2)
            if (j > 0 and j < len(subroute[0]) - 1):
                activity = 'service'  
            if (j == len(subroute[0]) - 1):
                activity = 'finish'
                if (time[j] > tt):
                    tt = time[j]
                td = td + dist[j]
                tc = tc + cost[j]
            report_lst.append(['#' + str(i+1), solution[2][i][0], activity, subroute[0][j], cap[j], leave_cap[j], round(wait[j],2), arrive_time, round(time[j],2), round(dist[j],2), round(cost[j],2) ])
        report_lst.append(['-//-', '-//-', '-//-', '-//-','-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-'])
    report_lst.append(['MAX TIME', '', '','', '', '', '', '', round(tt,2), '', ''])
    report_lst.append(['TOTAL', '', '','', '', '', '', '', '', round(td,2), round(tc,2)])
    report_df = pd.DataFrame(report_lst, columns = column_names)
    return report_df

def vehicle_output_report(output_report):
    vehicle_table = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (veh_table).csv', encoding='cp949')
    od_table = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (od_matrix) 수정완료.csv", encoding='cp949')
    orders_table = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv", encoding='cp949')
    distance_table = pd.read_csv("./distance_matrix.csv", index_col=0)
    column_names = ['VehicleID', 'Count', 'Volume', 'TravelDistance', 'WorkTime', 'TravelTime', 'ServiceTime', 'WaitingTime', 'TotalCost', 'FixedCost',	'VariableCost']
    vehicle_cnt = {}
    vehicle_volume = {}
    vehicle_traveldistance = {}
    vehicle_worktime = {}
    vehicle_servicetime ={} 
    vehicle_traveltime = {}
    vehicle_wait_time = {}

    report_lst = []

    # 초기화
    for i in range(len(vehicle_table)):
        vehicle_cnt[vehicle_table.loc[i]['VehNum']] = 0
        vehicle_volume[vehicle_table.loc[i]['VehNum']] = 0
        vehicle_traveldistance[vehicle_table.loc[i]['VehNum']] = 0
        vehicle_worktime[vehicle_table.loc[i]['VehNum']] = 0 #solution에서 받아오기
        vehicle_traveltime[vehicle_table.loc[i]['VehNum']] = 0
        vehicle_servicetime[vehicle_table.loc[i]['VehNum']] = 0
        vehicle_wait_time[vehicle_table.loc[i]['VehNum']] = 0
        vehicle_volume[vehicle_table.loc[i]['VehNum']] = 0

    for i in range(len(output_report)):
        # 지금 vehicleID 안맞아서 임시로 넣어놓은 if문
        if output_report.loc[i]['VehicleID'] not in vehicle_cnt.keys():
            print("error VehicleID not in Vehicle table")
            continue
        if output_report.loc[i]['Delivered'] == 'Yes':
            vehicle_cnt[output_report.loc[i]['VehicleID']] += 1
            vehicle_wait_time[output_report.loc[i]['VehicleID']] += float(output_report.loc[i]['WaitingTime']) #waiting time 누적
            vehicle_volume[output_report.loc[i]['VehicleID']] += float(orders_table[orders_table['주문ID']==output_report.loc[i]["ORD_NO"]]["CBM"].values[0])
        elif output_report.loc[i]['Delivered'] == "temp":
            vehicle_worktime[output_report.loc[i]['VehicleID']] += float(output_report.loc[i]['ServiceTime']) - 60
        elif output_report.loc[i]['Delivered'] == '-//-':
            continue
    #상차지에서 주문 pickup 아무 조건 없음

    for key, value in vehicle_cnt.items():
        fixedCost = vehicle_table[vehicle_table['VehNum']==key]["FixedCost"].values[0]
        varCost = vehicle_table[vehicle_table['VehNum']==key]["VariableCost"].values[0]
        #for total_distance
        # total report에서 한 차량을 이용하는 것에 대한 모든 기록
        VehID_table = output_report[output_report['VehicleID'] == key]
        for i in range(len(VehID_table.index[:-1])):
            if VehID_table.iloc[i]['SiteCode'] == VehID_table.iloc[i+1]['SiteCode']:
                continue
            vehicle_traveldistance[key] += distance_table.loc[VehID_table.iloc[i]['SiteCode'], VehID_table.iloc[i+1]['SiteCode']]
        total_distance = vehicle_traveldistance[key]
        vehicle_servicetime[key] = vehicle_cnt[key] * 60
        vehicle_traveltime[key] = vehicle_worktime[key] - vehicle_servicetime[key] - vehicle_wait_time[key]
        totalCost = varCost*total_distance + fixedCost
        # 'VehicleID', 'Count', 'Volume', 'TravelDistance', 'WorkTime', 'TravelTime', 'ServiceTime', 'WaitingTime', 'TotalCost', 'FixedCost',	'VariableCost'
        report_lst.append([key, vehicle_cnt[key], vehicle_volume[key], vehicle_traveldistance[key], vehicle_worktime[key], vehicle_traveltime[key], vehicle_servicetime[key], vehicle_wait_time[key], totalCost, fixedCost, varCost*total_distance])
    report_df = pd.DataFrame(report_lst, columns=column_names)
    return report_df
