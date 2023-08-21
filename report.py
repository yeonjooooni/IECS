import pandas as pd
import numpy as np
import copy
import os
from matplotlib import pyplot as plt
from eval import *
from utils import *
from datetime import datetime, timedelta
def output_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window, time_absolute, order_id, city_name_list, vehicle_index):
    column_names = ['ORD_NO', 'VehicleID', 'Sequence', 'SiteCode', 'ArrivalTime', 'WaitingTime', 'ServiceTime', 'DepartureTime', 'Delivered']
    tt = 0
    td = 0 
    tc = 0
    tw_st = parameters[:, 3]
    report_lst = []
    
    for i in range(0, len(solution[1])):
        reversed_sol = copy.deepcopy(solution[1][i])
        reversed_sol.reverse()
        cap          = evaluate_capacity(parameters, solution[0][i], reversed_sol) 
        cap.reverse()
        
        dist = evaluate_distance(distance_matrix, solution[0][i], solution[1][i],parameters)
        wait, time = evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i], velocity = [velocity[solution[2][i][0]]])[0:2]

        leave_cap = copy.deepcopy(cap)
        for n in range(1, len(leave_cap)-1):
            leave_cap[n] = cap[n+1] 
        cost = evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i], fixed_cost = [fixed_cost[solution[2][i][0]]], variable_cost = [variable_cost[solution[2][i][0]]], time_window = time_window)
        
        #solution[1][i]  = evaluate_subroute(solution[1][i],parameters)
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
                ORD_NO = 'Null'
                if (time[j] > tt):
                    tt = time[j]
                td = td + dist[j]
                tc = tc + cost[j]
                #continue
            city_name = city_name_list[subroute[0][j]]
                        
            # # Prepare data for the Delivered column
            # # 우리는 아직 service가 마무리 안된 건수가 없음. 그리고 finish는 출력할 필요 없음
            #activity = finish, 우리가 보려고 표시한 return한 차량
        
            report_lst.append([ORD_NO, 'VEH_' + str(vehicle_index[solution[2][i][0]]+2), j+1, city_name,  # 2 더함
                               min_to_day(arrive_time+time_absolute), round(wait[j], 2), 60 if activity != 'start' else None, min_to_day(round(time[j], 2)+time_absolute) if activity == 'service' else None, delivered_status])
        
        report_lst.append(['-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-'])
    
    report_df = pd.DataFrame(report_lst, columns=column_names)
    
    return report_df

def show_report(solution, distance_matrix,  parameters, velocity, fixed_cost, variable_cost, route, time_window, real_distance_matrix, fleet_used, time_absolute, city_name_list, vehicle_index):
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
        
        #solution[1][i] = evaluate_subroute(solution[1][i],parameters)
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
            report_lst.append(['#' + str(i+1), 'VEH_' + str(vehicle_index[solution[2][i][0]]+2), activity, city_name_list[subroute[0][j]], cap[j], leave_cap[j], round(wait[j],2), arrive_time+time_absolute, round(time[j],2)+time_absolute, round(dist[j],2), round(cost[j],2) ])
        report_lst.append(['-//-', '-//-', '-//-', '-//-','-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-'])
    report_lst.append(['MAX TIME', '', '','', '', '', '', '', round(tt,2), '', ''])
    report_lst.append(['TOTAL', '', '','', '', '', '', '', '', round(td,2), round(tc,2)])
    report_df = pd.DataFrame(report_lst, columns = column_names)
    return report_df

def vehicle_output_report(output_report):   # this output_report must include temp delivered state
    vehicle_table = pd.read_csv('./과제3 실시간 주문 대응 Routing 최적화 (veh_table).csv', encoding='cp949')
    orders_table = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv", encoding='cp949')
    distance_table = pd.read_csv("./distance_matrix.csv", index_col=0)
    pivot_table_filled = pd.read_csv("./pivot_table_filled.csv", index_col=0)
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
        vehicle_cnt[vehicle_table.iloc[i]['VehNum']] = 0
        vehicle_volume[vehicle_table.iloc[i]['VehNum']] = 0
        vehicle_traveldistance[vehicle_table.iloc[i]['VehNum']] = 0
        vehicle_worktime[vehicle_table.iloc[i]['VehNum']] = 0 #solution에서 받아오기
        vehicle_traveltime[vehicle_table.iloc[i]['VehNum']] = 0
        vehicle_servicetime[vehicle_table.iloc[i]['VehNum']] = 0
        vehicle_wait_time[vehicle_table.iloc[i]['VehNum']] = 0
        vehicle_volume[vehicle_table.iloc[i]['VehNum']] = 0

    for key, value in vehicle_cnt.items():
        fixedCost = vehicle_table[vehicle_table['VehNum']==key]["FixedCost"].values[0]
        varCost = vehicle_table[vehicle_table['VehNum']==key]["VariableCost"].values[0]
        #for total_distance
        # total report에서 한 차량을 이용하는 것에 대한 모든 기록
        VehID_table = output_report[output_report['VehicleID'] == key]
        # 차량 사용하지 않은 경우 -1
        last_end_time = -1
        for i in range(len(VehID_table.index[:-1])): #마지막에 복귀하는것 제외
            if VehID_table.iloc[i]['Delivered'] == 'Yes':
                vehicle_cnt[key] += 1
                vehicle_wait_time[key] += float(VehID_table.iloc[i]['WaitingTime']) #waiting time 누적
                vehicle_volume[key] += float(orders_table[orders_table['주문ID']==VehID_table.iloc[i]["ORD_NO"]]["CBM"].values[0])
                vehicle_worktime[key] += (day_to_min(VehID_table.iloc[i]['ArrivalTime']) - last_end_time)
                vehicle_worktime[key] += (day_to_min(VehID_table.iloc[i]['DepartureTime']) - day_to_min(VehID_table.iloc[i]['ArrivalTime']))
                last_end_time = day_to_min(VehID_table.iloc[i]['DepartureTime'])

            elif VehID_table.iloc[i]['Delivered'] == "temp":
                vehicle_worktime[key] += (day_to_min(VehID_table.iloc[i]['ArrivalTime']) - last_end_time)
                last_end_time = day_to_min(VehID_table.iloc[i]['ArrivalTime'])    

            else: #상차
                # reallocate로 인한 이동 시간 고려
                if last_end_time != -1 and VehID_table.iloc[i]['SiteCode'] != VehID_table.iloc[i+1]['SiteCode']:
                    vehicle_worktime[key] += pivot_table_filled.loc[VehID_table.iloc[i]['SiteCode'], VehID_table.iloc[i+1]['SiteCode']]
                last_end_time = day_to_min(VehID_table.iloc[i]['ArrivalTime'])
            if i == len(VehID_table.index)-2:  #마지막 vehicle_traveldistance 제외
                continue
            if VehID_table.iloc[i]['SiteCode'] == VehID_table.iloc[i+1]['SiteCode']:
                continue             
            vehicle_traveldistance[key] += distance_table.loc[VehID_table.iloc[i]['SiteCode'], VehID_table.iloc[i+1]['SiteCode']]
        total_distance = vehicle_traveldistance[key]
        vehicle_servicetime[key] = vehicle_cnt[key] * 60
        vehicle_traveltime[key] = vehicle_worktime[key] - vehicle_servicetime[key] - vehicle_wait_time[key]
        if vehicle_cnt[key] !=0 :
            totalCost = varCost*total_distance + fixedCost
        else:
            totalCost = 0
            fixedCost = 0
        # 'VehicleID', 'Count', 'Volume', 'TravelDistance', 'WorkTime', 'TravelTime', 'ServiceTime', 'WaitingTime', 'TotalCost', 'FixedCost',	'VariableCost'
        report_lst.append([key, vehicle_cnt[key], vehicle_volume[key], vehicle_traveldistance[key], vehicle_worktime[key], vehicle_traveltime[key], vehicle_servicetime[key], vehicle_wait_time[key], totalCost, fixedCost, varCost*total_distance])
    report_df = pd.DataFrame(report_lst, columns=column_names)
    return report_df

def get_submission_file_1(df, day, group, number_of_t, FOLDER_PATH, demand_df, start_day): 
    today = start_day + timedelta(days = day)
    today = today.strftime('%Y-%m-%d')
    df = df[df['ORD_NO'] != '-//-']
    df = df.sort_values(by=['VehicleID', 'ArrivalTime'])
    df = df.reset_index(drop=True)

    groups = df.groupby('VehicleID')
    for group_name, group_df in groups:
        temp_rows = group_df[group_df['Delivered'] == 'temp'].index
        for idx in temp_rows:
            if group_df.loc[idx, 'Delivered'] == 'temp':
                start_time = pd.to_datetime(group_df.loc[idx, 'ArrivalTime'])
                next_idx = idx + 1
                if next_idx < len(group_df) and pd.notna(group_df.loc[next_idx, 'ArrivalTime']):
                    end_time = pd.to_datetime(group_df.loc[next_idx, 'ArrivalTime'])
                    time_difference = (end_time - start_time).total_seconds() / 60
                    df.loc[next_idx, 'WaitingTime'] = time_difference
                    df.loc[next_idx, 'ArrivalTime'] = start_time
                    df.loc[next_idx, 'DepartureTime'] = end_time
                df.drop(idx, inplace=True)
    df = df.reset_index(drop=True)

    grouped = df.groupby('VehicleID')
    for group_name, group_data in grouped:
        df.loc[group_data.index, 'Sequence'] = range(1, len(group_data) + 1)
    df = df.reset_index(drop=True)

    df.loc[((df['ArrivalTime'].notnull()) & (df['ServiceTime'].isnull())), 'ServiceTime'] = 0
    df.loc[((df['ArrivalTime'].notnull()) & (df['DepartureTime'].isnull())), 'DepartureTime'] = df['ArrivalTime']

    df['ArrivalTime_datetime'] = pd.to_datetime(df['ArrivalTime'])
    df['ArrivalTime_ElapsedMinutes'] = (df['ArrivalTime_datetime'] - pd.to_datetime(f'{start_day} 00:00')).dt.total_seconds() / 60
    condition_arr = df['ArrivalTime_ElapsedMinutes'] > 1440 * day  +  360 * (group//number_of_t)
    df['DepartureTime_datetime'] = pd.to_datetime(df['DepartureTime'])
    df['DepartureTime_ElapsedMinutes'] = (df['DepartureTime_datetime'] - pd.to_datetime(f'{start_day} 00:00')).dt.total_seconds() / 60
    condition_dep = df['DepartureTime_ElapsedMinutes'] > 1440 * day  +  360 * (group//number_of_t)

    df.loc[(condition_arr)&(condition_dep), ['ArrivalTime', 'WaitingTime', 'ServiceTime', 'DepartureTime']] = 'Null'
    df.loc[(~condition_arr)&(condition_dep), ['WaitingTime', 'ServiceTime', 'DepartureTime']] = 'Null'
    df.loc[condition_dep & (df['Delivered'] == 'Yes'), 'Delivered'] = 'No'
    df.loc[((df['ArrivalTime'].notnull()) & (df['ServiceTime'].isnull())), 'ServiceTime'] = 0
    df.loc[((df['ArrivalTime'].notnull()) & (df['DepartureTime'].isnull())), 'DepartureTime'] = df['ArrivalTime']
    df.drop(['ArrivalTime_datetime', 'ArrivalTime_ElapsedMinutes', 'DepartureTime_datetime', 'DepartureTime_ElapsedMinutes'], axis = 1, inplace=True)
        
    tmp_df = demand_df[demand_df['date'] == today]
    tmp_df = tmp_df[tmp_df['Group'].isin([group//number_of_t])]

    total_order_ID = tmp_df['주문ID'].unique().tolist()
    assigned_order = df['ORD_NO'].unique().tolist()
    unassigned_order_ID = []
    for id in total_order_ID:
        if id not in assigned_order:
            unassigned_order_ID.append(id)
    unassigned_order_ID=pd.DataFrame(unassigned_order_ID, columns=['ORD_NO'])
    df = pd.concat([df, unassigned_order_ID], axis=0)
    df.fillna("Null", inplace=True)

    df.to_csv(f"{FOLDER_PATH}/제출파일1_최종/total_output_report_day_{day}_group_{group // number_of_t}.csv", index=False, encoding='cp949')