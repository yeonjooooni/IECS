import folium
import folium.plugins
import pandas as pd
import numpy as np
import copy

from itertools import cycle
from matplotlib import pyplot as plt
from eval import *
from utils import *
def output_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window, time_absolute):
    column_names = ['ORD_NO', 'VehicleID', 'Sequence', 'SiteCode', 'ArrivalTime', 'WaitingTime', 'ServiceTime', 'DepartureTime', 'Delivered']
    tt = 0
    td = 0 
    tc = 0
    tw_st = parameters[:, 3]
    report_lst = []
    
    # Create the Delivered table at specified times
    delivered_table = pd.DataFrame()
    
    for i in range(0, len(solution[1])):
        dist = evaluate_distance(distance_matrix, solution[0][i], solution[1][i])
        wait, time = evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i], velocity = [velocity[solution[2][i][0]]])[0:2]
        reversed_sol = copy.deepcopy(solution[1][i])
        reversed_sol.reverse()
        cap = evaluate_capacity(parameters, solution[0][i], reversed_sol) 
        cap.reverse()
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
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j], 2)
            if (j > 0 and j < len(subroute[0]) - 1):
                activity = 'service'  
                delivered_status = 'Yes'
            if (j == len(subroute[0]) - 1):
                activity = 'finish'
                delivered_status = "temp"
                if (time[j] > tt):
                    tt = time[j]
                td = td + dist[j]
                tc = tc + cost[j]
                continue
            
            # # Prepare data for the Delivered column
            # # 우리는 아직 service가 마무리 안된 건수가 없음. 그리고 finish는 출력할 필요 없음
            #activity = finish, 우리가 보려고 표시한 return한 차량
            report_lst.append([solution[2][i][0], 'VEH_' + str(solution[2][i][0]), j+1, subroute[0][j], 
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
        dist         = evaluate_distance(real_distance_matrix, solution[0][i], solution[1][i])
        wait, time   = evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i], velocity = [velocity[solution[2][i][0]]])[0:2]
        reversed_sol = copy.deepcopy(solution[1][i])
        reversed_sol.reverse()
        cap          = evaluate_capacity(parameters, solution[0][i], reversed_sol) 
        cap.reverse()
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