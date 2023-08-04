from pyVRP import *
def output_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window):
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
        elif (route == 'open'):
            subroute = [solution[0][i] + solution[1][i]]
        
        for j in range(0, len(subroute[0])):
            if (j == 0):
                activity = 'start'
                arrive_time = round(time[j], 2)
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j], 2)
            if (j > 0 and j < len(subroute[0]) - 1):
                activity = 'service'  
            if (j == len(subroute[0]) - 1):
                activity = 'finish'
                if (time[j] > tt):
                    tt = time[j]
                td = td + dist[j]
                tc = tc + cost[j]
            
            # Prepare data for the Delivered column
            delivered_status = 'Null' if activity == 'start' else 'Yes' if activity == 'finish' else 'No'
            
            report_lst.append([solution[2][i][0], 'VEH_' + str(solution[2][i][0]), j+1, subroute[0][j], arrive_time, round(wait[j], 2), round(time[j], 2) if activity != 'start' else 'Null', round(time[j], 2) if activity == 'finish' else 'Null', delivered_status])
        
        report_lst.append(['-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-'])
    
    report_df = pd.DataFrame(report_lst, columns=column_names)
    return report_df