import numpy as np
import copy
# Function: Subroute Distance
def evaluate_distance(distance_matrix, depot, subroute, parameters):    
    subroute      = evaluate_subroute(subroute, parameters)
    subroute_i    = depot + subroute
    subroute_j    = subroute + depot
    subroute_ij   = [(subroute_i[i], subroute_j[i]) for i in range(0, len(subroute_i))]
    distance      = list(np.cumsum(distance_matrix[tuple(np.array(subroute_ij).T)]))
    distance[0:0] = [0.0]
    return distance
def evaluate_subroute(index_lst,parameters):
    subroute = []
    for i in index_lst:
        subroute.append(parameters[:,0][i])
    return subroute
# Function: Subroute Time
# 각 subroute node마다 waite, arrival, departure 시간 찍어주는 함수
def evaluate_time(distance_matrix, parameters, depot, subroute, velocity):
    subroute   = evaluate_subroute(subroute, parameters)
    tw_early   = parameters[:, 1]
    tw_late    = parameters[:, 2]
    tw_st      = parameters[:, 3]
    subroute_i = depot + subroute
    subroute_j = subroute + depot
    wait       = [0]*len(subroute_j)
    time       = [0]*len(subroute_j)
    for i in range(0, len(time)):
        time[i] = time[i] + distance_matrix[(subroute_i[i], subroute_j[i])]/velocity[0]
        if (time[i] < tw_early[subroute_j][i][0]):
            wait[i] = tw_early[subroute_j][i][0] - time[i]
            time[i] = tw_early[subroute_j][i][0]
            day_num = 0
        elif (time[i] < tw_late[subroute_j][i][0]):
            wait[i] = 0
            day_num = 0          
        elif (time[i] < tw_early[subroute_j][i][1]):
            wait[i] = tw_early[subroute_j][i][1] - time[i]
            time[i] = tw_early[subroute_j][i][1]  
            day_num = 1
        elif (time[i] < tw_late[subroute_j][i][1]):
            wait[i] = 0
            day_num = 1
        elif (time[i] < tw_early[subroute_j][i][2]):
            wait[i] = tw_early[subroute_j][i][2] - time[i]
            time[i] = tw_early[subroute_j][i][2]  
            day_num = 2
        elif (time[i] < tw_late[subroute_j][i][2]):
            wait[i] = 0
            day_num = 2
        else:
            day_num = 2  
        time[i] = time[i] + tw_st[subroute_j][i]
                
        if (i  < len(time) - 1):
            time[i+1] = time[i]

    time[0:0] = [0]
    wait[0:0] = [0]
    return wait, time, day_num

# Function: Subroute Capacity
def evaluate_capacity(parameters, depot, subroute): 
    demand    = parameters[:, 5]
    subroute_ = depot + subroute + depot
    capacity  = list(np.cumsum(demand[subroute_]))
    return capacity 

# Function: Subroute Cost
def evaluate_cost(dist, wait, parameters, depot, subroute, fixed_cost, variable_cost, time_window):
    tw_wc     = parameters[:, 4]
    subroute  = evaluate_subroute(subroute, parameters)
    subroute_ = depot + subroute + depot
    cost      = [0]*len(subroute_)
    if (time_window == 'with'):
        cost = [fixed_cost[0] + y*z if x == 0 else fixed_cost[0] + x*variable_cost[0] + y*z for x, y, z in zip(dist, wait, tw_wc[subroute_])]
    else:
        cost = [fixed_cost[0]  if x == 0 else fixed_cost[0] + x*variable_cost[0]  for x in dist]
    return cost

# Function: Subroute Cost
def evaluate_cost_penalty(dist, time, wait, cap, capacity, parameters, depot, subroute, fixed_cost, variable_cost, penalty_value, time_window, route, day_num):
    tw_late = parameters[:, 2]
    tw_st   = parameters[:, 3]
    tw_wc   = parameters[:, 4]
    subroute= evaluate_subroute(subroute, parameters)
    if (route == 'open'):
        subroute_ = depot + subroute
    else:
        subroute_ = depot + subroute + depot
    pnlt = 0
    cost = [0]*len(subroute_)
    pnlt = pnlt + sum( x > capacity for x in cap[0:len(subroute_)] )
    if(time_window == 'with'):
        pnlt = pnlt + sum(x > y + z for x, y, z in zip(time, tw_late[subroute_][day_num] , tw_st[subroute_]))  
        cost = [fixed_cost[0] + y*z if x == 0 else cost[0] + x*variable_cost[0] + y*z for x, y, z in zip(dist, wait, tw_wc[subroute_])]
    else:
        cost = [fixed_cost[0] if x == 0 else cost[0] + x*variable_cost[0] for x in dist]        
    cost[-1] = cost[-1] + pnlt*penalty_value
    return cost[-1]

# Function: Routes Nearest Depot
def evaluate_depot(n_depots, individual, real_distance_matrix):
    d_1 = float('+inf')
    for i in range(0, n_depots):
        for j in range(0, len(individual[1])):
            d_2 = evaluate_distance(real_distance_matrix, [i], individual[1][j])[-1]
            if (d_2 < d_1):
                d_1 = d_2
                individual[0][j] = [i]
    return individual


