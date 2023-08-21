import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm
from utils import *
from report import *
from eval import *

############################################################################
# Function: Routes Best Vehicle
def evaluate_vehicle(vehicle_types, individual, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route,real_distance_matrix, fleet_available, fleet_available_no_fixed_cost = None):
    cost, _     = target_function([individual], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route,real_distance_matrix, fleet_available, fleet_available_no_fixed_cost) 
    individual_ = copy.deepcopy(individual)
    for i in range(0, len(individual[0])):
        for j in range(0, vehicle_types):
            individual_[2][i] = [j]
            cost_, _             = target_function([individual_], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, real_distance_matrix, fleet_available, fleet_available_no_fixed_cost) 
            if (cost_ < cost):
                cost             = cost_
                individual[2][i] = [j]     
                individual_      = copy.deepcopy(individual)
            else:
                individual_      = copy.deepcopy(individual)
    return individual

# Function: Route Evalution & Correction
# 각 population의 cost만 계산
def target_function(population, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, real_distance_matrix, fleet_available = [], fleet_available_no_fixed_cost=None):
    cost     = [[0] for i in range(len(population))]
    tw_late  = parameters[:, 2]
    tw_st    = parameters[:, 3]
    flt_cnt  = [0]*len(fleet_available)
    if (route == 'open'):
        end = 2 
    else:
        end = 1

    # k individuals, individual은 각 차들의 route
    # individual[i] = [depot(출발), route(list type), vehicle]
    # 거리*비용 + penalty
    for k in range(0, len(population)): # k individuals
        individual = copy.deepcopy(population[k])  
        size       = len(individual[1])
        i          = 0
        pnlt       = 0
        flt_cnt    = [0]*len(fleet_available)
        no_fixed_cost_count = [0]*len(fleet_available)
        while (size > i): # i subroutes 
            dist = evaluate_distance(real_distance_matrix, individual[0][i], individual[1][i], parameters)
            if(time_window == 'with'):
                wait, time, day_num = evaluate_time(distance_matrix, parameters, depot = individual[0][i], subroute = individual[1][i], velocity = [velocity[individual[2][i][0]]])
            else:
                wait       = []
                time       = []
            cap    = evaluate_capacity(parameters, depot = individual[0][i], subroute = individual[1][i])
                  
            pnlt   = pnlt + sum( x >  capacity[individual[2][i][0]] for x in cap[0:-1] )
            if(time_window == 'with'):
                if (route == 'open'):
                    subroute_ = individual[0][i] + individual[1][i]
                else:
                    subroute_ = individual[0][i] + individual[1][i] + individual[0][i]
                pnlt = pnlt + sum(x > y + z for x, y, z in zip(time, tw_late[subroute_][day_num] , tw_st[subroute_]))                      
            if (len(fleet_available) > 0):
                flt_cnt[individual[2][i][0]] = flt_cnt[individual[2][i][0]] + 1 
            if (size <= i + 1):
                for v in range(0, len(fleet_available)):
                    v_sum = flt_cnt[v] - fleet_available[v]
                    if (v_sum > 0):
                        pnlt = pnlt + v_sum #차량 대수 조절

            flag = True
            # 특정 차 종류의 사용 댓수가
            if fleet_available_no_fixed_cost[individual[2][i][0]] > no_fixed_cost_count[individual[2][i][0]]:
                flag = False
                no_fixed_cost_count[individual[2][i][0]] += 1
            if flag:
                cost_s = evaluate_cost(dist, wait, parameters, depot = individual[0][i], subroute = individual[1][i], fixed_cost = [fixed_cost[individual[2][i][0]]], variable_cost = [variable_cost[individual[2][i][0]]], time_window = time_window)
            else:
                cost_s = evaluate_cost(dist, wait, parameters, depot = individual[0][i], subroute = individual[1][i], fixed_cost = [0],                                 variable_cost = [variable_cost[individual[2][i][0]]], time_window = time_window)
            cost[k][0] = cost[k][0] + cost_s[-end] + pnlt*penalty_value

            size       = len(individual[1])
            i          = i + 1

    cost_total = copy.deepcopy(cost)
    return cost_total, population

# Function: Initial Population
# CBM만 넘지 않게 일단일단 차량 배정
def initial_population(parameters, coordinates='none', distance_matrix='none', population_size=5, vehicle_types=1, n_depots=1, model='vrp', capacity = [20,30,40,40,50], fleet_available = None):#[0]*len(glb_fleet_available_no_fixed_cost)):

    # Exclude clients with demand equal to 0
    # 1, 2, 3, ..., 한 터미널의 날짜의 그룹의 주문의 개수
    non_zero_demand_clients = [i for i in range(1, len(parameters[:,0]))] #[client for client in range(n_depots, distance_matrix.shape[0]) if coordinates[client][0] != 0]
    depots = [[i] for i in range(n_depots)]
    vehicles = [[i] for i in range(vehicle_types) if fleet_available[i]>0]
    
    print(f"처리해야하는 물량 수: {len(non_zero_demand_clients)}")
    print(f"현재 소속 차: {len(fleet_available)}")
    print(f"현재 가용 차: {sum(fleet_available)}")

    total_demand_unassigned = [] # 다음으로 넘길 주문

    population = []
    flag = True
    for i in range(population_size):
        if flag:
            clients_temp = copy.deepcopy(non_zero_demand_clients)
            
            #route는 출발지, route_depot: 도착지
            routes = []
            routes_depot = []
            routes_vehicles = []
            fleet_available_check = copy.deepcopy(fleet_available)
            
            repeat_count = 0
            while len(clients_temp) > 0:
                repeat_count += 1
                if repeat_count >= 1000:
                    if clients_temp:
                        total_demand_unassigned.append(clients_temp.pop())
                        repeat_count = 0
                    continue

                if not vehicles:
                    total_demand_unassigned.extend(clients_temp)
                    break

                e = random.sample(vehicles, 1)[0]

                if fleet_available_check != [0 for _ in range(len(fleet_available_check))]:
                    while fleet_available_check[e[0]]<=0:
                        e = random.sample(vehicles,1)[0]
                else:
                    total_demand_unassigned.extend(clients_temp)
                    break

                d = random.sample(depots, 1)[0]
                c = random.sample(clients_temp, random.randint(1, min(len(clients_temp), 2))) #차량 최대적재량/최대주문크기
                # 차량 적재량 넘으면 다시 돌리기
                if sum([parameters[:, 5][int(i)] for i in c]) > capacity[int(e[0])]:
                    continue

                clients_temp = [item for item in clients_temp if item not in c]

                tmp = []
                # 차량 배정
                # 주문 배정도 되도록 수정
                for idx in c:                
                    #tmp.append(int(parameters[:,0][int(idx)]))
                    tmp.append(int(idx))
                
                routes_vehicles.append(e)
                routes_depot.append(d)
                routes.append(tmp)
                fleet_available_check[e[0]]-=1

            if total_demand_unassigned:
                print(f"처리 못하고 다음으로 넘긴 물량: {len(total_demand_unassigned)}")
            
            non_zero_demand_clients = [client for client in non_zero_demand_clients if client not in total_demand_unassigned]
            first_individual = [routes_depot, routes, routes_vehicles, total_demand_unassigned]
            population.append([routes_depot, routes, routes_vehicles, total_demand_unassigned])
            flag = False
        else:    
            clients_temp = copy.deepcopy(non_zero_demand_clients)
            
            #route는 출발지, route_depot: 도착지
            routes = []
            routes_depot = []
            routes_vehicles = []
            fleet_available_check = copy.deepcopy(fleet_available)
            
            repeat_count = 0

            flag_first_is_best = False
            while len(clients_temp) > 0:
                repeat_count += 1
                if repeat_count >= 1000:
                    repeat_count = 0
                    flag_first_is_best = True
                    break

                if not vehicles:
                    flag_first_is_best = True
                    break

                e = random.sample(vehicles, 1)[0]
                if fleet_available_check != [0 for _ in range(len(fleet_available_check))]:
                    while fleet_available_check[e[0]]<=0:
                        e = random.sample(vehicles,1)[0]
                    
                else:
                    flag_first_is_best = True
                    break

                d = random.sample(depots, 1)[0]
                c = random.sample(clients_temp, random.randint(1, min(len(clients_temp), 3))) #차량 최대적재량/최대주문크기
                # 차량 적재량 넘으면 다시 돌리기
                if sum([parameters[:, 5][int(i)] for i in c]) > capacity[int(e[0])]:
                    continue

                clients_temp = [item for item in clients_temp if item not in c]

                tmp = []
                # 차량 배정
                # 주문 배정도 되도록 수정
                for idx in c:                
                    tmp.append(int(idx))
                
                routes_vehicles.append(e)
                routes_depot.append(d)
                routes.append(tmp)
                fleet_available_check[e[0]]-=1
            if flag_first_is_best:
                population.append(first_individual)
            else:
                population.append([routes_depot, routes, routes_vehicles, total_demand_unassigned])

    return population

# Function: Fitness
def fitness_function(cost, population_size, parameters): 
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1 + cost[i][0] + abs(np.min(cost)))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: VRP Crossover - BRBAX (Best Route Better Adjustment Recombination)
def crossover_vrp_brbax(parent_1, parent_2):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    subroute  = [ parent_1[0][s], parent_1[1][s], parent_1[2][s] ]
    offspring = copy.deepcopy(parent_2)
    
    for k in range(len(parent_2[1])-1, -1, -1):
        offspring[1][k] = [item for item in offspring[1][k] if item not in subroute[1] ] 
        if (len(offspring[1][k]) == 0):
            del offspring[0][k]
            del offspring[1][k]
            del offspring[2][k]
    offspring[0].append(subroute[0])
    offspring[1].append(subroute[1])
    offspring[2].append(subroute[2])
    #print("offspring _ crossover_vrp_brbax", offspring)
    return offspring

# Function: VRP Crossover - BCR (Best Cost Route Crossover)
def crossover_vrp_bcr(parent_1, parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route, real_distance_matrix):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    offspring = copy.deepcopy(parent_2)
    if (len(parent_1[1][s]) > 1):
        cut  = random.sample(list(range(0,len(parent_1[1][s]))), 2)
        gene = 2
    else:
        cut  = [0, 0]
        gene = 1
    for i in range(0, gene):
        d_1   = float('+inf')
        ins_m = 0
        A     = parent_1[1][s][cut[i]]
        best  = []
        for m in range(0, len(parent_2[1])):
            parent_2[1][m] = [item for item in parent_2[1][m] if item not in [A] ]
            if (len(parent_2[1][m]) > 0):
                insertion      = copy.deepcopy([ parent_2[0][m], parent_2[1][m], parent_2[2][m] ])
                dist_list      = [evaluate_distance(real_distance_matrix, insertion[0], insertion[1][:n] + [A] + insertion[1][n:], parameters) for n in range(0, len(parent_2[1][m]) + 1)]
                if(time_window == 'with'):
                    wait_time_list = [evaluate_time(distance_matrix, parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:], velocity = [velocity[parent_2[2][m][0]] ] ) for n in range(0, len(parent_2[1][m]) + 1)]
                else:
                    wait_time_list = [[0, 0]]*len(dist_list)
                day_num_list   = [evaluate_time(distance_matrix, parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:], velocity = [velocity[parent_2[2][m][0]] ] )[2] for n in range(0, len(parent_2[1][m]) + 1)]
                cap_list       = [evaluate_capacity(parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][m]) + 1)]
                insertion_list = [insertion[1][:n] + [A] + insertion[1][n:] for n in range(0, len(parent_2[1][m]) + 1)]
                d_2_list       = [evaluate_cost_penalty(dist_list[n], wait_time_list[n][1], wait_time_list[n][0], cap_list[n], capacity[parent_2[2][m][0]], parameters, insertion[0], insertion_list[n], [fixed_cost[parent_2[2][m][0]]], [variable_cost[parent_2[2][m][0]]], penalty_value, time_window, route, day_num_list[n]) for n in range(0, len(dist_list))]
                d_2 = min(d_2_list)
                if (d_2 <= d_1):
                    d_1   = d_2
                    ins_m = m
                    best  = insertion_list[d_2_list.index(min(d_2_list))]
        parent_2[1][ins_m] = best            
        if (d_1 != float('+inf')):
            offspring = copy.deepcopy(parent_2)
    for i in range(len(offspring[1])-1, -1, -1):
        if(len(offspring[1][i]) == 0):
            del offspring[0][i]
            del offspring[1][i]
            del offspring[2][i]
    #print("offspring _ bcr", offspring)
    return offspring

# breeding function이 제약조건을 만족하는지 check, 제약 조건은 initial population과 동일
def check_individual_capacity(individual, parameters, capacity):
    for route_num in range(len(individual[1])):
        if sum([parameters[:, 5][int(i)] for i in individual[1][route_num]]) > capacity[int(individual[2][route_num][0])] * 3:
            print("check_individual_capacity 통과 실패")
            return False
    return True

# Function: Breeding
def breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route, vehicle_types, fleet_available, real_distance_matrix, fleet_available_no_fixed_cost):
    offspring = copy.deepcopy(population) 
    if (elite > 0):
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        for i in range(0, elite):
            offspring[i] = copy.deepcopy(population[i])

    for i in range (elite, len(offspring)):
        # 좋은 조건의 부모 둘 뽑아서, 섞음(어떻게 섞는지는 일단 out of mind)
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = copy.deepcopy(population[parent_1])  
        parent_2 = copy.deepcopy(population[parent_2])
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
        if((len(parent_1[1]) > 1 and len(parent_2[1]) > 1)):
            if (rand > 0.5):
                offspring[i] = crossover_vrp_brbax(parent_1, parent_2)
                offspring[i] = crossover_vrp_bcr(offspring[i], parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route, real_distance_matrix=real_distance_matrix)              
            elif (rand <= 0.5): 
                offspring[i] = crossover_vrp_brbax(parent_2, parent_1)
                offspring[i] = crossover_vrp_bcr(offspring[i], parent_1, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route, real_distance_matrix=real_distance_matrix)

    return offspring

# Function: Mutation - Swap
def mutation_tsp_vrp_swap(individual):
    if len(individual[1]) == 0:
        return individual
    elif (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]  
    cut1                    = random.sample(list(range(0, len(individual[1][k1]))), 1)[0]
    cut2                    = random.sample(list(range(0, len(individual[1][k2]))), 1)[0]
    A                       = individual[1][k1][cut1]
    B                       = individual[1][k2][cut2]
    individual[1][k1][cut1] = B
    individual[1][k2][cut2] = A
    return individual

# Function: Mutation - Insertion
def mutation_tsp_vrp_insertion(individual):
    if len(individual[1]) == 0:
        return individual
    elif (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]
    cut1 = random.sample(list(range(0, len(individual[1][k1])))  , 1)[0]
    cut2 = random.sample(list(range(0, len(individual[1][k2])+1)), 1)[0]
    A    = individual[1][k1][cut1]
    del individual[1][k1][cut1]
    individual[1][k2][cut2:cut2] = [A]
    if (len(individual[1][k1]) == 0):
        del individual[0][k1]
        del individual[1][k1]
        del individual[2][k1]
    return individual

# Function: Mutation
def mutation(offspring, mutation_rate, elite):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability <= mutation_rate):
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand <= 0.5):
                offspring[i] = mutation_tsp_vrp_insertion(offspring[i])
            elif(rand > 0.5):
                offspring[i] = mutation_tsp_vrp_swap(offspring[i])
        for k in range(0, len(offspring[i][1])):
            if (len(offspring[i][1][k]) >= 2):
                probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                if (probability <= mutation_rate):
                    rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    cut  = random.sample(list(range(0, len(offspring[i][1][k]))), 2)
                    cut.sort()
                    C    = offspring[i][1][k][cut[0]:cut[1]+1]
                    if (rand <= 0.5):
                        random.shuffle(C)
                    elif(rand > 0.5):
                        C.reverse()
                    offspring[i][1][k][cut[0]:cut[1]+1] = C
    return offspring

# Function: Elite Distance
def elite_distance(individual, distance_matrix, route, parameters):
    if (route == 'open'):
        end = 2
    else:
        end = 1
    td = 0
    for n in range(0, len(individual[1])):
        td = td + evaluate_distance(distance_matrix, depot = individual[0][n], subroute = individual[1][n], parameters=parameters)[-end]
    return round(td,2)

# GA-VRP Function
def genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, real_distance_matrix, population_size = 5, vehicle_types = 1, n_depots = 1, route = 'closed', model = 'vrp', time_window = 'without', fleet_available = [], mutation_rate = 0.1, elite = 0, generations = 50, penalty_value = 1000, graph = True, selection = 'rw', fleet_available_no_fixed_cost=None, time_absolute=0, order_id=[], city_name_list=[], vehicle_index = []):    
    start           = tm.time()
    count           = 0
    solution_report = ['None']
    max_capacity    = copy.deepcopy(capacity)
    population       = initial_population(parameters, coordinates, distance_matrix, population_size = population_size, vehicle_types = vehicle_types, n_depots = n_depots, model = model, capacity = capacity, fleet_available=fleet_available)   
    cost, population = target_function(population, distance_matrix, parameters, velocity, fixed_cost, variable_cost, max_capacity, penalty_value, time_window = time_window, route = route, fleet_available = fleet_available, real_distance_matrix=real_distance_matrix,fleet_available_no_fixed_cost=fleet_available_no_fixed_cost) 
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    # 기존 코드, finess_function을 통해 각 경우의 population에 점수 배정
    if (selection == 'rw'):
        fitness          = fitness_function(cost, population_size, parameters)
    else: #elif (selection == 'rb'):
        rank             = [[i] for i in range(1, len(cost)+1)]
        fitness          = fitness_function(rank, population_size, parameters)
    elite_ind        = elite_distance(population[0], real_distance_matrix, route = route, parameters=parameters)
    cost             = copy.deepcopy(cost)
    elite_cst        = copy.deepcopy(cost[0][0])
    solution         = copy.deepcopy(population[0])

    while (count <= generations-1):
        offspring        = breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity, max_capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route, vehicle_types, fleet_available,real_distance_matrix, fleet_available_no_fixed_cost=fleet_available_no_fixed_cost)          
        offspring        = mutation(offspring, mutation_rate = mutation_rate, elite = elite)
        cost, population = target_function(offspring, distance_matrix, parameters, velocity, fixed_cost, variable_cost, max_capacity, penalty_value, time_window = time_window, route = route, fleet_available = fleet_available,real_distance_matrix=real_distance_matrix, fleet_available_no_fixed_cost=fleet_available_no_fixed_cost)
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        elite_child      = elite_distance(population[0], real_distance_matrix, route = route, parameters=parameters)    #elite는 그대로
        if (selection == 'rw'):
            fitness = fitness_function(cost, population_size, parameters)
        elif (selection == 'rb'):
            rank    = [[i] for i in range(1, len(cost)+1)]
            fitness = fitness_function(rank, population_size, parameters)
        if(elite_ind > elite_child):    #cost가 제일 낮은 한 개는 각 iter마다 일단 뽑아둠
            elite_ind = elite_child 
            solution  = copy.deepcopy(population[0])
            elite_cst = copy.deepcopy(cost[0][0])
        count = count + 1  
    
    fleet_used_now = [0] * len(fleet_available_no_fixed_cost)
    # 해당 시간대에 사용한 차량
    for sublist in solution[2]:
        value = sublist[0]
        fleet_used_now[value] += 1

    output = output_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window, time_absolute, order_id=order_id, city_name_list=city_name_list, vehicle_index = vehicle_index)
    solution_report = show_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window, real_distance_matrix, fleet_available_no_fixed_cost, time_absolute, city_name_list=city_name_list, vehicle_index = vehicle_index)

    end = tm.time()
    print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
    print('Algorithm Time: ', round((end - start), 2), ' seconds')
    return solution_report, output, solution, fleet_used_now

   ############################################################################
