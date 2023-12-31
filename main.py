import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm
from collections import defaultdict
from datetime import datetime, timedelta
import copy
from pyVRP import *
from utils import *

def run_ga(terminal_id, day, group, demand_df):
    global unassigned_orders_count_dict, unassigned_rows_dict, veh_table, unassigned_orders_forever, total_ga_report, total_output_report, future_rows_dict, start_day, total_days
    whole = False
    if group%number_of_t==0:
        whole = True

    real_distance_matrix = pd.read_csv("./distance_matrix.csv", index_col=0)

    if not whole:
        try:
            if unassigned_rows_dict[terminal_id]==None and future_rows_dict[terminal_id]==None:
                return None, None, [0], 0
        except:
            tmp_df = pd.concat([unassigned_rows_dict[terminal_id], future_rows_dict[terminal_id]], axis = 0)
    else:
        today = start_day + timedelta(days = day)
        today = today.strftime('%Y-%m-%d')
        print("today", today)
        tmp_df = demand_df[demand_df['date'] == today]
        tmp_df = tmp_df[tmp_df['Group'].isin([group//number_of_t])]
        tmp_df = tmp_df[tmp_df['터미널ID']==terminal_id]
        tmp_df = pd.concat([unassigned_rows_dict[terminal_id], future_rows_dict[terminal_id], tmp_df], axis=0)

    if day != total_days - 1:
        # 하차가능시작시간이 6시간 이내가 아닌 주문 미루기
        future_rows = tmp_df[~(tmp_df['landing_end_times'].apply(lambda x: any(v != 0 for v in x))) |     # landing_end_times의 값 중 0이 아닌 값이 없는 행
                            ~(tmp_df['landing_start_times'].apply(lambda x: any(v < 360 for v in x)))]  # landing_start_times의 값 중 360 미만인 값이 없는 행 
        
        if not future_rows.empty:
            #print("이전_future_rows_landing_start_times", future_rows['landing_start_times'].values.tolist())
            future_rows = future_rows.apply(lambda row: update_times(row, number_of_t), axis=1)
            print("하차가능시작시간이 6시간 이후인 주문 수", len(future_rows))
            #print("이후_future_rows_landing_start_times", future_rows['landing_start_times'].values.tolist())
            future_rows_dict.update({terminal_id: future_rows})
            tmp_df = tmp_df.drop(future_rows.index)
        else:
            future_rows_dict.update({terminal_id: None})
    else:
        future_rows_dict[terminal_id]=None

    if tmp_df.empty:
        return None, None, [0], 0
    order_id = [None] + tmp_df['주문ID'].tolist() # 주문ID order_id
    city_name_list = [terminal_id] + tmp_df['착지ID'].tolist()
    service_time_list = [0] + tmp_df['하차작업시간(분)'].tolist()
    
    id_list_only_in_tmp_df = list(set(tmp_df['터미널ID'].values.tolist() + tmp_df['착지ID'].values.tolist() + [terminal_id]))
    pivot_table = pd.read_csv("./pivot_table_filled.csv", encoding='cp949', index_col=[0])
    pivot_table = pivot_table.loc[id_list_only_in_tmp_df,id_list_only_in_tmp_df]
    pivot_table = pivot_table.sort_index(axis=1, ascending=False)
    pivot_table = pivot_table.sort_index(axis=0, ascending=False)

    coordinates = preprocess_coordinates(demand_df, pivot_table, id_list_only_in_tmp_df)

    # '착지_ID'열의 각 값의 인덱스를 담을 리스트 초기화
    index_positions = [list(pivot_table.index).index(terminal_id)]
    cbm_list = [0]

    # tmp_df의 '착지_ID'열의 각 값에 대해 pivot_table.index 리스트의 인덱스 찾기
    for i in range(len(tmp_df)):
        value = tmp_df['착지ID'].values.tolist()[i]
        index_positions.append(list(pivot_table.index).index(value))
        cbm_list.append(float(tmp_df['CBM'].values[i]))
    #print("tmp_df", tmp_df['하차가능시간_시작'].values.tolist())
    # 3일간의 하차 가능 시작과 끝 시간 리스트 계산
    landing_start_times = [[0,0,0]]
    landing_start_times.extend(tmp_df['landing_start_times'].tolist())
    #print("landing_start_times", landing_start_times)
    landing_end_times = [[(4320) for _ in range(3)]]
    landing_end_times.extend(tmp_df['landing_end_times'].tolist())
    #print("landing_end_times", landing_end_times)
    # print("index_positions", index_positions)
    tw_service_time = [0]
    tw_service_time.extend(tmp_df['하차작업시간(분)'].tolist())
    parameters = pd.DataFrame({
        'arrive_station': index_positions,
        'TW_early': landing_start_times,
        'TW_late': landing_end_times,
        'TW_service_time': tw_service_time,
        'TW_wait_cost':0,
        'cbm':cbm_list
    })

    # Tranform to Numpy Array
    coordinates = coordinates.values
    parameters  = parameters.values
    distance_matrix = pivot_table.values
    real_distance_matrix = real_distance_matrix.values
    
    # Parameters - Model
    n_depots    =  1           # The First n Rows of the 'distance_matrix' or 'coordinates' are Considered as Depots
    time_window = 'with'       # 'with', 'without'
    route       = 'closed'     # 'open', 'closed'
    model       = 'vrp'        # 'tsp', 'mtsp', 'vrp'
    graph       = False         # True, False

    total_dict = get_total_dict(veh_table)

    tmp_veh = veh_table[veh_table['CurrentCenter'] == terminal_id]
    vehicle_index       = tmp_veh.index.to_list()
    #print("vehicle_index", vehicle_index)
    vehicle_types       = tmp_veh.shape[0] # 해당 출발지에 속한 차량 수
    fixed_cost          = tmp_veh['FixedCost'].values.tolist()
    variable_cost       = tmp_veh['VariableCost'].values.tolist()
    capacity            = tmp_veh['MaxCapaCBM'].values.tolist()
    velocity            = [1] * vehicle_types # 전부 1
    fleet_available     = total_dict[terminal_id][0] # 사용 가능하면 1, 불가능하면 0
    fleet_available_no_fixed_cost = total_dict[terminal_id][1] 

    # Parameters - GA
    penalty_value   = 1000000    # GA Target Function Penalty Value for Violating the Problem Constraints
    population_size = 5      # GA Population Size
    mutation_rate   = 0.2     # GA Mutation Rate
    elite           = 3        # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained 
    generations     = 3    # GA Number of Generations
    
    # Run GA Function
    ga_report, output_report, solution, fleet_used_now = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, real_distance_matrix, population_size, vehicle_types, n_depots, route, model, time_window, fleet_available, mutation_rate, elite, generations, penalty_value, graph, 'rw', fleet_available_no_fixed_cost, time_absolute = 1440 * day  +  plan_time * group,  order_id = order_id, city_name_list=city_name_list, vehicle_index = vehicle_index)   
    
    # 사용한 차량의 복귀 시간대 파악
    clean_report = ga_report[ga_report['Route'].str.startswith('#')]
    return_time = vehicle_return_time(clean_report, vehicle_types, veh_table, vehicle_index, time_absolute = 1440 * day  +  plan_time * group)
    update_veh_table(veh_table, vehicle_index, return_time, vehicle_types, terminal_id)
    #print("CenterArriveTime", veh_table['CenterArriveTime'].values.tolist())

    # 미처리 주문 파악
    unassigned_idx = solution[3]
    adjusted_unassigned_idx = [index - 1 for index in unassigned_idx]
    unassigned_rows = tmp_df.iloc[adjusted_unassigned_idx]

    unassigned_orders_count_dict.update({terminal_id: len(unassigned_rows)})
    print("unassigned_idx :", unassigned_idx)
    if not unassigned_rows.empty:
        # 넘기기 전에 하차가능시간 조정
        #print("이전_unassigned_rows_landing_start_times", unassigned_rows['landing_start_times'].values.tolist())
        unassigned_rows = unassigned_rows.apply(lambda row: update_times(row, number_of_t), axis=1)
        #print("이후_unassigned_rows_landing_start_times", unassigned_rows['landing_start_times'].values.tolist())
        unassigned_rows_dict.update({terminal_id: unassigned_rows})
        if day == total_days - 1 and group == number_of_t * 4 - 1:
            unassigned_orders_forever.update({terminal_id: len(unassigned_rows)})
    else:
        unassigned_rows_dict.update({terminal_id: None})
    
    return ga_report, output_report, fleet_used_now, len(unassigned_idx)

# distance_matrix랑 pivot_table_filled 만드는 함수 : 코드 완성한 다음에 테스트할때 실행
# make_matrix_csv()

# 제출 파일 폴더들 저장될 위치
FOLDER_PATH = './테스트2'
if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)
if not os.path.exists(f'{FOLDER_PATH}/report'):
    os.makedirs(f'{FOLDER_PATH}/report')
if not os.path.exists(f'{FOLDER_PATH}/제출파일1'):
    os.makedirs(f'{FOLDER_PATH}/제출파일1')
if not os.path.exists(f'{FOLDER_PATH}/제출파일1_최종'):
    os.makedirs(f'{FOLDER_PATH}/제출파일1_최종')
if not os.path.exists(f'{FOLDER_PATH}/제출파일2_최종'):
    os.makedirs(f'{FOLDER_PATH}/제출파일2_최종')

random.seed(42)

total_days = 7
start_day = '2023-05-01'
start_day = datetime.strptime(start_day, '%Y-%m-%d')
plan_time_hour = 6 # 몇시간 단위로 출발시키고 싶은지
number_of_t = int(6//plan_time_hour)
plan_time = plan_time_hour*60

terminal_table = pd.read_csv('./terminals.csv', encoding='cp949')
terminal_lst = terminal_table['ID'].unique()

unassigned_orders_count_dict = defaultdict(lambda : None)

veh_table = pd.read_csv('./vehicles.csv', encoding='cp949')
# veh_table에 'CurrentCenter', 'CenterArriveTime', 'IsUsed' 열 추가
# cueerntcenter : 이 veh이 도착할 center, centerarrivetime : 이 veh이 center에 도착할 시간, isused : 이 veh이 사용한 적 있는지 여부
veh_table['CurrentCenter'] = veh_table['StartCenter']
veh_table['CenterArriveTime'] = 0
veh_table['IsUsed'] = 0

dist_matrix = pd.read_csv("./distance_matrix.csv", index_col=0)
time_matrix = pd.read_csv("./pivot_table_filled.csv", index_col=0)

asc_dist_dict = time_distance_in_order(time_matrix, dist_matrix, terminal_lst)

# demand_df 불러온 후 하차가능시간 열 추가
demand_df = preprocess_demand_df(number_of_t=number_of_t)

max_car = set_max_car(terminal_lst)

infeasible_solution = []
unassigned_orders_forever = {}

total_cost = 0
unassigned_rows_dict = defaultdict(lambda : None)
future_rows_dict = defaultdict(lambda : None)

ga_column_names = ['Route', 'Vehicle', 'Activity', 'Job_도착지점의 index', 'Arrive_Load', 'Leave_Load', 'Wait_Time', 'Arrive_Time','Leave_Time', 'Distance', 'Costs']
total_ga_report = pd.DataFrame([], columns = ga_column_names)
output_column_names = ['ORD_NO', 'VehicleID', 'Sequence', 'SiteCode', 'ArrivalTime', 'WaitingTime', 'ServiceTime', 'DepartureTime', 'Delivered']
total_output_report = pd.DataFrame([], columns=output_column_names)

moved_df = pd.DataFrame(columns=['Veh_ID', 'Origin', 'Destination', 'day', 'group', 'travel_cost'])
for day in range(0, total_days): 
    for group in range(number_of_t*4): 
        tot_veh_num = 0
        for terminal_id in terminal_lst:
            print("terminal id:", terminal_id)
            print(f"day {day} group {group}")
            ga_report, output_report_, fleet_used_now, num_unassigned = run_ga(terminal_id, day, group, demand_df)
            if fleet_used_now:
                print("사용한 차량 수 :", sum(fleet_used_now))
            print("####################################")
            if ga_report is None:
                continue
            ga_report['Arrive_Time'] = ga_report['Arrive_Time'].apply(min_to_day)
            ga_report['Leave_Time']  = ga_report['Leave_Time'].apply(min_to_day)
            #ga_report.to_csv(f'{FOLDER_PATH}/report/ga_report-day-{day}-group-{group}-{terminal_id}.csv', encoding= 'cp949', index = False)
            output_report_.to_csv(f'{FOLDER_PATH}/report/output_report-day-{day}-group-{group}-{terminal_id}.csv', encoding= 'cp949', index = False)

            total_ga_report = pd.concat([total_ga_report, ga_report])
            total_output_report = pd.concat([total_output_report, output_report_])    

            total_cost += float(ga_report['Costs'].tolist()[-1])
            if float(ga_report['Costs'].tolist()[-1]) > 1000000:
                infeasible_solution.append(f"day-{day}-group-{group}-{terminal_id}")

            max_car = check_max_car(terminal_id, max_car, fleet_used_now, day, num_unassigned)

        # 미처리 주문에 대한 차량 재배치
        if not (day == total_days-1 and group == number_of_t*4-1):
            reallocate_veh(max_car, veh_table, asc_dist_dict, unassigned_orders_count_dict, terminal_lst, day, group, moved_df)
        # 시간 6시간 흐름
        veh_table['CenterArriveTime'] = veh_table['CenterArriveTime'].apply(lambda x: max(x - plan_time, 0))

        total_output_report.to_csv(f"{FOLDER_PATH}/제출파일1/total_output_report_day_{day}_group_{group}.csv", index=False, encoding='cp949')
        if group % number_of_t == number_of_t-1:
            get_submission_file_1(total_output_report, day, group, number_of_t, FOLDER_PATH, demand_df, start_day)

total_vehicle_report = vehicle_output_report(total_output_report)
total_vehicle_report.to_csv(f"{FOLDER_PATH}/제출파일2_최종/total_vehicle.csv", index=False, encoding='cp949')
print("total_cost :", total_cost)
print("infeasible_solution :", infeasible_solution)
print("unassigned_orders_forever :", unassigned_orders_forever)
print("terminal to terminal cost:", moved_df['travel_cost'].sum())
