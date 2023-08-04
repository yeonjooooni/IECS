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
