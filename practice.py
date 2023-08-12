import pandas as pd
output_report = pd.read_csv('total_output_Report.csv')
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
print(report_df.head(3))