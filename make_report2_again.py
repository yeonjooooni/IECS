import os
import pandas as pd
from tqdm.auto import tqdm
df = pd.read_csv("./테스트2/제출파일2_최종/total_vehicle.csv", encoding='cp949')
orders = pd.read_csv("./orders.csv", encoding='cp949')
result = pd.read_csv("./테스트2/제출파일1_최종/total_output_report_day_3_group_3.csv",encoding='cp949')

veh_ids = df['VehicleID'].unique()
for veh_id in tqdm(veh_ids):
    tmp = 0
    order_did = result[result['VehicleID']==veh_id]['ORD_NO'].values.tolist()
    order_did = [i for i in order_did if i != 'Null']
    for i in order_did:
        tmp += orders[orders['주문ID'] == i]['하차작업시간(분)'].values[0]
    df.loc[df['VehicleID'] == veh_id, 'ServiceTime'] = tmp  # Use boolean indexing to update the 'ServiceTime' column
    df.loc[df['VehicleID'] == veh_id, 'WorkTime'] = df.loc[df['VehicleID'] == veh_id, 'TravelTime'] + df.loc[df['VehicleID'] == veh_id, 'ServiceTime'] + df.loc[df['VehicleID'] == veh_id, 'WaitingTime']

df.to_csv("./test2222.csv", encoding='cp949')
