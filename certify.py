import pandas as pd
result = pd.read_csv("./제출파일1/total_output_report_day_4_group_{마지막그룹}.csv", encoding='cp949')
result = result[(result['ORD_NO'] !='-//-')]
result = result.dropna(subset=['ORD_NO'])
print("총 처리 주문 수:", len(result))


import numpy as np
output_report = pd.read_csv("./제출파일1/total_output_report_day_4_group_{마지막그룹}.csv", encoding='cp949')
orders_table = pd.read_csv("./과제3 실시간 주문 대응 Routing 최적화 (orders_table) 수정완료.csv", encoding='cp949')
output_report = output_report[output_report['ORD_NO'] != '-//-']
output_report = output_report.dropna(subset=['ORD_NO'])

group_to_start_time = {
    0: '00:00',
    1: '06:00',
    2: '12:00',
    3: '18:00'
}

merged_df = pd.merge(output_report, orders_table, left_on='ORD_NO', right_on='주문ID', how='left')
merged_df['StartTime'] = merged_df['Group'].map(group_to_start_time)
merged_df['StartDateTime'] = pd.to_datetime(merged_df['date'] + ' ' + merged_df['StartTime'])
merged_df['ArrivalTime'] = pd.to_datetime(merged_df['ArrivalTime'])
merged_df['TimeDifference'] = merged_df['ArrivalTime'] - merged_df['StartDateTime']
merged_df['Within3Days'] = merged_df['TimeDifference'].apply(lambda x: True if x.days <= 3 else False)
print("3일 이내에 처리안된 주문 수", len(merged_df[merged_df['Within3Days'] == False]))