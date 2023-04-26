import pandas as pd
import os

# output 폴더 내의 모든 csv 파일 불러오기
output_dir = 'output'
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

# 불러온 csv 파일들을 vote dataframe으로 변환
vote_df = pd.DataFrame()
for file in csv_files:
    df = pd.read_csv(os.path.join(output_dir, file))
    vote_df = pd.concat([vote_df, df['ans']], axis=1)

# vote count 계산 후 majority vote 적용
vote_count = vote_df.apply(pd.Series.value_counts, axis=1)
majority_vote = vote_count.idxmax(axis=1)

# 결과 csv 파일 저장
result_df = pd.DataFrame({'ImageID': df['ImageID'], 'ans': majority_vote})
result_df.to_csv(os.path.join(output_dir, 'result.csv'), index=False)