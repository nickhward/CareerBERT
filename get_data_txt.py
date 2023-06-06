import pandas as pd
from tqdm import tqdm

# assuming df is your DataFrame and 'text' is the column with the text
df = pd.read_csv('glassdoor_jobs.csv')

# use tqdm in your loop for a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    with open(f'train_data/file_{index}.txt', 'w', encoding='utf-8') as f:
        f.write(str(row['Job Description']))
