import pandas as pd
import numpy as np


urls = 'data.csv'
csv = pd.read_csv(urls, error_bad_lines=False)
data = pd.DataFrame(csv)

data = np.array(data)
#y = [d[1] for d in
cnt = 0
good = bad = 0
for d in data:
    if d[1] == 'good':
        good += 1
    else:
        bad += 1
    cnt += 1
print("total count:", cnt)
print("good count:", good)
print("bad count", bad)
