import pandas as pd
attentionres = [0.2,0.3,0.4,0.5,0.6,0.7,0.]
dfattention = pd.DataFrame(attentionres)
print(dfattention)
dfattention.to_excel(r'./attention.xlsx', index = False)
dfattention.to_csv(r'./attention.csv', index = False)
