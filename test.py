import pandas as pd
import numpy as np

'''
data = pd.read_csv("Dataset/BP.csv", usecols=['Timestamp'], nrows=174982)

dataset = pd.read_csv("Dataset/hypertension_dataset.csv",usecols=['Systolic_BP','Diastolic_BP'])
dataset.insert(0, 'Timestamp', data.values.ravel())
print(dataset)
labels = []
data = dataset.values

for i in range(len(data)):
    sys = data[i,1]
    dia = data[i,2]
    if sys >= 90 and sys < 120 and dia <= 80:
        labels.append("Normal")
    elif sys > 0 and sys < 90 and dia <= 60:
        labels.append("Low_BP_Hypotension")
    elif sys >= 120 and sys <= 129 and dia < 80:
        labels.append("Elevated")
    elif sys >= 130 and sys <= 139 and dia >= 80 and dia < 90:
        labels.append("High_BP_Stage1_Hypertension")
    elif sys >= 140 and sys < 180 and dia > 90 and dia < 120:
        labels.append("High_BP_Stage2_Hypertension")
    else:
        labels.append("Hypertensive_Crisis")

labels = np.asarray(labels)
unique, count = np.unique(labels, return_counts=True)
print(unique)
print(count)

dataset['label'] = labels
dataset.to_csv("Dataset/BP.csv", index=False)
'''

dataset = pd.read_csv("Dataset/BP.csv")

d1 = dataset.loc[dataset['label'] == 'Elevated']
d2 = dataset.loc[dataset['label'] == 'High_BP_Stage1_Hypertension']
d3 = dataset.loc[dataset['label'] == 'High_BP_Stage2_Hypertension']
d4 = dataset.loc[dataset['label'] == 'Hypertensive_Crisis']
d5 = dataset.loc[dataset['label'] == 'Normal']

d3 = d3.iloc[0:5000]
d4 = d4.iloc[0:8000]
d5 = d5.iloc[0:7100]

data = pd.concat([d1, d2, d3, d4, d5])
print(data)
data.to_csv("Dataset/test.csv", index=False)

















