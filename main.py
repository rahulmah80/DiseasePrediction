import numpy as np
import pandas as pd

data = pd.read_csv("DatasetKaggle/dataset.csv")
print(data.head())
print(data.shape)

data["list_of_Symps"] = 0
for i in range(data.shape[0]):
    values = data.iloc[i].values

    values = values.tolist()

    data["list_of_Symps"][i] = values[1:values.index(0)]

# print(data.tail)
# print(data.head())
description = pd.read_csv("DatasetKaggle/symptom_Description.csv")

column_values = data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
                      'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
                      'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
                      'Symptom_15', 'Symptom_16', 'Symptom_17']].values.ravel()
# print(column_values)
symptoms = pd.unique(column_values.tolist())
# nan values
symptoms = [i for i in symptoms if str(i) != "nan"]
# print(len(symptoms))

new_data = pd.DataFrame(columns=symptoms, index=data.index)
new_data['list_of_Symps'] = data['list_of_Symps']

# filling in new data
for i in new_data:
    new_data[i] = data.apply(lambda x: 1 if i in x.list_of_Symps else 0, axis=1)

new_data['Disease'] = data['Disease']
new_data = new_data.drop('list_of_Symps', axis=1)

print(new_data.head())
print(new_data.shape)
new_data.to_csv('DatasetKaggle/modified_dataset.csv')