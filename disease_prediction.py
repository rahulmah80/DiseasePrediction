import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from gui import *
from sklearn.metrics import classification_report
from tkinter import *

# reading raw data
raw_data = pd.read_csv("DatasetKaggle/dataset.csv")
# preprocessing of raw dataset
'''
# print(raw_data.head())
# print(raw_data.shape)
# print(raw_data.describe())
'''
# analysing no. of null values in columns
'''
null_checker = raw_data.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_checker)
# graphical representation of number of null values in dataset
plt.figure(figsize=(10, 5), dpi=140)
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(null_checker.index, null_checker.index, rotation=45, horizontalalignment='right')
plt.title('No. of Null values')
plt.xlabel('Symptoms')
plt.margins(0.1)
plt.show()
'''

# The dataset can not be directly used for training and testing. 
# It needs to be reformatted into a different form where columns are unique symptoms, 
# whereas each entry is a binary number depending on 
# whether the symptom is part of corresponding disease symptoms.

# creating list of symptoms in new column
raw_data["symp_list"] = 0
for i in range(raw_data.shape[0]):
    values = raw_data.iloc[i].values
    values = values.tolist()
    raw_data["symp_list"][i] = values[1:values.index(0)]
# print(raw_data.head())

# to count number of diseases
column_values = raw_data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
                          'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
                          'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
                          'Symptom_16', 'Symptom_17']].values.ravel()
symptoms = pd.unique(column_values.tolist())
# removing nan values
symptoms = [i for i in symptoms if str(i) != "nan"]
# print(len(symptoms))

# creating modified dataframe
data = pd.DataFrame(columns=symptoms, index=raw_data.index)
data['symp_list'] = raw_data['symp_list']
# filling in binary data for diseases
for i in data:
    data[i] = raw_data.apply(lambda x: 1 if i in x.symp_list else 0, axis=1)
# copying prognosis/disease column
data['Disease'] = raw_data['Disease']
# dropping symp_list column since all data extracted from it
data = data.drop('symp_list', axis=1)
# print(data.head())
# print(data.tail())
# print(data.shape)
# print(raw_data.describe())
# saving modified dataframe into a csv file
# data.to_csv('DatasetKaggle/modified_dataset.csv')

# checking whether the dataset is balanced or not
'''
disease_counts = data["Disease"].value_counts()
dis_count_df = pd.DataFrame({"Disease": disease_counts.index, "Counts": disease_counts.values})
plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=dis_count_df)
plt.xticks(rotation=90)
plt.show()
'''

# A new dataset was found to be used for testing purpose of models.
# Before using, it needs to be cleaned and reformatted/reshaped into a form similar to training data

# reading new test data
test_data_raw1 = pd.read_csv("DatasetKaggle/Testing.csv")
# print(test_data_raw1.head())
# print(test_data_raw1.shape)

# reformatting (essentially aligning the symptoms sequence with that of training dataset)
test_data1 = pd.DataFrame(columns=data.columns.values.tolist())
# print(test_data1)
for ind in test_data_raw1.index:
    for colname, colval in test_data_raw1.iteritems():
        if test_data_raw1.at[ind, colname] == 1:
            for col, colval1 in test_data1.iteritems():
                if col.split() == colname.split():
                    test_data1.at[ind, col] = 1
        elif test_data_raw1.at[ind, colname] == 0:
            for col, colval1 in test_data1.iteritems():
                if col.split() == colname.split():
                    test_data1.at[ind, col] = 0
        else:
            test_data1.at[ind, 'Disease'] = test_data_raw1.at[ind, 'prognosis']
# saving modified dataframe into a csv file
# test_data1.to_csv('DatasetKaggle/modified_Testing.csv')

# separating train and test data from main dataset
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)
x_train = training_data.drop("Disease", axis=1)
y_train = training_data["Disease"].copy()
x_test = testing_data.drop("Disease", axis=1)
y_test = testing_data["Disease"].copy()
x_test1 = test_data1.drop("Disease", axis=1)
y_test1 = test_data1["Disease"].copy()
# print(x_train.head())
# print(y_train)
# print(x_test.head())
# print(y_test)
# print(x_test1.head())
# print(y_test1)

# input symptoms from user
input_symptoms = []
for i in range(len(symptoms)):
    input_symptoms.append(0)
# GUI related
root = Tk()
root.title('Disease Prediction')
root.configure(background='grey')
Symptom1 = StringVar()
Symptom1.set("Select Here")
Symptom2 = StringVar()
Symptom2.set("Select Here")
Symptom3 = StringVar()
Symptom3.set("Select Here")
Symptom4 = StringVar()
Symptom4.set("Select Here")
Symptom5 = StringVar()
Symptom5.set("Select Here")
Name = StringVar()

print("\n----------Random Forest Classifier----------")
rnd_forest = RandomForestClassifier()
rnd_forest.fit(x_train, y_train)
rnd_pred = rnd_forest.predict(x_test)
rnd_pred1 = rnd_forest.predict(x_test1)
print("Accuracy for main dataset:", metrics.accuracy_score(y_test, rnd_pred) * 100, "%")
print("Accuracy for new testing dataset:", metrics.accuracy_score(y_test1, rnd_pred1) * 100, "%")
print("Cross Validation Score:", cross_val_score(rnd_forest, x_train, y_train, cv=5).mean() * 100, "%")
cf_matrix = metrics.confusion_matrix(y_test, rnd_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Split Test Data")
# plt.show()
cf_matrix = metrics.confusion_matrix(y_test1, rnd_pred1)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on New Test Data")
# plt.show()

print("\n----------K Neighbors Classifier----------")
knc = KNeighborsClassifier(n_neighbors=7).fit(x_train, y_train)
knc_pred = knc.predict(x_test)
knc_pred1 = knc.predict(x_test1)
print("Accuracy for main dataset:", metrics.accuracy_score(y_test, knc_pred) * 100, "%")
print("Accuracy for new testing dataset:", metrics.accuracy_score(y_test1, knc_pred1) * 100, "%")
print("Cross Validation Score:", cross_val_score(knc, x_train, y_train, cv=5).mean() * 100, "%")
cf_matrix = metrics.confusion_matrix(y_test, knc_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for K-neighbors Classifier on Split Test Data")
# plt.show()
cf_matrix = metrics.confusion_matrix(y_test1, knc_pred1)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for K-neighbors Classifier on New Test Data")
# plt.show()

print("\n----------Support Vector Machine----------")
svm_model = SVC()
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)
svm_pred1 = svm_model.predict(x_test1)
print("Accuracy for main dataset:", metrics.accuracy_score(y_test, svm_pred) * 100, "%")
print("Accuracy for new testing dataset:", metrics.accuracy_score(y_test1, svm_pred1) * 100, "%")
print("Cross Validation Score:", cross_val_score(svm_model, x_train, y_train, cv=5).mean() * 100, "%")
cf_matrix = metrics.confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Support Vector Machine on Split Test Data")
# plt.show()
cf_matrix = metrics.confusion_matrix(y_test1, svm_pred1)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Support Vector Machine on New Test Data")
# plt.show()

print("\n----------Gaussian Naive Bayes----------")
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_pred = gnb.predict(x_test)
gnb_pred1 = gnb.predict(x_test1)
print("Accuracy for main dataset:", metrics.accuracy_score(y_test, gnb_pred) * 100, "%")
print("Accuracy for new testing dataset:", metrics.accuracy_score(y_test1, gnb_pred1) * 100, "%")
print("Cross Validation Score:", cross_val_score(gnb, x_train, y_train, cv=5).mean() * 100, "%")
cf_matrix = metrics.confusion_matrix(y_test, gnb_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Gaussian Naive Bayes on Split Test Data")
# plt.show()
cf_matrix = metrics.confusion_matrix(y_test1, gnb_pred1)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Gaussian Naive Bayes on New Test Data")
# plt.show()


def random_forest():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(symptoms)):
        for z in psymptoms:
            if z == symptoms[k]:
                input_symptoms[k] = 1
    inputtest = [input_symptoms]
    predict = rnd_forest.predict(inputtest)
    predicted = predict[0]
    print(predicted)
    t1.delete("1.0", END)
    t1.insert(END, predicted)


def k_neighbors_classifier():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(symptoms)):
        for z in psymptoms:
            if z == symptoms[k]:
                input_symptoms[k] = 1
    inputtest = [input_symptoms]
    predict = knc.predict(inputtest)
    predicted = predict[0]
    t2.delete("1.0", END)
    t2.insert(END, predicted)


def support_vector_machine():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(symptoms)):
        for z in psymptoms:
            if z == symptoms[k]:
                input_symptoms[k] = 1
    inputtest = [input_symptoms]
    predict = svm_model.predict(inputtest)
    predicted = predict[0]
    t3.delete("1.0", END)
    t3.insert(END, predicted)


def gaussian_naive_bayes():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(symptoms)):
        for z in psymptoms:
            if z == symptoms[k]:
                input_symptoms[k] = 1
    inputtest = [input_symptoms]
    predict = gnb.predict(inputtest)
    predicted = predict[0]
    t4.delete("1.0", END)
    t4.insert(END, predicted)


# GUI

w2 = Label(root, justify=CENTER, anchor=CENTER, text="Disease Prediction using Machine Learning", fg="Black", bg='grey')
w2.config(font=("Helvetica", 20, "bold"))
w2.grid(row=1, column=0, columnspan=3, padx=0)

NameLb = Label(root, text="Name of the Patient", bg="grey")
NameLb.config(font=("Helvetica", 13, "bold"))
NameLb.grid(row=6, column=0, pady=13, sticky=W)

S1Lb = Label(root, text="Enter Symptom 1", bg="grey")
S1Lb.config(font=("Helvetica", 13, "bold"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)
S2Lb = Label(root, text="Enter Symptom 2", bg="grey")
S2Lb.config(font=("Helvetica", 13, "bold "))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)
S3Lb = Label(root, text="Enter Symptom 3", bg="grey")
S3Lb.config(font=("Helvetica", 13, "bold "))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)
S4Lb = Label(root, text="Enter Symptom 4", bg="grey")
S4Lb.config(font=("Helvetica", 13, "bold "))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)
S5Lb = Label(root, text="Enter Symptom 5", bg="grey")
S5Lb.config(font=("Helvetica", 13, "bold "))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

rfcLB = Label(root, text="Random Forest Classification", bg="grey")
rfcLB.config(font=("Helvetica", 13, "bold "))
rfcLB.grid(row=13, column=0, pady=10, sticky=W)
knbLb = Label(root, text="K neighbour classification", bg="grey")
knbLb.config(font=("Helvetica", 13, "bold "))
knbLb.grid(row=17, column=0, pady=10, sticky=W)
svmLB = Label(root, text="Support Vector Machine", bg="grey")
svmLB.config(font=("Helvetica", 13, "bold "))
svmLB.grid(row=19, column=0, pady=10, sticky=W)
gnbLB = Label(root, text="Gaussian Naive Bayes", bg="grey")
gnbLB.config(font=("Helvetica", 13, "bold "))
gnbLB.grid(row=21, column=0, pady=10, sticky=W)

OPTIONS = sorted(symptoms)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1 = OptionMenu(root, Symptom1, *OPTIONS)
S1.grid(row=7, column=1)
S2 = OptionMenu(root, Symptom2, *OPTIONS)
S2.grid(row=8, column=1)
S3 = OptionMenu(root, Symptom3, *OPTIONS)
S3.grid(row=9, column=1)
S4 = OptionMenu(root, Symptom4, *OPTIONS)
S4.grid(row=10, column=1)
S5 = OptionMenu(root, Symptom5, *OPTIONS)
S5.grid(row=11, column=1)

rfc_button = Button(root, text="Predict", command=random_forest)
rfc_button.config(font=("Helvetica", 12, "bold "))
rfc_button.grid(row=13, column=2, padx=10)

knc_button = Button(root, text="Predict", command=k_neighbors_classifier)
knc_button.config(font=("Helvetica", 12, "bold "))
knc_button.grid(row=17, column=2, padx=10)

svm_button = Button(root, text="Predict", command=support_vector_machine)
svm_button.config(font=("Helvetica", 12, "bold "))
svm_button.grid(row=19, column=2, padx=10)

gnb_button = Button(root, text="Predict", command=gaussian_naive_bayes)
gnb_button.config(font=("Helvetica", 12, "bold "))
gnb_button.grid(row=21, column=2, padx=10)

t1 = Text(root, height=1, width=25)
t1.config(font=("Helvetica", 13, "bold "))
t1.grid(row=13, column=1, padx=10)
t2 = Text(root, height=1, width=25)
t2.config(font=("Helvetica", 13, "bold "))
t2.grid(row=17, column=1, padx=10)
t3 = Text(root, height=1, width=25)
t3.config(font=("Helvetica", 13, "bold "))
t3.grid(row=19, column=1, padx=10)
t4 = Text(root, height=1, width=25)
t4.config(font=("Helvetica", 13, "bold "))
t4.grid(row=21, column=1, padx=10)

root.mainloop()