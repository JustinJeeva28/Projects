import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import time
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 1000)
pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)

# importing and setting up data

redWineData = pd.DataFrame(pd.read_csv('RedWineData.csv', delimiter=';'))
whiteWineData = pd.DataFrame(pd.read_csv('whiteWineData.csv', delimiter=';'))

print("\n\nSample redWineData head(5) :- \n\n", redWineData.head(5))
print("\n\nSample whiteWineData head(5) :- \n\n", whiteWineData.head(5))

print('\n\nshape of the redWineData: \n ', redWineData.shape)
print('\n\nshape of the whiteWineData: \n ', whiteWineData.shape)

redWineData['WineType'] = 'Red'
whiteWineData['WineType'] = 'White'

print("\n\nSample redWineData head(5) :- \n\n", redWineData.head(5) )
print("\n\nSample whiteWineData head(5) :- \n\n", whiteWineData.head(5) )

# Since most of the features are same, I am going To combine the two dataset for making the program more efficient

data = redWineData.append(whiteWineData, ignore_index=True)

print("\n\nSample Dataset head(5) :- \n\n", data.head(5) )
print('\n\nshape of the data: \n ', data.shape)
print( '\n\ndata.describe:\n', data.describe(include="all") )
print("\n\nNan in data:\n", data.isnull().sum())


print(data["WineType"].value_counts())
print(data["quality"].value_counts())
# Visualisation and modification of the data


plt.figure(figsize=(13,13))
sns.heatmap(data.corr(), annot=True, cmap= 'coolwarm')
plt.show()

columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
           "total sulfur dioxide","density","pH", "sulphates", "alcohol"]

for hnames in columns:
    sns.boxplot(data=data[hnames])
    plt.ylabel(hnames)
    plt.title(f'Box plot of {hnames} in wine')
    plt.show()

sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data=data)
plt.title('Box plot of the winedataset')
plt.show()

for column in columns:
    sns.displot(data[column], kde=True)
    plt.title(f'Distribution of {column} in wine dataset')
    plt.xlabel(column)
    plt.show()
# Adjust layout and display the plots


# From the boxplot we can observe thet the three columns residual sugar, free sulfur dioxide, total sulfur dioxide
# and fixed acidity has some outlier
# removing the outliers

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return df_no_outliers


data = remove_outliers(data, 'fixed acidity')
data = remove_outliers(data, 'residual sugar')
data = remove_outliers(data, 'free sulfur dioxide')
data = remove_outliers(data, 'total sulfur dioxide')
data = remove_outliers(data, 'chlorides')


print("\n\nSample Dataset head(5) :- \n\n", data.head(5) )
print("\n\ndatashape after removing outlier:\n",data.shape)

print("\n\n",data["WineType"].value_counts())
print('\n\n',data["quality"].value_counts())



sns.set()
plt.figure(figsize=(20, 10))
sns.boxplot(data=data)
plt.title('Box plot of the wine Dataset after removing outliers')
plt.show()

for column in columns:
    sns.displot(data[column], kde=True)
    plt.title(f'Distribution of {column} after removing outliers')
    plt.xlabel(column)
    plt.show()

#from corelation graph we can see that there is relation between alcohol and quality
plt.figure(figsize=(8, 6))
plt.bar(data['quality'], data['alcohol'], color='skyblue')
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.title('Bar Plot between quality and alcohol')
plt.show()



# count of quality of the  wine type
plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='WineType', hue=data.quality)
plt.xlabel("WineType")
plt.ylabel("Count")
plt.title("WineType Vs quality")
plt.show()


correlation_matrix = data.corr()
target_correlations = correlation_matrix['quality']
sorted_features = target_correlations.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(sorted_features.index, sorted_features.values, color='skyblue')
plt.xlabel('Correlation with Target Variable')
plt.ylabel('Features')
plt.title('Correlation between Features and quality ')
plt.show()

# model selection
data = pd.get_dummies(data)

input_predictors = data.drop(['quality'], axis=1)
ouptut_target = data["quality"]
print(ouptut_target.value_counts())

print("\n\nTraining Model on imbalanced data:\n")
x_train, x_val, y_train, y_val  = train_test_split(input_predictors, ouptut_target,
                                                    test_size = 0.25, random_state = 6)
# x_val = x_train , y_val = y_test

# Initialize the StandardScaler

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)





# training model with imbalanced data

# MODEL-0 )Linear Regression model
LinearRegression1 = LinearRegression()
start_time = time.time()
LinearRegression1.fit(x_train_scaled, y_train)
end_time = time.time()
predictions = LinearRegression1.predict(x_val_scaled)

# Calculate and print metrics
mse = mean_squared_error(y_val, predictions)
r2 = r2_score(y_val, predictions)

print("\n\nMean Squared Error of linearRegression:", mse*100)
print("R-squared of LinearRegression:", r2*100)

print("Training Time: {:.2f} seconds".format(end_time - start_time))

# MODEL-1) LogisticRegression
LogisticRegression1 = LogisticRegression()
start_time = time.time()
LogisticRegression1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = LogisticRegression1.predict(x_val_scaled)
acc_LogisticRegression = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-1: Accuracy of LogisticRegression: ", acc_LogisticRegression)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

#MODEL-2) Gaussian Naive Bayes

GaussianNB1 = GaussianNB()
start_time = time.time()
GaussianNB1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = GaussianNB1.predict(x_val_scaled)
acc_GaussianNB = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-2: Accuracy of GaussianNB: ", acc_GaussianNB)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

RandomForestClassifier1 = RandomForestClassifier()
start_time = time.time()
RandomForestClassifier1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = RandomForestClassifier1.predict(x_val_scaled)
acc_RandomForestClassifier = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-3: Accuracy of RandomForestClassifier: ", acc_RandomForestClassifier)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

KNeighborsClassifier1 = KNeighborsClassifier()
start_time = time.time()
KNeighborsClassifier1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = KNeighborsClassifier1.predict(x_val_scaled)
acc_KNeighborsClassifier = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-4: Accuracy of KNeighborsClassifier: ", acc_KNeighborsClassifier)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

LinearDiscriminantAnalysis1 = LinearDiscriminantAnalysis()
start_time = time.time()
LinearDiscriminantAnalysis1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = LinearDiscriminantAnalysis1.predict(x_val_scaled)
acc_LinearDiscriminantAnalysis = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-5: Accuracy of LinearDiscriminantAnalysis: ", acc_LinearDiscriminantAnalysis)
print("Training Time: {:.2f} seconds".format(end_time - start_time))


#MODEL-6) decisiontree
decisiontree1 = DecisionTreeClassifier()
start_time = time.time()
decisiontree1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = decisiontree1.predict(x_val_scaled)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-6: Accuracy of DecisionTreeClassifier: ", acc_decisiontree)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

#MODEL-7) SVC
SVC1 = SVC()
start_time = time.time()
SVC1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = SVC1.predict(x_val_scaled)
acc_SVC = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-7: Accuracy of SVC: ", acc_SVC)
print("Training Time: {:.2f} seconds".format(end_time - start_time))


print("\n\nTraining Model on balanced data:\n")

smote = SMOTE(k_neighbors=4)
input_predictors, ouptut_target = smote.fit_resample(input_predictors, ouptut_target)
print(ouptut_target.value_counts())

x_train, x_val, y_train, y_val  = train_test_split(input_predictors, ouptut_target,
                                                    test_size = 0.25, random_state = 6)
# x_val = x_train , y_val = y_test

# Initialize the StandardScaler

scaler = MinMaxScaler()
# Fit and transform the training data
x_train_scaled = scaler.fit_transform(x_train)

# Transform the validation data using the same scaler
x_val_scaled = scaler.transform(x_val)





# training model with imbalanced data

# MODEL-0 )Linear Regression model
LinearRegression1 = LinearRegression()
start_time = time.time()
LinearRegression1.fit(x_train_scaled, y_train)
end_time = time.time()
predictions = LinearRegression1.predict(x_val_scaled)

# Calculate and print metrics
mse = mean_squared_error(y_val, predictions)
r2 = r2_score(y_val, predictions)

print("\n\nMean Squared Error of linearRegression:", mse*100)
print("R-squared of LinearRegression:", r2*100)

print("Training Time: {:.2f} seconds".format(end_time - start_time))

# MODEL-1) LogisticRegression
LogisticRegression1 = LogisticRegression()
start_time = time.time()
LogisticRegression1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = LogisticRegression1.predict(x_val_scaled)
acc_LogisticRegression = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-1: Accuracy of LogisticRegression: ", acc_LogisticRegression)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

#MODEL-2) Gaussian Naive Bayes

GaussianNB1 = GaussianNB()
start_time = time.time()
GaussianNB1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = GaussianNB1.predict(x_val_scaled)
acc_GaussianNB = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-2: Accuracy of GaussianNB: ", acc_GaussianNB)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

RandomForestClassifier1 = RandomForestClassifier()
start_time = time.time()
RandomForestClassifier1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = RandomForestClassifier1.predict(x_val_scaled)
acc_RandomForestClassifier = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-3: Accuracy of RandomForestClassifier: ", acc_RandomForestClassifier)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

KNeighborsClassifier1 = KNeighborsClassifier()
start_time = time.time()
KNeighborsClassifier1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = KNeighborsClassifier1.predict(x_val_scaled)
acc_KNeighborsClassifier = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-4: Accuracy of KNeighborsClassifier: ", acc_KNeighborsClassifier)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

LinearDiscriminantAnalysis1 = LinearDiscriminantAnalysis()
start_time = time.time()
LinearDiscriminantAnalysis1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = LinearDiscriminantAnalysis1.predict(x_val_scaled)
acc_LinearDiscriminantAnalysis = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-5: Accuracy of LinearDiscriminantAnalysis: ", acc_LinearDiscriminantAnalysis)
print("Training Time: {:.2f} seconds".format(end_time - start_time))


#MODEL-6) decisiontree
decisiontree1 = DecisionTreeClassifier()
start_time = time.time()
decisiontree1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = decisiontree1.predict(x_val_scaled)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-6: Accuracy of DecisionTreeClassifier: ", acc_decisiontree)
print("Training Time: {:.2f} seconds".format(end_time - start_time))

#MODEL-7) SVC
SVC1 = SVC()
start_time = time.time()
SVC1.fit(x_train_scaled, y_train)
end_time = time.time()
y_pred = SVC1.predict(x_val_scaled)
acc_SVC = round(accuracy_score(y_pred, y_val) * 100, 2)

# Print accuracy and training time
print("\n\nMODEL-7: Accuracy of SVC: ", acc_SVC)
print("Training Time: {:.2f} seconds".format(end_time - start_time))






