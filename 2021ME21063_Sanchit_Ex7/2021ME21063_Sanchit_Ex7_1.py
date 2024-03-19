# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

# Importing the datasheet
data = pd.read_excel("data2sep.xlsx", sheet_name="fin_full")

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=20, n_init='auto')
labels = kmeans.fit_predict(data[['tot_deaths_pm']])
data['label'] = labels
data.to_excel("dsml.xlsx", index=False) # Data exported to dsml.xlsx file


# Splitting in train and test dataset
# Dropping columns with non-numerical values and tot_deaths_pm (used for labels)
data.drop(['continent', 'location', 'tot_deaths_pm'], axis=1, inplace=True)
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)


# SMOTE technique
print(f"Training set dimensions before oversampling: X:{X_train.shape}, y:{y_train.shape}")
smote = SMOTE(random_state=20)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Training set dimensions after oversampling: X:{X_train_smote.shape}, y:{y_train_smote.shape}")

# Scaling between [0,1] using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Defining the classifiers for predicting
classifiers = [
    RandomForestClassifier(n_estimators=100, random_state=20),
    MLPClassifier(hidden_layer_sizes=(50, 50, 10), max_iter=500, activation='relu', random_state=20),
    LogisticRegression(random_state=20)
]

# Looping over all the classifiers
for clf in classifiers:
    clf.fit(X_train_scaled, y_train_smote)
    y_pred = clf.predict(X_test_scaled)
    print(f"\n{clf.__class__.__name__} results:")
    print(classification_report(y_test, y_pred))
    print(f"AUC score: {roc_auc_score(y_test, y_pred):.2f}")
    print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred):.2f}")
    print("-------------------------------------------------------------------")
