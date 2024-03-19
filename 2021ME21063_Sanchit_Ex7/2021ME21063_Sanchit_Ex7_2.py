# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

# Importing the datasheet
data = pd.read_excel("data2sep.xlsx", sheet_name="fin_full")

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=20, n_init='auto')
labels = kmeans.fit_predict(data[['tot_deaths_pm']])
data['label'] = labels

# Splitting in train and test dataset
# Dropping columns with non-numerical values and tot_deaths_pm (used for labels)
data.drop(['continent', 'location', 'tot_deaths_pm'], axis=1, inplace=True)
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)

# SMOTE technique
smote = SMOTE(random_state=20)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scaling between [0,1] using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


# Random Forest
rf_param_grid = {'n_estimators': range(50, 501, 50)}
kf = KFold(n_splits=5, shuffle=True, random_state=20) # 5-Fold 
rf_scores = [] # Initialise the random forest scores array

# Looping over all the parameters (n_estimators)
for n_estimators in rf_param_grid['n_estimators']: 
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=20)
    scores = []
    
    # Splitting the training dataset into train test data for 5-Fold validation
    for train_idx, test_idx in kf.split(X_train_scaled):
        X_train_cv, X_test_cv = X_train_scaled[train_idx], X_train_scaled[test_idx]
        y_train_cv, y_test_cv = y_train_smote[train_idx], y_train_smote[test_idx]
        rf.fit(X_train_cv, y_train_cv)
        y_pred_cv = rf.predict(X_test_cv)
        scores.append(balanced_accuracy_score(y_test_cv, y_pred_cv)) # scores based on balanced_accuracy
    rf_scores.append((n_estimators, sum(scores) / len(scores))) 

rf_best_params = max(rf_scores, key=lambda x: x[1]) # Extracting the best scores
print(f'Random Forest Optimal Hyperparameters: n_estimators={rf_best_params[0]}')
rf_data = pd.DataFrame(rf_scores, columns=['n_estimators', 'Scores']) # Converting array to dataframe to export to excel


# Neural Network
nn_param_grid = {
    'hidden_layer_sizes': [(50, 50), (55, 55), (60, 60), (65, 65), (70, 70), (75, 75), (80, 80),
                            (50, 50, 50), (55, 55, 55), (60, 60, 60), (65, 65, 65), (70, 70, 70), (75, 75, 75), (80, 80, 80)],
    'activation': ['relu', 'logistic'] # Defining the params (logistic -> sigmoid)
}
kf = KFold(n_splits=5, shuffle=True, random_state=20) # 5-fold
nn_scores = [] # Initialise the neural network scores array

# Looping over all the parameters (activation, hidden_layer_sizes)
for activation in nn_param_grid['activation']:
    for hidden_layer_sizes in nn_param_grid['hidden_layer_sizes']:
        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=500, random_state=20)
        scores = []
        
        # Splitting the training dataset into train test data for 5-Fold validation
        for train_idx, test_idx in kf.split(X_train_scaled):
            X_train_cv, X_test_cv = X_train_scaled[train_idx], X_train_scaled[test_idx]
            y_train_cv, y_test_cv = y_train_smote[train_idx], y_train_smote[test_idx]
            nn.fit(X_train_cv, y_train_cv)
            y_pred_cv = nn.predict(X_test_cv)
            scores.append(balanced_accuracy_score(y_test_cv, y_pred_cv)) # scores based on balanced_accuracy
        nn_scores.append((hidden_layer_sizes, activation, sum(scores) / len(scores)))

nn_best_params = max(nn_scores, key=lambda x: x[2]) # Extracting the best scores
print(f'Neural Network Optimal Hyperparameters: hidden_layer_sizes={nn_best_params[0]}, activation="{nn_best_params[1]}"')
# Converting array to dataframe to export to excel
nn_data = pd.DataFrame(nn_scores, columns=['Layers', 'Activation', 'Scores'])
nn_data['Activation'] = nn_data['Activation'].replace({'logistic' : 'sigmoid'})

# Exporting to excel file
with pd.ExcelWriter('hptoutput.xlsx') as writer:
    rf_data.to_excel(writer, sheet_name='random_forest', index=False)
    nn_data.to_excel(writer, sheet_name='neural_net', index=False)


