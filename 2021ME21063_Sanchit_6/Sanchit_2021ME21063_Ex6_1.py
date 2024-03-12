def main():
    # Importing necessary libraries
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report

    # Setting random seed
    np.random.seed(1234)

    # Importing the excel file
    data = pd.read_excel('mmpp_data_g12.xls', header=None)
    X = data.drop(2, axis=1)
    y = data[2]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

    # Gradient Boosting Classifier
    def gradient_boosting_classifier(n_estimators, random_state):
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state).fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test,y_pred)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Gradient Boosting Classifier Accuracy measures (AUC, F1 Score): ", (auc,f1))
        print("\nGradient Boosting Classifier Classification report: ")
        print(report)
        print('----------------------\n')


    # Neural Network Classifier
    def neural_network_classifier(hidden_layer_sizes, activation, random_state):
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=random_state).fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test,y_pred)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Neural Network Accuracy measures (AUC, F1 Score): ", (auc,f1))
        print("\nNeural Network Classification report: ")
        print(report)
        print('----------------------\n')
        

    # Logistic Regression Classifier
    def logistic_regression_classifier(random_state):
        model = LogisticRegression(random_state=random_state).fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test,y_pred)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Logistic Regression Accuracy measures (AUC, F1 Score): ", (auc,f1))
        print("\nLogistic Regression Classification report: ")
        print(report)
        print('----------------------\n')

    # Support Vector Machine
    def support_vector_machine(gamma, kernel,random_state):
        model = SVC(gamma=gamma, kernel=kernel, random_state=random_state).fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test,y_pred)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Support Vector Machine Accuracy measures (AUC, F1 Score): ",(auc,f1))
        print("\nSupport Vector Machine Classification report: ")
        print(report)
        print('----------------------\n')
        
        
    # Calling the defined functions
    gradient_boosting_classifier(250,10)
    neural_network_classifier((150,100,30),'relu',10)
    logistic_regression_classifier(10)
    support_vector_machine('auto','rbf',10)
    
    
if __name__ == '__main__':
    main()