def main():
    # Importing necessary libraries
    import numpy as np
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    import matplotlib.pyplot as plt
    
    # Setting random seed
    np.random.seed(1234)
    
    # Importing the excel file
    data = pd.read_excel('mmpp_data_g12.xls', header=None)
    
    # Convert the zeroes to ones
    def convert_zeroes_to_ones(data,num):
        zero_indices = data[data[2] == 0].index
        num_changes = min(num-461,len(zero_indices))
        selected_indices = np.random.choice(zero_indices,num_changes,replace=False)
        dataset = data.copy()
        
        dataset.loc[selected_indices,2] = 1
        return dataset
    
    #  750, 1250, 1750, 2250 and 3000
    data1 = convert_zeroes_to_ones(data,750)
    data2 = convert_zeroes_to_ones(data,1250)
    data3 = convert_zeroes_to_ones(data,1750)
    data4 = convert_zeroes_to_ones(data,2250)
    data5 = convert_zeroes_to_ones(data,3000)
    
    # Neural Network
    def neural_network_classifier(hidden_layer_sizes, activation, random_state, data):
        # data preparation
        from sklearn.model_selection import train_test_split
        X = data.drop(2, axis=1)
        y = data[2]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
        
        # model preparation with the given params
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=random_state).fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        # Accuracy Scores
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return auc,f1,accuracy
    
    # Getitng the roc, f1, accuracy data points
    data_arr = [data1,data2,data3,data4,data5]
    roc = []
    f1 = []
    accuracy = []
    ones_count = [750,1250,1750,2250,3000]
    for ele in data_arr:
        accuracy_data = neural_network_classifier((150,100,30),'relu',10,ele)
        roc.append(accuracy_data[0])
        f1.append(accuracy_data[1])
        accuracy.append(accuracy_data[2])
        
    # Plotting the roc, f1, accuracy 
    plt.plot(roc, marker='s', label="ROC")
    plt.plot(f1, marker = 'o', label="F1 Score")
    plt.plot(accuracy, marker = '*', label="Accuracy")
    plt.legend(['ROC', 'F1', 'Accuracy'])
    plt.xticks(np.arange(0,5,1), ones_count)
    plt.xlabel('Number of Ones')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Ones')
    plt.show()

if __name__ == "__main__":
    main()
    
    
# The scores ROC, F1, Classification Accuracy decreases as the number of ones increases.