import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from FCClassifier import FullyConnectedClassifier

KFOLD_N = 5


def main():
    # load the data
    data = pd.read_csv('dataset.csv')
    labels = data['output']

    # dropping the label column
    data = data.drop(columns=['output'])
    data = np.array(data)
    labels = np.array(labels)

    # Random Forest tuning using GridSearchCV
    args_rf = tune_rf(data, labels)

    # Random Forest
    kf = KFold(n_splits=KFOLD_N, random_state=42, shuffle=True)
    sum_acc = 0
    run_index = 0
    for train_index, test_index in kf.split(data):
        model = RandomForestClassifier(n_estimators=args_rf["n_estimators"], max_depth=args_rf["max_depth"],
                                       criterion=args_rf["criterion"], max_features=args_rf["max_features"])
        model = make_pipeline(StandardScaler(), model)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        lr = make_pipeline(StandardScaler(), model)
        lr.fit(X_train, y_train)
        y_hat = lr.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        sum_acc += acc
        run_index += 1
        if run_index == KFOLD_N - 1:
            print_cm(y_test, y_hat, "Random Forest")
        # print(f'Acc: {acc}%')
    print(f"RF:Avg Acc:{100*sum_acc / KFOLD_N}")

    # Logistic Regression
    kf = KFold(n_splits=KFOLD_N, random_state=42, shuffle=True)
    sum_acc = 0
    run_index = 0
    for train_index, test_index in kf.split(data):
        model = LogisticRegression()
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        lr = make_pipeline(StandardScaler(), model)
        lr.fit(X_train, y_train)
        y_hat = lr.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        sum_acc += acc
        run_index += 1
        if run_index == KFOLD_N - 1:
            print_cm(y_test, y_hat, "Logistic Regression")
        # print(f'Acc: {acc}%')
    print(f"LR:Avg Acc:{100*sum_acc / KFOLD_N}%")

    # SVM
    kf = KFold(n_splits=KFOLD_N, random_state=42, shuffle=True)
    sum_acc = 0
    run_index = 0
    for train_index, test_index in kf.split(data):
        model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        lr = make_pipeline(StandardScaler(), model)
        lr.fit(X_train, y_train)
        y_hat = lr.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        sum_acc += acc
        run_index += 1
        if run_index == KFOLD_N - 1:
            print_cm(y_test, y_hat, "SVM")
        # print(f'Acc: {acc}%')
    print(f"SVM:Avg Acc:{100*sum_acc / KFOLD_N}%")

    # KNN
    # Tune using GridSearchCV
    args_rf = tune_knn(data, labels)
    kf = KFold(n_splits=KFOLD_N, random_state=42, shuffle=True)
    sum_acc = 0
    run_index = 0
    for train_index, test_index in kf.split(data):
        model = make_pipeline(StandardScaler(),
                              KNeighborsClassifier(n_neighbors=args_rf['n_neighbors'], metric=args_rf['metric'],
                                                   weights=args_rf['weights']))
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        lr = make_pipeline(StandardScaler(), model)
        lr.fit(X_train, y_train)
        y_hat = lr.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        sum_acc += acc
        run_index += 1
        if run_index == KFOLD_N - 1:
            print_cm(y_test, y_hat, "KNeighbors")
        # print(f'Acc: {acc}%')
    print(f"KNN:Avg Acc:{100*sum_acc / KFOLD_N}%")

    # Fully Connected
    kf = KFold(n_splits=KFOLD_N, random_state=42, shuffle=True)
    scaler = MinMaxScaler()
    scaler.fit(data)
    sum_acc = 0
    for train_index, test_index in kf.split(data):
        model = FullyConnectedClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        tensor_x = torch.Tensor(X_train)  # transform to torch tensor
        tensor_y = torch.Tensor(y_train)
        test_tensor_x = torch.Tensor(X_test)  # transform to torch tensor
        test_tensor_y = torch.Tensor(y_test)
        train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)  # create your datset
        train_loader = DataLoader(train_dataset)  # create your dataloader
        test_loader = DataLoader(test_dataset)  # create your dataloader

        for epoch in range(10):  # loop over the dataset multiple times
            for i, x in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, l = x
                l = l.type(torch.LongTensor)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, l)
                loss.backward()
                optimizer.step()
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for x in test_loader:
                images, l = x
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += l.size(0)
                correct += (predicted == l).sum().item()
        sum_acc += (100 * correct / total)
    print(f"FC:Avg Acc:{sum_acc / KFOLD_N}%")


def tune_rf(data, labels):
    tree = RandomForestClassifier()
    args = {"n_estimators": [50, 100, 250, 500],
            "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2"]}
    gcv = GridSearchCV(tree, args, cv=KFOLD_N, scoring="accuracy")
    y = labels[:270]
    X = data[:270]
    gcv.fit(X, y)
    args_rf = gcv.best_params_
    print("RF Best Params: " + str(args_rf))
    return args_rf


def tune_knn(data, labels):
    knn = KNeighborsClassifier()
    args = {
        'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['euclidean', 'manhattan'], 'weights': ['uniform', 'distance']
    }
    gcv = GridSearchCV(knn, args, cv=KFOLD_N, scoring="accuracy")
    y = labels[:270]
    X = data[:270]
    gcv.fit(X, y)
    args_rf = gcv.best_params_
    print("KNN Best Params: " + str(args_rf))
    return args_rf


def print_cm(y, y_hat, name):
    # detailed score analysis
    cm = confusion_matrix(y, y_hat)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".1f")
    plt.title(name + " Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


if __name__ == "__main__":
    main()
