import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_tut_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.drop(columns=df.columns[0], inplace=True)
    
    y_str = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    
    return X, y_str

def encode_labels(y_str):
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    return y_int, le

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


class Cnn1D(nn.Module):
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * (self.seq_len // 4), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   
        x = self.pool1(x)          
        x = F.relu(self.conv2(x))  
        x = self.pool2(x)          
        x = F.relu(self.conv3(x))  

        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.hidden_size = 100
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def train_and_predict_nn(model,
                         X_train_np, y_train_np,
                         X_test_np,  y_test_np,
                         num_classes,
                         epochs=100, batch_size=32,
                         in_channels=1,
                         need_transpose=False):
    model = model.to(device)

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test_np,  dtype=torch.long)

    if need_transpose:
        X_train_tensor = X_train_tensor.permute(0, 2, 1)
        X_test_tensor  = X_test_tensor.permute(0, 2, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = torch.utils.data.TensorDataset(X_test_tensor,  y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            y_pred_list.append(preds.cpu().numpy())
            y_true_list.append(batch_y.cpu().numpy())

    y_pred_int = np.concatenate(y_pred_list, axis=0)
    y_true_int = np.concatenate(y_true_list, axis=0)
    return y_pred_int, y_true_int


def main():
    data_dir = "data/extracted"
    
    tut_datasets = [
        ("tut2019", 
         os.path.join(data_dir, "train_features_tut2019.csv"), 
         os.path.join(data_dir, "test_features_tut2019.csv")),
        ("tut2018", 
         os.path.join(data_dir, "train_features_tut2018.csv"), 
         os.path.join(data_dir, "test_features_tut2018.csv")),
        ("tut2017", 
         os.path.join(data_dir, "train_features_tut2017.csv"), 
         os.path.join(data_dir, "test_features_tut2017.csv")),
        ("tut2016", 
         os.path.join(data_dir, "train_features_tut2016.csv"), 
         os.path.join(data_dir, "test_features_tut2016.csv")),
    ]
    
    columns = ["SVM", "KNN",
               "1D-CNN", "LSTM"]
    results_df = pd.DataFrame(columns=columns)
    
    with open("tut_results.txt", "w") as f_out:
        for dataset_name, train_csv, test_csv in tut_datasets:
            f_out.write(f"===== DATASET: {dataset_name} =====\n")
            X_train, y_str_train = load_tut_dataset(train_csv)
            X_test,  y_str_test  = load_tut_dataset(test_csv)
            
            y_train_int, le = encode_labels(y_str_train)
            y_test_int = le.transform(y_str_test)
            
            def evaluate_and_log(model_name, y_true, y_pred):
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average="macro")
                rec = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")
                cm = confusion_matrix(y_true, y_pred)
                f_out.write(f"{model_name} Accuracy: {acc:.4f}\n")
                f_out.write(f"{model_name} Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}\n")
                f_out.write(f"{model_name} Confusion Matrix:\n{cm}\n\n")
                print(f"[{model_name}] Acc = {acc:.4f}, Prec = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}")

            # SVM
            svm_pred = train_and_evaluate_svm(X_train, y_train_int, X_test, y_test_int)
            evaluate_and_log("SVM", y_test_int, svm_pred)

            # KNN
            knn_pred = train_and_evaluate_knn(X_train, y_train_int, X_test, y_test_int)
            evaluate_and_log("KNN", y_test_int, knn_pred)

            X_train_nn = X_train[:, np.newaxis, :]  # shape=(N,1,d)
            X_test_nn  = X_test[:, np.newaxis, :]   # shape=(N,1,d)
            seq_len = X_train_nn.shape[2]
            num_classes = len(le.classes_)

            # 1D-CNN
            cnn_model = Cnn1D(seq_len=seq_len, num_classes=num_classes, in_channels=1)
            y_pred_cnn, y_true_cnn = train_and_predict_nn(
                cnn_model,
                X_train_nn, y_train_int,
                X_test_nn,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=1,
                need_transpose=False 
            )
            evaluate_and_log("1D-CNN", y_true_cnn, y_pred_cnn)

            # LSTM
            lstm_model = LSTMModel(seq_len=seq_len, num_classes=num_classes, in_channels=1)
            y_pred_lstm, y_true_lstm = train_and_predict_nn(
                lstm_model,
                X_train_nn, y_train_int,
                X_test_nn,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=1,
                need_transpose=False
            )
            evaluate_and_log("LSTM", y_true_lstm, y_pred_lstm)

if __name__ == "__main__":
    main()
