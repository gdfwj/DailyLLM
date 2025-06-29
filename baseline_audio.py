import os
import numpy as np
import pandas as pd

# ==== scikit-learn 相关 ====
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report  # 添加必要的导入

# ==== PyTorch 相关 ====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------------------------------------
# 1) 数据加载相关
# --------------------------------------------------------------------------------
def load_tut_dataset(csv_path):
    """
    从给定 csv_path 中读取数据：
      - 忽略第一列
      - 最后一列是 label
      - 中间列都是特征
    返回: X (ndarray shape=(N, d)), y_str (list或ndarray, shape=(N,))
    """
    df = pd.read_csv(csv_path, header=None)  # 如果文件里有表头，可以去掉 header=None 或加上适当的参数
    # 忽略第一列
    df.drop(columns=df.columns[0], inplace=True)
    
    # 最后一列是 label
    y_str = df.iloc[:, -1].values
    # 其余都是特征
    X = df.iloc[:, :-1].values
    
    return X, y_str

def encode_labels(y_str):
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    return y_int, le

# --------------------------------------------------------------------------------
# 2) 各传统ML模型的训练 & 测试 (sklearn)
# --------------------------------------------------------------------------------
def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_and_evaluate_dt(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_and_evaluate_xgb(X_train, y_train, X_test, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# --------------------------------------------------------------------------------
# 3) PyTorch 深度学习模型定义
#    （与之前相同，这里保留示例：1D-CNN / LSTM / CNN+LSTM / TCN / Transformer）
# --------------------------------------------------------------------------------

class Cnn1D(nn.Module):
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # 这里假设卷积后池化 2次 => 序列长度减到 seq_len//4
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
        # x: (B, in_channels, seq_len)
        # 先变成 (B, seq_len, in_channels)
        x = x.transpose(1, 2)
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class CnnLstmModel(nn.Module):
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.lstm_hidden = 100
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.lstm_hidden, batch_first=True)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.lstm_hidden, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # => (B, 128, seq_len//4)

        x = x.transpose(1, 2)  # => (B, seq_len//4, 128)
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]  # (B, 100)

        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class TCNModel(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.block1 = TCNBlock(in_channels, 64, kernel_size=3, dilation=1)
        self.block2 = TCNBlock(64, 64, kernel_size=3, dilation=2)
        self.block3 = TCNBlock(64, 64, kernel_size=3, dilation=4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = 64

        self.linear_embed = nn.Linear(in_channels, self.embedding_dim)
        self.mha = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4, batch_first=True)
        self.ln1 = nn.LayerNorm(self.embedding_dim)
        self.ff1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.ff2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.ln2 = nn.LayerNorm(self.embedding_dim)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.seq_len * self.embedding_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, seq_len, in_channels)
        x = self.linear_embed(x)  # => (B, seq_len, 64)

        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)

        ff = F.relu(self.ff1(x))
        ff = self.ff2(ff)
        x = self.ln2(x + ff)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --------------------------------------------------------------------------------
# 4) 训练 & 预测通用函数 (PyTorch)
# --------------------------------------------------------------------------------
def train_and_predict_nn(model,
                         X_train_np, y_train_np,
                         X_test_np,  y_test_np,
                         num_classes,
                         epochs=100, batch_size=32,
                         in_channels=1,
                         need_transpose=False):
    """
    通用的神经网络训练流程:
      - X_train_np / X_test_np: numpy数组, shape=(N, in_channels, seq_len) 或 shape=(N, seq_len) 等
      - 如果 X 的形状是 (N, feat), 可以先 .unsqueeze(1) => (N,1,feat), 并指定 need_transpose=False
      - need_transpose=True 时，会自动给 X 做个 permute(0,2,1)
    """
    model = model.to(device)

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test_np,  dtype=torch.long)

    # 如果需要转置，就转成 (N, in_channels, seq_len)
    if need_transpose:
        X_train_tensor = X_train_tensor.permute(0, 2, 1)
        X_test_tensor  = X_test_tensor.permute(0, 2, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = torch.utils.data.TensorDataset(X_test_tensor,  y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ----- 训练 -----
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

    # ----- 测试阶段 -----
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


# --------------------------------------------------------------------------------
# 5) 主逻辑: 这里示范如何加载 TUT2019 / TUT2016 数据集并跑一遍
# --------------------------------------------------------------------------------
def main():
    # 你自己的路径
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
    
    # 用于存放各数据集各模型的 accuracy
    columns = ["SVM", "RandomForest", "DecisionTree", "KNN", "XGBoost",
               "1D-CNN", "LSTM", "CNN+LSTM", "TCN", "Transformer"]
    results_df = pd.DataFrame(columns=columns)
    
    with open("tut_results.txt", "w") as f_out:  # 打开输出文件
        for dataset_name, train_csv, test_csv in tut_datasets:
            print(f"\n===== 处理数据集: {dataset_name} =====")
            f_out.write(f"===== DATASET: {dataset_name} =====\n")
            X_train, y_str_train = load_tut_dataset(train_csv)
            X_test,  y_str_test  = load_tut_dataset(test_csv)
            
            # 编码 label
            y_train_int, le = encode_labels(y_str_train)
            y_test_int = le.transform(y_str_test)
            
            print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
            
            # ========== (A) 传统ML模型 ==========

            # 定义一个函数来计算并写入结果
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

            # Random Forest
            rf_pred = train_and_evaluate_rf(X_train, y_train_int, X_test, y_test_int)
            evaluate_and_log("RandomForest", y_test_int, rf_pred)

            # Decision Tree
            dt_pred = train_and_evaluate_dt(X_train, y_train_int, X_test, y_test_int)
            evaluate_and_log("DecisionTree", y_test_int, dt_pred)

            # KNN
            knn_pred = train_and_evaluate_knn(X_train, y_train_int, X_test, y_test_int)
            evaluate_and_log("KNN", y_test_int, knn_pred)

            # XGBoost
            xgb_pred = train_and_evaluate_xgb(X_train, y_train_int, X_test, y_test_int)
            evaluate_and_log("XGBoost", y_test_int, xgb_pred)

            # ========== (B) 深度学习模型 ==========

            # 准备数据: (N, d) => (N, 1, d)
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
                need_transpose=False  # 我们已经是 (N,1,seq_len)，不需要再 permute
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

            # CNN+LSTM
            cnn_lstm_model = CnnLstmModel(seq_len=seq_len, num_classes=num_classes, in_channels=1)
            y_pred_cnn_lstm, y_true_cnn_lstm = train_and_predict_nn(
                cnn_lstm_model,
                X_train_nn, y_train_int,
                X_test_nn,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=1,
                need_transpose=False
            )
            evaluate_and_log("CNN+LSTM", y_true_cnn_lstm, y_pred_cnn_lstm)

            # TCN
            tcn_model = TCNModel(num_classes=num_classes, in_channels=1)
            y_pred_tcn, y_true_tcn = train_and_predict_nn(
                tcn_model,
                X_train_nn, y_train_int,
                X_test_nn,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=1,
                need_transpose=False
            )
            evaluate_and_log("TCN", y_true_tcn, y_pred_tcn)

            # Transformer
            trans_model = SimpleTransformer(seq_len=seq_len, num_classes=num_classes, in_channels=1)
            y_pred_trans, y_true_trans = train_and_predict_nn(
                trans_model,
                X_train_nn, y_train_int,
                X_test_nn,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=1,
                need_transpose=False
            )
            evaluate_and_log("Transformer", y_true_trans, y_pred_trans)

    print("\n===== 所有实验结束，结果已保存到 tut_results.txt =====")
    # results_df.loc[dataset_name, "SVM"]          = svm_acc
    # results_df.loc[dataset_name, "RandomForest"] = rf_acc
    # results_df.loc[dataset_name, "DecisionTree"] = dt_acc
    # results_df.loc[dataset_name, "KNN"]          = knn_acc
    # results_df.loc[dataset_name, "XGBoost"]      = xgb_acc
    # results_df.loc[dataset_name, "1D-CNN"]       = cnn_acc
    # results_df.loc[dataset_name, "LSTM"]         = lstm_acc
    # results_df.loc[dataset_name, "CNN+LSTM"]     = cnn_lstm_acc
    # results_df.loc[dataset_name, "TCN"]          = tcn_acc
    # results_df.loc[dataset_name, "Transformer"]  = transformer_acc
    
    # print("\n===== 所有实验结束，汇总 Accuracy 如下：=====")
    # print(results_df)
    # # 也可保存到 csv
    # results_df.to_csv("tut_results_accuracy.csv", index=True)
    print("已将结果保存到 tut_results_accuracy.csv")

if __name__ == "__main__":
    main()
