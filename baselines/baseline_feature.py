import os
import numpy as np
import pandas as pd

# ==== scikit-learn 相关 ====
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# ==== PyTorch 相关 ====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==== 一些工具函数 ====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


import ast

def parse_xyz_column(col_str):
    """
    将存储在 CSV 中的字符串(形如 "[[x0,y0,z0],[x1,y1,z1],...]")解析成 np.array(shape=(T,3))。
    解析失败返回 None。
    """
    try:
        arr_2d = ast.literal_eval(col_str)  # 将字符串解析为 list of lists
        arr_np = np.array(arr_2d, dtype=np.float32)
        if arr_np.ndim != 2 or arr_np.shape[1] != 3:
            return None
        return arr_np
    except:
        return None

def pad_sequences(seq_list, max_len=None, padding_value=0.0):
    """
    将一组 (L_i, feature_dim) 的时序用 0 填充到统一长度 max_len (或自动=所有中最大).
    返回 np.array shape=(batch, max_len, feature_dim)
    """
    if max_len is None:
        max_len = max(arr.shape[0] for arr in seq_list)
    feature_dim = seq_list[0].shape[1]
    batch_size = len(seq_list)

    out = np.full((batch_size, max_len, feature_dim), fill_value=padding_value, dtype=np.float32)
    for i, seq in enumerate(seq_list):
        length = seq.shape[0]
        out[i, :length, :] = seq
    return out


def load_raw_xyz_data(csv_path, train_size, val_size, random_state=42):
    """
    从原始xyz文件(含 gyro_data, accel_data, label) 里读取，并走你自定义的 split_by_label。
    最终返回: 
        X_train_pad, y_train_int, X_test_pad, y_test_int, 
        max_len (padding长度), num_classes
    """
    df = pd.read_csv(csv_path)

    # 先用你的 split_by_label 得到 train_df, test_df
    #   注意: split_by_label 返回的是 (train_df, val_df, test_df)，
    #   但你在当前脚本只用到了 (train_df, test_df) 这两个，就按需取。
    train_df, test_df = split_by_label(df, train_size=train_size, val_size=val_size, random_state=random_state)

    # 分别解析 train_df / test_df 里的原始数据
    X_train_list = []
    y_train_list = []
    for _, row in train_df.iterrows():
        gyro_np = parse_xyz_column(row["gyro_data"])
        accel_np= parse_xyz_column(row["accel_data"])
        if gyro_np is None or accel_np is None:
            continue
        length = min(len(gyro_np), len(accel_np))
        if length < 5:
            continue
        gyro_np  = gyro_np[:length]
        accel_np = accel_np[:length]
        combined = np.concatenate([gyro_np, accel_np], axis=0)  # (L,3)
        X_train_list.append(combined)
        y_train_list.append(str(row["label"]))

    X_test_list = []
    y_test_list = []
    for _, row in test_df.iterrows():
        gyro_np = parse_xyz_column(row["gyro_data"])
        accel_np= parse_xyz_column(row["accel_data"])
        if gyro_np is None or accel_np is None:
            continue
        length = min(len(gyro_np), len(accel_np))
        if length < 5:
            continue
        gyro_np  = gyro_np[:length]
        accel_np = accel_np[:length]
        combined = np.concatenate([gyro_np, accel_np], axis=0)  # (L,3)
        X_test_list.append(combined)
        y_test_list.append(str(row["label"]))

    # 对 label 做编码
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train_list)
    y_test_int  = le.transform(y_test_list)  # 注意：test集直接 transform

    # padding 到统一长度
    max_len = max(max(arr.shape[0] for arr in X_train_list), 
                  max(arr.shape[0] for arr in X_test_list)) if (X_test_list) else max(arr.shape[0] for arr in X_train_list)
    X_train_pad = pad_sequences(X_train_list, max_len=max_len, padding_value=0.0) # shape=(N, max_len,6)
    X_test_pad  = pad_sequences(X_test_list,  max_len=max_len, padding_value=0.0) # shape=(N, max_len,6)

    num_classes = len(le.classes_)

    return X_train_pad, y_train_int, X_test_pad, y_test_int, max_len, num_classes


# --------------------------------------------------------------------------------
# 1) 读取 CSV 并做基本预处理
# --------------------------------------------------------------------------------
def load_dataset(csv_path):
    """
    读取单个CSV，假设最后一列为label(字符串)，其余为特征(浮点)。
    返回: X(2D array), y_str(list of label strings)
    """
    df = pd.read_csv(csv_path)
    y_str = df.iloc[:, -1].values  # 最后一列是label
    X = df.iloc[:, :-1].values    # 其余列是特征
    return X, y_str

def encode_labels(y_str):
    """
    将字符串标签转换为数字 0,1,2,...
    返回: y_int (np.array, shape=(N,)), 以及label_encoder对象(可反查映射)
    """
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    return y_int, le

# --------------------------------------------------------------------------------
# 2) 各传统ML模型的训练 & 测试 (sklearn)
# --------------------------------------------------------------------------------
def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    model = LinearSVC(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
# 这里将每个模型写成一个 nn.Module 类
# --------------------------------------------------------------------------------

# ========== (A) 1D-CNN ==========

# ========== (A) 1D-CNN ==========

class Cnn1D(nn.Module):
    """
    输入 shape: (B, in_channels, seq_len)
    """
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.seq_len = seq_len
        # conv1d in_channels 变成可调
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # 假设 2次pool，长度 //4
        self.fc1 = nn.Linear(128 * (self.seq_len // 4), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape = (B, in_channels, seq_len)
        x = F.relu(self.conv1(x))   # => (B,64,seq_len)
        x = self.pool1(x)          # => (B,64,seq_len//2)
        x = F.relu(self.conv2(x))  # => (B,128,seq_len//2)
        x = self.pool2(x)          # => (B,128,seq_len//4)
        x = F.relu(self.conv3(x))  # => (B,128,seq_len//4)

        # flatten
        x = x.view(x.size(0), -1)  # => (B, 128*(seq_len//4))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # => (B, num_classes)
        return x


# ========== (B) LSTM ==========

class LSTMModel(nn.Module):
    """
    输入 shape: (B, in_channels, seq_len).
    先转成 (B, seq_len, in_channels), 再喂 LSTM(input_size=in_channels).
    """
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.hidden_size = 100
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape=(B, in_channels, seq_len)
        # 先转成 (B, seq_len, in_channels)
        x = x.transpose(1, 2)  # => (B, seq_len, in_channels)

        out, (h, c) = self.lstm(x)  # => out shape=(B, seq_len, hidden_size)
        # 取最后时刻
        out = out[:, -1, :]   # => (B, hidden_size)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ========== (C) CNN + LSTM ==========

class CnnLstmModel(nn.Module):
    """
    输入 shape: (B, in_channels, seq_len).
    先做1D Conv(多次) => shape=(B,128,seq_len//4)，再 permute=> (B, seq_len//4, 128) => LSTM => ...
    """
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
        # x shape = (B, in_channels, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # => (B,128,seq_len//4)

        # 送入 LSTM 前先 permute => (B, seq_len//4, 128)
        x = x.transpose(1, 2)
        out, (h, c) = self.lstm(x)
        # => (B, seq_len//4, hidden_size=100)
        out = out[:, -1, :]  # (B, 100)

        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ========== (D) TCN ==========

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        self.padding = dilation*(kernel_size-1)
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
    """
    输入 shape: (B, in_channels, seq_len)
    """
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        # 这里最初写死了 1->64，这里改为 in_channels->64
        self.block1 = TCNBlock(in_channels, 64, kernel_size=3, dilation=1)
        self.block2 = TCNBlock(64, 64, kernel_size=3, dilation=2)
        self.block3 = TCNBlock(64, 64, kernel_size=3, dilation=4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape=(B, in_channels, seq_len)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # => (B,64,seq_len左右)

        x = self.pool(x)  # => (B,64,1)
        x = x.squeeze(-1) # => (B,64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ========== (E) Transformer ==========

class SimpleTransformer(nn.Module):
    """
    输入 shape: (B, in_channels, seq_len).
    先转 (B, seq_len, in_channels) => 一个线性到 embedding_dim => MultiHeadAttention => ...
    """
    def __init__(self, seq_len, num_classes, in_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = 64

        # from in_channels -> embedding_dim
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
        # x shape=(B, in_channels, seq_len)
        x = x.transpose(1, 2)  # => (B, seq_len, in_channels)

        # 1) linear embed => (B, seq_len, 64)
        x = self.linear_embed(x)

        # 2) MHA
        attn_out, _ = self.mha(x, x, x)  # => (B, seq_len, 64)
        x = self.ln1(x + attn_out)

        # 3) FF
        ff = F.relu(self.ff1(x))
        ff = self.ff2(ff)
        x = self.ln2(x + ff)

        # 4) flatten
        x = self.flatten(x)  # => (B, seq_len*64)
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
                         epochs=10, batch_size=32, 
                         in_channels=1, 
                         need_transpose=False):
    """
    X_train_np / X_test_np 都是 numpy 数组.
      - UCI: shape=(N, feat)  => 先 unsqueeze(1) => (N,1,feat).
      - 原始xyz: shape=(N, max_len, 6) => 如果想直接(B,6,max_len), 可以先 permute(0,2,1).

    need_transpose: 如果外部还没转到 (B,in_channels,seq_len)，就在这里补一下
    """

    model = model.to(device)

    # 转成 tensor
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test_np,  dtype=torch.long)

    # 如果需要：把 (N, max_len, 6) => (N, 6, max_len)
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
        # 可加 print 看 loss

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



def split_by_label(df, train_size=500, val_size=500, random_state=42):
    """
    对DataFrame按'label'列分组。
    - 每个类别先打乱，
    - 取前 train_size 行放到train_df
    - 再取 val_size 行放到 val_df
    - 其余放到 test_df
    返回 train_df, val_df, test_df
    """
    train_list = []
    val_list   = []
    test_list  = []

    unique_labels = df['label'].unique()
    print(unique_labels)

    for lbl in unique_labels:
        if lbl=="rope":
            continue
        sub_df = df[df['label'] == lbl].copy()
        print(len(sub_df))
        # 打乱(sub_df是同一类)，random_state保证可复现
        sub_df = sub_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
        if train_size < 1:
            train_size = int(len(sub_df) * 0.9)
            val_size   = int(len(sub_df) * 0.1)

        # 根据需求切片
        train_part = sub_df.iloc[:train_size]
        val_part   = sub_df.iloc[train_size:train_size+val_size]
        test_part  = sub_df.iloc[:]

        train_list.append(train_part)
        val_list.append(val_part)
        test_list.append(test_part)
    
    # 合并
    train_df = pd.concat(train_list, ignore_index=True)
    val_df   = pd.concat(val_list,   ignore_index=True)
    test_df  = pd.concat(test_list,  ignore_index=True)

    return train_df, val_df

# --------------------------------------------------------------------------------
# 5) 主逻辑: 读取4个数据集, 训练 & 评估
# --------------------------------------------------------------------------------
def main():
    data_dir = "data/extracted"
    datasets = [
        "uci", 
        "HHAR_features.csv",
        "motion_features.csv",
        "pamap2_features.csv",
        "shoaib_features.csv"
    ]
    dataset_names = ["uci", "HHAR", "motion", "pamap2", "shoaib"]  # 用来做行名
    
    ml_model_names = ["SVM", "RandomForest", "DecisionTree", "KNN", "XGBoost"]
    dl_model_names = ["1D-CNN", "LSTM", "CNN+LSTM", "TCN", "Transformer"]
    all_model_names = ml_model_names + dl_model_names

    accuracy_results = pd.DataFrame(index=dataset_names, columns=all_model_names)
    
    with open("results_summary.txt", "w") as f_out:
        for idx, csv_file in enumerate(datasets):
            dataset_path = os.path.join(data_dir, csv_file)
            dataset_name = dataset_names[idx]
            f_out.write(f"===== DATASET: {dataset_name} =====\n")

            if dataset_name=="uci":
                X_train, y_str_train = load_dataset(os.path.join(data_dir, "uci_extract_all_train.csv"))
                X_test,  y_str_test  = load_dataset(os.path.join(data_dir, "uci_extract_all_test.csv"))
                y_train_int, label_encoder = encode_labels(y_str_train)
                y_test_int, _ = encode_labels(y_str_test)
            else:
                if dataset_name=="HHAR":
                    train_size = 500
                    val_size   = 500
                elif dataset_name=="motion":
                    train_size = 0.9
                    val_size   = 0.1
                elif dataset_name=="pamap2":
                    train_size = 300
                    val_size   = 300
                elif dataset_name=="shoaib":
                    train_size = 500
                    val_size   = 500
                print(dataset_path)
                train_df, test_df = split_by_label(pd.read_csv(dataset_path), train_size=train_size, val_size=val_size, random_state=42)
                X_train = []
                y_train = []
                for _, row in train_df.iterrows():
                    X_train.append(pd.to_numeric(row.iloc[0:52], errors='coerce').round(4).values.tolist())
                    y_train.append(row['label'])
                X_train = np.array(X_train)
                y_train_int, label_encoder = encode_labels(y_train)
                X_test = []
                y_test = []
                for _, row in test_df.iterrows():
                    X_test.append(pd.to_numeric(row.iloc[0:52], errors='coerce').round(4).values.tolist())
                    y_test.append(row['label'])
                X_test = np.array(X_test)
                y_test_int, _ = encode_labels(y_test)
                
                
            print(f"数据集 {dataset_name} 读取完成, 训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

            # ========== (A) 先跑 传统ML 模型 ==========
            svm_pred  = train_and_evaluate_svm(X_train, y_train_int, X_test, y_test_int)
            print("svm done")
            rf_pred   = train_and_evaluate_rf(X_train, y_train_int, X_test, y_test_int)
            print("rf done")
            dt_pred   = train_and_evaluate_dt(X_train, y_train_int, X_test, y_test_int)
            print("dt done")
            knn_pred  = train_and_evaluate_knn(X_train, y_train_int, X_test, y_test_int)
            print("knn done")
            xgb_pred  = train_and_evaluate_xgb(X_train, y_train_int, X_test, y_test_int)
            print("xgb done")

            svm_acc = accuracy_score(y_test_int, svm_pred)
            svm_f1  = f1_score(y_test_int, svm_pred, average="macro")
            svm_prec= precision_score(y_test_int, svm_pred, average="macro")
            svm_rec = recall_score(y_test_int, svm_pred, average="macro")
            svm_cm  = confusion_matrix(y_test_int, svm_pred)

            rf_acc  = accuracy_score(y_test_int, rf_pred)
            rf_f1   = f1_score(y_test_int, rf_pred, average="macro")
            rf_prec = precision_score(y_test_int, rf_pred, average="macro")
            rf_rec  = recall_score(y_test_int, rf_pred, average="macro")
            rf_cm   = confusion_matrix(y_test_int, rf_pred)

            dt_acc  = accuracy_score(y_test_int, dt_pred)
            dt_f1   = f1_score(y_test_int, dt_pred, average="macro")
            dt_prec = precision_score(y_test_int, dt_pred, average="macro")
            dt_rec  = recall_score(y_test_int, dt_pred, average="macro")
            dt_cm   = confusion_matrix(y_test_int, dt_pred)

            knn_acc = accuracy_score(y_test_int, knn_pred)
            knn_f1  = f1_score(y_test_int, knn_pred, average="macro")
            knn_prec= precision_score(y_test_int, knn_pred, average="macro")
            knn_rec = recall_score(y_test_int, knn_pred, average="macro")
            knn_cm  = confusion_matrix(y_test_int, knn_pred)

            xgb_acc = accuracy_score(y_test_int, xgb_pred)
            xgb_f1  = f1_score(y_test_int, xgb_pred, average="macro")
            xgb_prec= precision_score(y_test_int, xgb_pred, average="macro")
            xgb_rec = recall_score(y_test_int, xgb_pred, average="macro")
            xgb_cm  = confusion_matrix(y_test_int, xgb_pred)

            # === 写到结果文件中 ===
            f_out.write("\n----- Machine Learning Models -----\n")
            f_out.write(f"SVM  Accuracy: {svm_acc:.4f}, F1: {svm_f1:.4f}, "
                        f"Precision: {svm_prec:.4f}, Recall: {svm_rec:.4f}\n"
                        f"Confusion Matrix:\n{svm_cm}\n")

            f_out.write(f"RF   Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}, "
                        f"Precision: {rf_prec:.4f}, Recall: {rf_rec:.4f}\n"
                        f"Confusion Matrix:\n{rf_cm}\n")

            f_out.write(f"DT   Accuracy: {dt_acc:.4f}, F1: {dt_f1:.4f}, "
                        f"Precision: {dt_prec:.4f}, Recall: {dt_rec:.4f}\n"
                        f"Confusion Matrix:\n{dt_cm}\n")

            f_out.write(f"KNN  Accuracy: {knn_acc:.4f}, F1: {knn_f1:.4f}, "
                        f"Precision: {knn_prec:.4f}, Recall: {knn_rec:.4f}\n"
                        f"Confusion Matrix:\n{knn_cm}\n")

            f_out.write(f"XGB  Accuracy: {xgb_acc:.4f}, F1: {xgb_f1:.4f}, "
                        f"Precision: {xgb_prec:.4f}, Recall: {xgb_rec:.4f}\n"
                        f"Confusion Matrix:\n{xgb_cm}\n")

            accuracy_results.loc[dataset_name, "SVM"]          = svm_acc
            accuracy_results.loc[dataset_name, "RandomForest"] = rf_acc
            accuracy_results.loc[dataset_name, "DecisionTree"] = dt_acc
            accuracy_results.loc[dataset_name, "KNN"]          = knn_acc
            accuracy_results.loc[dataset_name, "XGBoost"]      = xgb_acc
            X_train_pad = X_train.reshape(X_train.shape[0], -1, 1)
            X_test_pad  = X_test.reshape(X_test.shape[0], -1, 1)
            max_len = len(X_train_pad[0])
            num_classes = len(set(y_train_int.tolist()))
            in_channels = 1
            seq_len     = max_len

            # 1) 1D-CNN
            cnn_model = Cnn1D(seq_len=seq_len, num_classes=num_classes, in_channels=in_channels)
            y_pred_int, y_true_int = train_and_predict_nn(
                cnn_model,
                X_train_pad, y_train_int,
                X_test_pad,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=in_channels,
                need_transpose=True 
            )
            cnn_acc = accuracy_score(y_true_int, y_pred_int)
            print("cnn done")

            # 2) LSTM
            lstm_model = LSTMModel(seq_len=seq_len, num_classes=num_classes, in_channels=in_channels)
            y_pred_int2, y_true_int2 = train_and_predict_nn(
                lstm_model,
                X_train_pad, y_train_int,
                X_test_pad,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=in_channels,
                need_transpose=True
            )
            lstm_acc = accuracy_score(y_true_int2, y_pred_int2)
            print("lstm done")

            # 3) CNN+LSTM
            cnn_lstm_model = CnnLstmModel(seq_len=seq_len, num_classes=num_classes, in_channels=in_channels)
            y_pred_int3, y_true_int3 = train_and_predict_nn(
                cnn_lstm_model,
                X_train_pad, y_train_int,
                X_test_pad,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=in_channels,
                need_transpose=True
            )
            cnn_lstm_acc = accuracy_score(y_true_int3, y_pred_int3)
            print("cnn+lstm done")

            # 4) TCN
            tcn_model = TCNModel(num_classes=num_classes, in_channels=in_channels)
            y_pred_int4, y_true_int4 = train_and_predict_nn(
                tcn_model,
                X_train_pad, y_train_int,
                X_test_pad,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=in_channels,
                need_transpose=True
            )
            tcn_acc = accuracy_score(y_true_int4, y_pred_int4)
            print("tcn done")

            # 5) Transformer
            transformer_model = SimpleTransformer(seq_len=seq_len, num_classes=num_classes, in_channels=in_channels)
            y_pred_int5, y_true_int5 = train_and_predict_nn(
                transformer_model,
                X_train_pad, y_train_int,
                X_test_pad,  y_test_int,
                num_classes=num_classes,
                epochs=30, batch_size=32,
                in_channels=in_channels,
                need_transpose=True
            )
            transformer_acc = accuracy_score(y_true_int5, y_pred_int5)
            print("transformer done")
            
            cnn_cm = confusion_matrix(y_true_int, y_pred_int)
            lstm_cm = confusion_matrix(y_true_int2, y_pred_int2)
            cnn_lstm_cm = confusion_matrix(y_true_int3, y_pred_int3)
            tcn_cm = confusion_matrix(y_true_int4, y_pred_int4)
            transformer_cm = confusion_matrix(y_true_int5, y_pred_int5)


            # ========== 记录深度学习结果到 txt & DF ==========
            f_out.write("\n----- Deep Learning Models (PyTorch) -----\n")
            f_out.write(f"1D-CNN Accuracy: {cnn_acc:.4f}\nConfusion Matrix:\n{cnn_cm}\n")
            cnn_f1   = f1_score(y_true_int, y_pred_int, average="macro")
            cnn_prec = precision_score(y_true_int, y_pred_int, average="macro")
            cnn_rec  = recall_score(y_true_int, y_pred_int, average="macro")
            f_out.write(f"1D-CNN F1: {cnn_f1:.4f}, Precision: {cnn_prec:.4f}, Recall: {cnn_rec:.4f}\n")

            f_out.write(f"LSTM Accuracy: {lstm_acc:.4f}\nConfusion Matrix:\n{lstm_cm}\n")
            lstm_f1   = f1_score(y_true_int2, y_pred_int2, average="macro")
            lstm_prec = precision_score(y_true_int2, y_pred_int2, average="macro")
            lstm_rec  = recall_score(y_true_int2, y_pred_int2, average="macro")
            f_out.write(f"LSTM F1: {lstm_f1:.4f}, Precision: {lstm_prec:.4f}, Recall: {lstm_rec:.4f}\n")

            f_out.write(f"CNN+LSTM Accuracy: {cnn_lstm_acc:.4f}\nConfusion Matrix:\n{cnn_lstm_cm}\n")
            cnn_lstm_f1   = f1_score(y_true_int3, y_pred_int3, average="macro")
            cnn_lstm_prec = precision_score(y_true_int3, y_pred_int3, average="macro")
            cnn_lstm_rec  = recall_score(y_true_int3, y_pred_int3, average="macro")
            f_out.write(f"CNN+LSTM F1: {cnn_lstm_f1:.4f}, Precision: {cnn_lstm_prec:.4f}, Recall: {cnn_lstm_rec:.4f}\n")

            f_out.write(f"TCN Accuracy: {tcn_acc:.4f}\nConfusion Matrix:\n{tcn_cm}\n")
            tcn_f1   = f1_score(y_true_int4, y_pred_int4, average="macro")
            tcn_prec = precision_score(y_true_int4, y_pred_int4, average="macro")
            tcn_rec  = recall_score(y_true_int4, y_pred_int4, average="macro")
            f_out.write(f"TCN F1: {tcn_f1:.4f}, Precision: {tcn_prec:.4f}, Recall: {tcn_rec:.4f}\n")

            f_out.write(f"Transformer Accuracy: {transformer_acc:.4f}\nConfusion Matrix:\n{transformer_cm}\n")
            transformer_f1   = f1_score(y_true_int5, y_pred_int5, average="macro")
            transformer_prec = precision_score(y_true_int5, y_pred_int5, average="macro")
            transformer_rec  = recall_score(y_true_int5, y_pred_int5, average="macro")
            f_out.write(f"Transformer F1: {transformer_f1:.4f}, Precision: {transformer_prec:.4f}, Recall: {transformer_rec:.4f}\n")

            accuracy_results.loc[dataset_name, "1D-CNN"]      = cnn_acc
            accuracy_results.loc[dataset_name, "LSTM"]        = lstm_acc
            accuracy_results.loc[dataset_name, "CNN+LSTM"]    = cnn_lstm_acc
            accuracy_results.loc[dataset_name, "TCN"]         = tcn_acc
            accuracy_results.loc[dataset_name, "Transformer"] = transformer_acc

            f_out.write("\n========================================\n\n")
            print(f"Dataset {dataset_name} 完成。")
    # 最后把 accuracy_results 写出到 CSV
    accuracy_results.to_csv("results_accuracy.csv")
    print("所有实验完成。结果已写入 results_summary.txt 和 results_accuracy.csv。")

if __name__ == "__main__":
    main()
