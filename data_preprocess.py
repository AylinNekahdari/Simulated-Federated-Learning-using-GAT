import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
try:
    from torch_geometric.utils import from_scipy_sparse_array as from_scipy
except ImportError:
    from torch_geometric.utils import from_scipy_sparse_matrix as from_scipy


def load_nsl_kdd(train_path, test_path):
    cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
        'root_shell','su_attempted','num_root','num_file_creations','num_shells',
        'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
        'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
        'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
    ]

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    train_df.columns = cols + ['difficulty']
    test_df.columns = cols + ['difficulty']
    train_df.drop(columns=['difficulty'], inplace=True)
    test_df.drop(columns=['difficulty'], inplace=True)
    df = pd.concat([train_df, test_df], ignore_index=True)
    return df


def preprocess_data(df):
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    cat_features = ['protocol_type', 'service', 'flag']
    for c in cat_features:
        df[c] = LabelEncoder().fit_transform(df[c])
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pyg_data(X_np, y_np, k=5):
    A = kneighbors_graph(X_np, k, mode='connectivity', include_self=True)
    edge_index, _ = from_scipy(A)
    x = torch.tensor(X_np, dtype=torch.float)
    y = torch.tensor(y_np, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)
