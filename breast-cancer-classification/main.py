import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame
print(df)

X = df.drop(columns=['target']).values
y = df['target'].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean)/X_std  

m, n = X.shape
idx = np.random.permutation(m)
train_size = int(0.8*m)
X_train, X_test = X[idx[:train_size]], X[idx[train_size:]]
y_train, y_test = y[idx[:train_size]], y[idx[train_size:]]

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def compute_gradient(X, y, w, b, lambda_):
    m = X.shape[0]
    g_wb = np.dot(X, w) + b
    f_wb = sigmoid(g_wb)
    error = f_wb - y
    dj_dw = (1/m)*np.dot(X.T, error) + (lambda_/m)*w
    dj_db = (1/m)*np.sum(error)
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, iters, lambda_):
    m = X.shape[0]
    loss_prev = 0
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)
        lr = alpha*(0.9**(i//1000))
        w = w - lr*dj_dw
        b = b - lr*dj_db

        if(i%100 == 0):
            g_wb = np.dot(X, w) + b
            f_wb = sigmoid(g_wb)
            epsilon = 1e-15
            f_wb = np.clip(f_wb, epsilon, 1 - epsilon)

            loss = -np.mean(y*np.log(f_wb) + (1-y)*np.log(1 - f_wb)) + (lambda_/(2*m))*np.sum(w**2)
            print(f"Epoch {i}: Loss = {loss}")

            if(abs(loss - loss_prev) < 0.00001 and i>100):
                break
            loss_prev = loss
    return w, b

m_in, n_in = X_train.shape
w_in = np.zeros(n_in)
b_in = 0
alpha = 0.001
it = 100000
lambda_ = 0.05

w, b = gradient_descent(X_train, y_train, w_in, b_in, alpha, it, lambda_)
y_pred = sigmoid(np.dot(X_test, w) + b)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred_class = (y_pred >= 0.4).astype(int)

accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")