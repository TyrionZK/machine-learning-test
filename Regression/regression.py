from numpy import *


def load_data_set(filename):
    num_feat = len(open(filename).readline().split('\0'))-1
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        current_line = line.split('\0')
        for i in range(num_feat):
            line_arr.append(float(current_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(current_line[-1]))
    return data_mat,label_mat


def stand_regression(x_arr, y_arr):
    x_mat = mat(x_arr); y_mat = mat(y_arr).T
    xTx = x_mat.T*xMat
    if linalg.det(xTx) == 0:
        print("This matrix is singluar, cannot do inverse");
        return
    ws = xTx.I*(x_mat.T*y_mat)
    return ws


def lwlr(test_point, x_arr, y_arr,k=1):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m = shape(x_mat)[0]
    weights = mat(eye((m)))
    for i in range(m):
        diff_mat = test_point - x_mat[i, :]
        weights[i, i] = exp(diff_mat*diff_mat.T/(-2*k**2))
    xTx = x_mat.T*weights*x_mat
    if linalg.det(xTx) == 0:
        print("this matrix is singluar, cannot do inverser");
        return
    ws = xTx.I*x_mat.T*weights*y_mat
    return ws


def lwl_test(test_arr, x_arr, y_arr, k=1):
    m = shape(test_arr)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i, :], x_arr, y_arr, k)
    return y_hat
