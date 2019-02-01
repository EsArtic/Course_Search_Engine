import math
import numpy as np
import matplotlib.pyplot as plt

def svd(A):
    A1 = np.dot(A.T, A)
    A2 = np.dot(A, A.T)
    e1, V = np.linalg.eig(A1)
    e2, initial_U = np.linalg.eig(A2)
    U = np.zeros(initial_U.shape)
    U[:, :] = initial_U[:, :]
    S = np.zeros(A.shape)
    n, m = A.shape
    for i in range(len(e1)):
        if i == n or i == m:
            break

        e = e1[i]
        if e > 0:
            S[i, i] = math.sqrt(e)

    return U, S, V

def display(M):
    if len(M.shape) > 1:
        n, m = M.shape
        for i in range(n):
            for j in range(m):
                if abs(M[i, j]) < 0.00005:
                    print(' 0.0000', end = ' ')
                elif M[i, j] >= 0.0:
                    print(' %.4f' % M[i, j], end = ' ')
                else:
                    print('%.4f' % M[i, j], end = ' ')
            print()
        print()
    else:
        for i in range(M.shape[0]):
            if M[i] >= 0.0:
                print(' %.4f' % M[i], end = ' ')
            else:
                print('%.4f' % M[i], end = ' ')
        print('\n')

def main():
    A = np.array([[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 1],
                  [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]])
    q = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1])
    U, S, V = svd(A.T)
    print('Problem (a):')
    print('Matrix U:')
    display(U)
    print('Matrix Sigma:')
    display(S)
    print('Matrix V:')
    display(V)

    estimate_A = np.dot(np.dot(U, S), V.T)
    print('Problem (b):')
    print('Matrix A\' = U * Sigma * V.T:')
    display(estimate_A)

    print('Score for A:')
    for i in range(A.shape[0]):
        di = A[i, :]
        score = np.dot(q, di)
        print('Document %d: %.4f' % (i + 1, score))
    print()
    print('Score for A\':')
    for i in range(estimate_A.shape[1]):
        di = estimate_A[:, i]
        score = np.dot(q, di)
        print('Document %d: %.4f' % (i + 1, score))
    print()

    S2 = np.zeros((2, 2))
    S2[0:2, 0:2] = S[0:2, 0:2]
    U2 = np.zeros((U.shape[0], 2))
    U2[:, 0:2] = U[:, 0:2]
    V2 = np.zeros((V.shape[0], 2))
    V2[:, 0:2] = V[:, 0:2]
    estimate_A2 = np.dot(np.dot(U2, S2), V2.T)
    print('Problem (c):')
    print('Matrix U2:')
    display(U2)
    print('Matrix Sigma2:')
    display(S2)
    print('Matrix V2:')
    display(V2)
    print('Rank-2 approximate of A:')
    display(estimate_A2)

    print('problem (d):')
    # qk = Sigma_k^-1 * U_k^t * q
    print('mapped qk:')

    S_inv_2 = np.linalg.inv(S2)
    temp_U2 = U2[:, 0:2]
    qk = np.dot(np.dot(S_inv_2.T, temp_U2.T), q)
    display(qk)

    print('mapped d1:')
    d1 = estimate_A2[:, 0]
    d1_k = np.dot(np.dot(S_inv_2.T, temp_U2.T), d1)
    display(d1_k)

    print('mapped d2:')
    d2 = estimate_A2[:, 1]
    d2_k = np.dot(np.dot(S_inv_2.T, temp_U2.T), d2)
    display(d2_k)

    print('mapped d3:')
    d3 = estimate_A2[:, 2]
    d3_k = np.dot(np.dot(S_inv_2.T, temp_U2.T), d3)
    display(d3_k)

    x = np.array([qk[0], d1_k[0], d2_k[0], d3_k[0]])
    y = np.array([qk[1], d1_k[1], d2_k[1], d3_k[1]])
    plt.grid(True)
    plt.scatter(x, y, c = 'r')
    plt.scatter([0], [0], s = 5)
    plt.text(qk[0] - 0.03, qk[1], 'qk')
    plt.text(d1_k[0] - 0.03, d1_k[1], 'd1')
    plt.text(d2_k[0] - 0.03, d2_k[1], 'd2')
    plt.text(d3_k[0] - 0.03, d3_k[1], 'd3')
    plt.annotate("", xy=(qk[0], qk[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(d1_k[0], d1_k[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(d2_k[0], d2_k[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(d3_k[0], d3_k[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->"))
    plt.show()

    print('Problem (e):')
    print('Score for Rank-2 approximate:')
    container = [d1_k, d2_k, d3_k]
    for i in range(len(container)):
        score = np.dot(qk, container[i])
        print('Document %d: %.4f' % (i + 1, score))

if __name__ == '__main__':
    main()