def unboundedKnapsack_DP2(W, n, val, wt):
    K = [0 for i in range(W + 1)]

    for x in range(W + 1):
        for i in range(n):
            if (wt[i] <= x):
                K[x] = max(K[x], K[x - wt[i]] + val[i])
    #     print(K)
    return (K)


def backtracking(K, W, n, val, wt):
    l = []
    current_capacity = W
    for i in range(n):
        while current_capacity - wt[i] >= 0 and K[current_capacity - wt[i]] + val[i] == K[current_capacity]:
            l.append(i)
            current_capacity = current_capacity - wt[i]

    return (l)