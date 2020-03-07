def DTW(a,b,d=lambda x,y:abs(x-y)):
    #mww:max warping window
    m,n = len(a),len(b)
    cost = np.zeros([m,n])
    cost[0,0] = d(a[0],b[0])
    for i in range(1,m):
        cost[i,0] = cost[i-1,0] + d(a[i],b[0]])
    for j in range(1,n):
        cost[0,j] = cost[0,j-1] + d(a[0],b[j])
    for i in range(1,m):
        for j in range(1,n):
            choices = cost[i-1,j-1],cost[i-1,j],cost[i,j-1]
            cost[i][j] = d(a[i],b[j]) + min(choices)
    return cost[-1,-1]
