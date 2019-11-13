def kmeans(X, k, max_iter=100):
    # n = amount of data points / d = dimension of data points
    n = X.shape[0]
    # k random indices
    randIdx = random.sample(range(n), k)
    # choose random data points as initial cluster centers
    mu = copy.copy(X[randIdx,:])
    # initalize other variables
    r = np.zeros(n, dtype=int)
    rNew = np.zeros(n, dtype=int)
    i = 0
    
    while i < max_iter:
        
        # minimization of the loss function: "assignment step"
        # calculates distance matrix
        dist = scipy.spatial.distance.cdist(X, mu, 'euclidean')
        # set cluster membership refering to clostest distance
        rNew = np.argsort(dist)[:,:1].reshape(n)

        # minimization of the loss function: "update step"
        for j in range(k):
            # select indices for points from cluster j
            idx = np.where(rNew == j)
            # calculate the mean over the cluster members, wich represents the new cluster center
            mu[j] = X[idx[0],:].mean(axis=0)
            # if a cluster becomes empty, take a random point as cluster center and continue
            if len(idx[0]) == 0:
                # 1 random index
                randIdx = random.sample(range(n), 1)
                # choose random data point as initial cluster centers for empty cluster
                mu[j] = copy.copy(X[randIdx,:])
                # adjust rNew vector to new cluster center
                rNew[randIdx] = j
                print('The computation resulted in an empty cluster. '
                      'A random point is assigned to the empty cluster.\n')

        # compute and print output
        i = i+1
        print('Iteration no. %d' %i)
        num = np.sum(r != rNew)
        print('Cluster memberships changed: %d' %num)
        # loss = sum of the distances of datapoints to their respective cluster centre
        # sort distances ascending
        sortedDist = np.sort(dist)
        #take only the first distance for the loss calculation
        loss = sum(sortedDist[:,0])
        print('Loss function value: %.2f\n' %loss)
        # stop computing if no data point label changed any more
        if (r == rNew).all():
            break
        # assign new labels to r
        r = copy.copy(rNew)
    
    return mu, r, loss


