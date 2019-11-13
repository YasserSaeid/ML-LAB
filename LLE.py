def lle(X, m, tol, k=5, n_rule='knn', epsilon=1.):
    n = X.shape[0]
    #1) check parameters
    if tol <= 0:
        raise ValueError('Tolerance (tol) must be > 0.\n')
    if m < 1 or m > X.shape[1]:
        raise ValueError('Number of eigenvectors (m) not within the range (1,%d).\n' %X.shape[1])
    if k < 1 or k > n:
        raise ValueError('Number of k nearest neighbors (k) not within the range (1, %d).\n' %n)
    if (n_rule != 'knn') and (n_rule != 'eps-ball'):
        raise ValueError('Only n_rules knn and eps-ball are supported.\n')
    if (n_rule == 'eps-ball') and (epsilon <= 0):
        raise ValueError('The radius (epsilon) must be > 0.\n')
        
    #2) get neighbouring indices by either knn or epsilon ball
    #2.1) calculate distance matrix 
    dist = scipy.spatial.distance.pdist(np.asmatrix(X), 'euclidean')
    dist = scipy.spatial.distance.squareform(dist)
    #set diagonal distance to infinite as it is the distance of the point to itself 
    np.fill_diagonal(dist, float('inf'), wrap=False)
    
    #2.2) get neighbouring indices
    if n_rule == 'knn':
        #sort the indices of all points by distance descending -take out k indices
        idx = np.argsort(dist)[:,:k]
        #print('knn idx:\n', idx)
    elif n_rule == 'eps-ball':
        #do eps-ball
        #find indices of all elements of the distance matrix which are lower than epsilon
        mask = dist <= epsilon
        #all distances which are not < epsilos are set to zero
        erg = dist*mask
        #print('erg:\n', erg)
        tree = scipy.spatial.KDTree(X)
        idx = tree.query_ball_point(X, epsilon)
        #print('eps-ball idx:\n', idx)
        
    #3)find reconstruction weights for each datapoint
    #initialize reconstruction matrix
    W = np.zeros(shape=(n,n))
    #for each datapoint - calculate weigths    
    for j, point in enumerate(idx):
        #for each datapoint - get the neighbouring points 
        neighbours = []
        for i in range(len(point)):
            #center around datapoint
            neighbours.append(np.asarray(X[point[i]])-np.asarray(X[j]))
        #calculate covariance matrix of centered neighbours
        C = np.dot(np.asmatrix(neighbours),np.asmatrix(neighbours).T)  
        #calculating weights
        a = C + tol*np.identity(len(point))
        w,u,t,v = np.linalg.lstsq(a,np.ones(shape=(len(point),1)))
        #normalizing
        w = w/sum(w)
        
        #set the previously calculated weights into the associated row/column of the reconstruction matrix W
        for l in range(len(point)):
            W[j][point[l]] = w[l]       
        
    #4)calculating matrix M 
    M = np.identity(n)-W
    M = np.dot(M.T,M)
    
    #5)getting the new coordinates as the m smallest eigenvectors of M
    evals,evecs = np.linalg.eigh(M)    
    Y = evecs[:,1:m+1]    
    return Y
