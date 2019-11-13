def kmeans_agglo(X, r):
    # amount of data points
    n = len(r)
    # amount of clusters
    k = len(np.unique(r))
    # initialize cluster with data points X as cluster centers mu
    # X needs to be copied because mu is altered within the for loop, but X shouldnt be affected by this
    mu = copy.copy(X)
    mergeidx = np.zeros((k-1, 2), dtype=int)
    kmloss  = np.zeros(k, dtype=float)
    R = np.zeros((k-1, n), dtype=int)
    # "The first row of R is the initial clustering r"
    R[0] = copy.copy(r)    
    
    
    def computeClusterCenter(r):
        # Compute set C that contains the remaining cluster indices in r.
        C = np.unique(r)
        # compute new cluster center depending on labels in r (labels in C)
        for j in C:
            # find all indices of data points with label j
            idxAllPoints = np.where(r == j)[0]
            # compute mean of given data points
            # cluster label of the points are now the indices of the cluster centers in mu
            # "For each cluster 1, ..., k, its centre is the mean over its members"
            mu[j] = np.mean(X[idxAllPoints, :], axis=0)
        # set all remaining entries of mu to inf
        indices = [o for p, o in enumerate(list(range(n))) if p not in C]
        mu[indices] = np.inf
        
        return mu
    
    
    def computeClusteringCost(mu):
        # "kmloss (...) which contains the loss function value after each agglomeration
        # step, where the first entry is the loss of the initial clustering r."
        
        # Loss funtion (PCdist = point cluster distance)
        # same as in kmeans
        PCdist = scipy.spatial.distance.cdist(X, mu, 'euclidean')
        # fill_diagonal can be used because the PCdist matrix is symmetric
        sortedDist = np.sort(PCdist)
        kmloss = sum(sortedDist[:,0])        
        
        return kmloss
    
    
    def rVector(r, mergeIdx):
        # the indices of mu are used as labels in r
        #
        # find all indices which belong to data points from cluster C0 / C1
        idxAllPointsC1 = np.where(r == mergeIdx[0])[0]
        # assign points with a high label to points in cluster with low label
        r[idxAllPointsC1] = mergeIdx[1]

        return r
        

    def findBestMergeidx(possibleClusterCombinations):
        # Find best merge indices / minimize the cost function
        # Find the two cluster indices c1, c2 ∈ C such that if we merge the clusters c1 and c2
        # the cost function l(X, r(c1=c2)) is minimal among all possible mergers

        # initialize output kmloss
        possible_kmloss = float('inf')
        # check all possible cluster center combinations for kmloss
        for l in range(len(possibleClusterCombinations)):
            # pick cluster combination out of list
            thisMergeidx = [max(possibleClusterCombinations[l]), min(possibleClusterCombinations[l])]
            # save vector r in "thisr", so the computation doesn't change the original labels
            thisr = copy.copy(r)
            # set the resulting labels from the vector and the merge
            thisr = rVector(thisr, thisMergeidx)
            # recompute cluster centers
            mu = computeClusterCenter(thisr)
            # recompute the value of kmloss
            thisKmloss = computeClusteringCost(mu)
            # if the loss is bette (lower) replace the variables with the current ones
            if  thisKmloss < possible_kmloss:
                bestr = copy.copy(thisr)
                possible_kmloss = copy.copy(thisKmloss)
                mergeIdx = copy.copy(thisMergeidx)
            #print('\n\nIteration no. %d' %i, 'Cluster merge no. %d' %l)
            #print('possible kmloss:', possible_kmloss)
            #print('this merge:', thisMergeidx, '\n',
            #  'best merge:', mergeIdx, '\n',
            #  'this r:', thisr, '\n',
            #  'best r:', bestr, '\n',
            #  'this kmloss:', thisKmloss, '\n',
            #  'best kmloss', possible_kmloss, '\n')

        return bestr, possible_kmloss, mergeIdx
        
        
    # compute initial cluster centers
    mu = computeClusterCenter(r)
    # compute initial clustering cost
    kmloss[0] = computeClusteringCost(mu)        
        
    for i in range(k-1):
        # Compute set C that contains the remaining cluster indices in r.
        C = np.unique(r)
        # list of all possible combination of cluster
        possibleClusterCombinations = list(itertools.combinations(C, 2))
        # call function to get best mergeidx
        bestr, possible_kmloss, mergeidx[i] = findBestMergeidx(possibleClusterCombinations)
        # assign values (mergeidx[i] already is):
        kmloss[i+1] = copy.copy(possible_kmloss)
        # save a copy of the "best" label vector in r and R[i+1]
        r = copy.copy(bestr)
        if i < k-2:
            R[i+1] = copy.copy(bestr)

    return R, kmloss, mergeidx



##############################################################################################
# Assignment 3

def agglo_dendro(kmloss, mergeidx):

# The distance between clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. 
# The fourth value ``Z[i, 3]`` represents the number of original observations 
# in the newly formed cluster.

    # This function counts all merged clusters (data points). 
    # The returned value corresponds with Z[i, 3] from the linkage function
    def countClusters(position, mergeIndices):
        # value list is initialized
        valueList = list(mergeIndices[position].flatten())
        # go through mergeidx to find all subcluster
        for i in range(position,-1,-1):
            if set(valueList).isdisjoint(mergeIndices[i]) == False:
                # add new cluster to the list
                valueList = list(valueList) + list(mergeIndices[i])
                # delete multiple occurring values
                valueList = np.unique(valueList)
        # the legth of the list is the amount of merged clusters
        amountCluster = len(valueList)
        return amountCluster
    
    # prepare own Z array for dendogram function
    ownZ = np.zeros((len(mergeidx), 4))
    # prepare mergeidx array for dendrogram function
    newMergeIndices = copy.copy(mergeidx)
    # iterate through all cluster merges
    for i in range(len(newMergeIndices)):
        # alter the merge indices according to the input of the dendrogram function 
        # (Z[i, 0] and Z[i, 1])
        while (list(newMergeIndices[:i].flatten()).count(newMergeIndices[i][1]) > 0) or (newMergeIndices[i][1] == newMergeIndices[i][0]):
            # increase label if they already occure in the array
            newMergeIndices[i][1] = newMergeIndices[i][1]+1
        while (list(newMergeIndices[:i].flatten()).count(newMergeIndices[i][0]) > 0) or (newMergeIndices[i][1] == newMergeIndices[i][0]):
            # increase label if they already occure in the array
            newMergeIndices[i][0] = newMergeIndices[i][0]+1
        # call function for Z[i, 3]
        numberClusterMerges = countClusters(i, mergeidx[:i+1])
        # assign values to ownZ
        ownZ[i] = [newMergeIndices[i][1], newMergeIndices[i][0], kmloss[i], numberClusterMerges]
    
    # plot dendrogram using the scipy function
    dendrogram(ownZ)
    plt.ylabel('Loss function value')
    plt.title('Dendrogram initialized with K = %d clusters' %(len(mergeidx)+1))
    plt.show()
    
    # THE FOLLOWING NEEDS TO BE APPLIED:
    # "two lines joined together at a height corresponding to the increase in the clustering loss function."
    
    
    
##############################################################################################
# Assignment 4

def norm_pdf(X, mu, C):
     
    try:
        n, d = X.shape
    except: 
        n = 1
        d = len(X)
    C = np.asmatrix(C)+np.identity(d)*1e-15
     
    X_c = X - np.asarray(mu)
    c_inv = lin.pinv(C)
    norm_const = 1.0/(np.power(2.0*np.pi, d/2.0)*np.sqrt(lin.det(C)))
     
    exponent = np.diagonal(-0.5*np.dot(X_c, np.dot(c_inv, X_c.T)))
    p2 = np.exp(exponent)
       
    return norm_const*p2



##############################################################################################
# Assignment 5

def em_gmm(X, k, max_iter = 100, init_kmeans = False, tol = 1e-5):
     
    #initialization
    n, d = X.shape
    sigma = [np.identity(d)]*k
    pi = [1/k]*k
     
    if init_kmeans is False:     
        mu = np.asmatrix(random.sample(list(X),k))
    else: 
        mu, r, kloss = kmeans(X, k)
        mu = np.asmatrix(list(mu))

    step = 1
    loglike_old = 0
    gamma = np.zeros((n,k))
    
    
    while step <= max_iter:
         
        ###########################################E-Step  
        #variant as in presentation ---> definetly wrong (yet)        
        #for i in range (n):
        #    for j in range (k):
        #        X_c = X[i]-mu[j]
        #        log_gamma[i,j] = np.log(pi[j]) - 0.5*((np.log(lin.det(sigma[j]))+np.dot(X_c, np.dot(lin.pinv(sigma[j]), X_c.T)))) 
        #gamma = np.exp(log_gamma/misc.logsumexp(log_gamma,axis=0))
         
        #variant as in guide 
        a = []
        for i in range(n):
            for j in range(k):
                a.append(pi[j]*norm_pdf(X[i], mu[j], np.asmatrix(sigma[j]).reshape(d,d)))
        gamma_up = np.asarray(a).reshape(n,k)
        gamma_down = np.sum(gamma_up, axis = 1)
        gamma = gamma_up/gamma_down.reshape(n,1) 
         
        ###########################################M-Step
        bigN = np.sum(gamma, axis = 0)
        pi = bigN/n
         
        #estimating mu
        for cluster in range(k):
            mu[cluster]=(np.sum((gamma[:,cluster]*X.T).T,axis=0))
        mu = mu/bigN.reshape(k,1)
         
        #estimating sigma
        data_centered_mu_K = (np.asarray(X)[:, None] - np.asarray(mu))
        unweighted_cov_summands =(data_centered_mu_K.reshape(k*n,d,1)*data_centered_mu_K.reshape(k*n,1,d)).reshape(n,k,d*d)
        weighted_cov_summands = unweighted_cov_summands*np.repeat(gamma, d*d).reshape(n,k,d*d)
        sigma = np.sum(weighted_cov_summands, axis=0)
        sigma = sigma.reshape(k,d,d)/bigN.reshape(k,1,1)
         
        ############################################Log Likelihood
        #log likeliehood:       
        loglike = np.zeros((n,k))
        for i in range(n):
            for j in range(k):
                loglike[i,j] = pi[j]*norm_pdf(X[i],mu[j],np.asmatrix(sigma[j]).reshape(d,d))
        loglike = np.sum(np.log(np.sum(np.asarray(loglike).reshape(n,k), axis = 1)))      
         
        #print step and LOG LIKELIHOOD
        print('Iteration no.', step, 'Log likelihood:', loglike)
        step += 1
         
        #check for convergence
        if np.abs(loglike_old-loglike)<tol:
            break
        else:
            loglike_old=loglike
             
    return pi, mu, sigma, loglike

    
    
##############################################################################################
# Assignment 6

def plot_gmm_solution(X, mu, sigma):

    #fig, ax = plt.subplots(facecolor='white',dpi=150)
    fig, ax = plt.subplots()

    # plot 2D data points as scatterplot
    plt.scatter(X[:,0],X[:,1])
    # plot mean of each cluster as red cross
    plt.scatter(mu[:,0],mu[:,1],marker='x',color='red',s=100)

    #plot ellipse visualizing the covariance matrix sigma
    for i in range(len(mu)):
        nstd = 2
        eigvals, eigvecs = np.linalg.eigh(sigma[i])
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
        theta = np.arctan2(vy, vx)
        width, height = 2 * nstd * np.sqrt(eigvals)
        ellip = Ellipse(xy=(mu[i,0],mu[i,1]), width=width, height=height, angle=theta, alpha=0.2, color='green')

        ax.add_artist(ellip)
    
    plt.title('Clustering with GMM\n K = %d' %len(mu))
    plt.show()
