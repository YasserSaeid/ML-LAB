class krr:
    
    def __init__(self, kernel, kernelparameter = 1., regularization = 1e-5):
        self.kernel = kernel
        self.kernelparameter = krr.get_correct_float(kernelparameter)
        self.regularization = krr.get_correct_float(regularization)
    
    def get_correct_float(x):
        try:
            if len(x) == 1:
                return x[0]
        except:
            return x      
    
    def kernel_matrix(self, X):        
        if self.kernel in ('linear',['linear']):
            K = np.dot(X, X.T)
        elif self.kernel in ('polynomial',['polynomial']):
            K = np.power((np.dot(X, X.T) + 1), self.kernelparameter)
        elif self.kernel in ('gaussian',['gaussian']):
            K = np.exp( -np.power(cdist(X, X, 'euclidean'), 2)/(2.*self.kernelparameter**2))
        else: 
            print('kernel not implemented. please choose a kernel out of: linear, polynomial, gaussian')
        return K
        
    def fit(self, X, y):        
        self.X_train = np.asarray(X)
        #HERE CHANGED
        self.d = np.min(X.shape)
        
        K = krr.kernel_matrix(self, X)
        
        #efficient LOO-cv
        if self.regularization == 0:
            eigval, U = np.linalg.eigh(K)
            U = np.asmatrix(U)
            L = np.asmatrix(np.eye(len(eigval)) * eigval)
            candidates = np.power(10., np.arange(-5, 6)) * np.mean(eigval)
            error = []
            for c in candidates:
                #do efficient leave-one-out cross validation
                #for computing the optimal regularization parameter                
                S = U * L * np.linalg.inv(L + c * np.eye(len(L))) * U.T
                error_vector = np.power(((y - np.dot(S,y))/(np.ones_like(y)-np.diagonal(S))),2)
                error.append(np.sum(error_vector)/len(y))                           
            #now chose the parameter with the lowest error
            self.regularization = candidates[error.index(min(error))]        
        self.alpha = np.asarray(np.dot(np.linalg.inv(K + self.regularization * np.eye(len(K))), y)).reshape(len(X), 1)
        
    def predict (self, Z):
        if self.kernel in ('linear',['linear']):            
            prediction = (np.asmatrix(np.dot(self.alpha.T, self.X_train))*np.asmatrix(Z).T).T        
        elif self.kernel in ('polynomial',['polynomial']):
            prediction = np.dot(self.alpha.T, np.power((np.dot(self.X_train, Z.T) + 1), self.kernelparameter)).T
        elif self.kernel in ('gaussian',['gaussian']):
            prediction = np.dot(self.alpha.T, np.exp( -np.power(cdist(self.X_train, Z, 'euclidean'), 2)/(2.*self.kernelparameter**2))).T
        return prediction
