def auc(y_true, y_val, plot=False):  
    #check input
    if len(y_true) != len(y_val):
        raise ValueError('Label vector (y_true) and corresponding value vector (y_val) must have the same length.\n')
    #empty arrays, true positive and false positive numbers
    tp = []
    fp = []
    #count 1's and -1's in y_true
    cond_positive = list(y_true).count(1)
    cond_negative = list(y_true).count(-1)
    #all possibly relevant bias parameters stored in a list
    bias_set = sorted(list(set(y_val)), key=float, reverse=True)
    bias_set.append(min(bias_set)*0.9)
    
    #initialize y_pred array full of negative predictions (-1)
    y_pred = np.ones(len(y_true))*(-1)
    
    #the computation time is mainly influenced by this for loop
    #for a contamination rate of 1% it already takes ~8s to terminate
    for bias in bias_set:
        #"lower values tend to correspond to label −1" [sheet1.pdf]
        #indices of values which exceed the bias
        posIdx = np.where(y_val > bias)
        #set predicted values to 1
        y_pred[posIdx] = 1
        #the following function simply calculates results which enable a distinction 
        #between the cases of true positive and  false positive
        results = np.asarray(y_true) + 2*np.asarray(y_pred)
        #append the amount of tp's and fp's
        tp.append(float(list(results).count(3)))
        fp.append(float(list(results).count(1)))
        
    #calculate false positive/negative rate
    tpr = np.asarray(tp)/cond_positive
    fpr = np.asarray(fp)/cond_negative
    #optional scatterplot
    if plot == True:
        plt.scatter(fpr,tpr)
        plt.show()
    #calculate AUC
    AUC = np.trapz(tpr,fpr)
    
    return AUC
