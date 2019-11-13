def cv(X, y, method, parameters, nfolds = 10, nrepetitions = 5, loss_function = zerooneloss):
    
    d = np.min(np.asarray(X).shape)
    #n, d = np.asarray(X).shape
    
    #Preparation: create a df containing all parameter combinations 
    combs = list(it.product(*parameters.values()))    
    param_names = list(parameters.keys())    
    combs_df = pd.DataFrame(combs)
    combs_df.columns = np.array(list(parameters.keys()))             
    #nfold groups of indices 
    indices = np.arange(0,len(X),1)
    avg_error = []    
    step = 1
    
    #progress bar     
    total_steps = nrepetitions * len(combs) * nfolds
    load = 0
    steps_done = 0
    loading = '.' * total_steps
    
    while step <= nrepetitions:           
        np.random.shuffle(indices)        
        nfold_indices = np.array_split(indices.copy(),nfolds)
        
        for i in range(len(combs)):                 
            #chose combination of parameters 
            single_comb = combs_df.loc[i:i].to_dict('list')
            #instantiate method with given parameters
            method_comb = method(**single_comb)                        
            error = []
            
            #perform cv 
            for j in range(nfolds):                
                #choosing test and training set
                               
                testX = np.asarray(np.take(X,nfold_indices[j],axis = 0)).reshape(len(nfold_indices[j]),d)               
                testY = np.take(y,nfold_indices[j], axis = 0).reshape(len(nfold_indices[j]),1) 
                
                training_idx = [a for a in indices if a not in nfold_indices[j]]
                
                trainingX = np.asarray(np.take(X,training_idx,axis = 0)).reshape(len(training_idx),d)                               
                trainingY = np.take(y,training_idx, axis = 0).reshape(len(training_idx),1)                
                
                
                #fit
                method_comb.fit(trainingX, trainingY)
                #predict
                y_pred = method_comb.predict(testX)            
                #calculate loss                
                error.append(loss_function(testY, y_pred)) 
                
                #progressbar                
                steps_done +=1
                print('\r%s Cross Validation at %3d percent!' % (loading, steps_done*100/total_steps), end='')
                loading = loading[:steps_done] + '#' + loading[steps_done+1:]
                
                
            #calculate mean loss of test/training set loss for a specific parameter combination
            avg_error.append(np.mean(error))    
            
        step += 1        
        
    error_matrix = np.asarray(avg_error).reshape(nrepetitions, len(combs))  
    #if only one combination as input, return the average loss
    if len(combs) == 1:
        return np.mean(error_matrix)    
    avg_error_total  = np.mean(error_matrix, axis=0)
       
    #chose best combination resulting in the least error    
    best_comb = combs_df.loc[avg_error_total.argmin():avg_error_total.argmin()].to_dict('list')
    
    #return the class with the best parameter combination evalued by the loss function    
    #return method(**best_comb)
    
    result = method(**best_comb)
    result.fit(X,y)
    result.cvloss = np.min(avg_error_total)
    
    return result


#############################################assignment 2###################################################

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
