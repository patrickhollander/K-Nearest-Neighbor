# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:09:05 2020
@author: ph542724
"""
import numpy as np

def knn(k, data, dataClass, inputs):
    nInputs = np.shape(inputs)[0]
    closest = np.zeros(nInputs)
    
    for n in range(nInputs):
        distances = np.sum((data-inputs[n,:])**2,axis = 1)
        manhattonDistance = abs(np.sum((data-inputs[n,:]),axis = 1))
        indices = np.argsort(manhattonDistance, axis=0)
        if n == 0:
            print(distances)
            print(manhattonDistance)
            print(indices)
        
        classes = np.unique(dataClass[indices[:k]])
        if len(classes) == 1:
            closest[n] = np.unique(classes)
        else:
            counts = np.zeros(max(classes)+1)
            for i in range(k):
                counts[dataClass[indices[i]]] += 1
            closest[n] = np.argmax(counts)
            
    return closest
        
    
#start the main
def main():
    k = 1
    myTrain = np.genfromtxt('Lab14Training.csv', delimiter=',', dtype = 'str')
    myTest = np.genfromtxt('Lab14Testing.csv', delimiter=',', dtype = 'str')
    
    #divide the training data into the data and the class
    myTrainClass = myTrain[:,np.shape(myTrain)[1]-1]
    myTrain = myTrain[:,:np.shape(myTrain)[1]-1]
    
    #divide the testing data into the data and the class
    myTestClass = myTest[:,np.shape(myTest)[1]-1]
    myTest = myTest[:,:np.shape(myTest)[1]-1]
    
    #convert the numeric strings in the data to floats
    myTrain = myTrain.astype(np.float)
    myTest = myTest.astype(np.float)
    
    #convert the numeric strings in the class to int since they will index the count array
    myTrainClass = myTrainClass.astype(np.int)
    myTestClass = myTestClass.astype(np.int)
    
    #run k-nearest neighbor
    results = knn(k,myTrain, myTrainClass, myTest)
    print("\nResults:", results)
    
    diff = myTestClass - results  #any results that are different will be nonzero
    diff = np.where(diff!=0,0,1)  #give incorrect (nonzero) results a vaule of 0, and correct ones 1
    correct = np.sum(diff)
    
    #print percent correct
    print("Correctness:", 100*correct/np.shape(diff)[0])

if __name__ == "__main__":
    main()