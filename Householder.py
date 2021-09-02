import numpy as np

def Housevector(x:np.ndarray):
    v = x.copy() #Copy the input vector, x, in to the vector v
    v = v.astype('float64') #Values will be 64-bit floats
    
    alpha = v[0] #Store the value of the first index in the vector into the variable, alpha
    
    u = np.dot(np.transpose(v[1:]),v[1:]) #v.T*v part of beta 
    
    #The following is how to process the sign of v[0] to avoid cancellation that can occur if a multiple of identity matrix column (Parlett 1971)
    if u==0: 
        beta=0
    else:
        t = np.sqrt(alpha**2 + u)
        if alpha <= 0:
            v[0] = alpha - t 
        else:
            v[0] = -u/(alpha + t)

        beta = 2 * v[0]**2 / (u + v[0]**2)
        v = v/v[0] #Normalize vector so that v[0] = 1 , this is the direction of v 

    return v, beta

def Householder(A:np.ndarray):
    
    if (np.size(A)==0): #Check if the input matrix A is empty
        return [],[]
    
    (p,q) = A.shape #Get the dimensions of the matrix A
    Q = np.identity(p) #Set Q to a p x p identity matrix
    R = A.copy() #Copy the input matrix A, into R
    P = np.array([]) #Set P to an empty array
    
#     if np.linalg.matrix_rank(A) != q: #Check if the input matrix A is full rank
#         return [],[]
    
    if (p & q == 1): #If the input matrix is 1 x 1, return empty arrays
        return [], []
    
    R = R.astype('float64') #Values will be 64-bit floats
    A = A.astype('float64') #Values will be 64-bit floats
    P = P.astype('float64') #Values will be 64-bit floats
    Q = Q.astype('float64') #Values will be 64-bit floats
    
    for i in range(q): #R will be contain 0...q-1 columns
        (v,beta) = Housevector(R[i:,i]) #Calculate the Householder vector from the first column vector of A[0:q-1]
        P = np.identity(p) #Set P to a p x p identity matrix 
        P[i:,i:] = P[i:,i:] - beta*(np.outer(v,np.transpose(v))) #Calculate the Householder Matrix
        R = np.dot(P,R) #Calculate R 
        Q = np.dot(P,Q) #Store the Householder Matrix in Q
    return Q, R #Q and R, the results of the QR factorization of A 

#This function is used in to generate my report, ran in the Report.py module
def leastSquares(A:np.ndarray, b:np.ndarray):
      
    (Q,R) = Householder(A)
    
    invR = np.linalg.inv(R)
      
    x = np.dot(invR,(np.dot(Q,b)))
    
    return x

#Test the Householder function using the code below
# A = np.array([[1,-1,4],[1,4,-2],[1,4,2],[1,-1,0]])
# Q,R = (Householder(A))
# print(np.dot(Q.T,R))



