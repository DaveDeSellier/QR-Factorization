import numpy as np

def GS_QR(A:np.ndarray):
    
    if (np.size(A)==0): #If the size of the input matrix is 0, return empty vectors 
        return [],[]
    
    dim = np.shape(A) #Get the dimensions of the matrix A
    (p,q) = dim #Set the number rows to p and the number columns to q
    
#     if np.linalg.matrix_rank(A) != q: #If matrix A isn't full rank, return empty vectors (A must have full rank for QR factorization to occur)
#         return [],[]
    
    Q = np.array([[0 for _ in range(q)] for _ in range(p)]) #Set Q to a p x q matrix and fill it with zeros
    R = np.array([[0 for _ in range(q)] for _ in range(q)]) #Set R to a q x q matrix and fill it with zeros
    z = np.array([[0 for _ in range(1)] for _ in range(p)]) #Set z to a p x 1 matrix and fill it with zeros
    rq = np.array([0 for _ in range(p)]) #Set rq to a vector of size p and fill it with zeros    
    sumRQ = rq #this vector will store the summation of Rik*Qi
    
    Q = Q.astype('float64') #Values will be 64-bit floats
    R = R.astype('float64') #Values will be 64-bit floats
    rq = rq.astype('float64') #Values will be 64-bit floats
    sumRQ = sumRQ.astype('float64') #Values will be 64-bit floats
    
    R[0][0] = np.linalg.norm(A[:,0])
    Q[:,0] = A[:,0]/R[0][0]
    np.fill_diagonal(R, 1) #Fill diagonal of R with 1s
    
    for i in range(q): #Ri0 in range q - amount of rows in R 
        for j in range(q):  #R0j in range q - amount of columns in R 
            if R[i][j]!=1: #If not the diagonal, Rij = Q.T * Ak
                R[i,j] = np.dot(Q[:,i].T,A[:,j]) 
            else:
                for k in range(i): #Compute the summation of Rik*Qi in range i (i = 1..q))
                    rq = Q[:,k]*R[k][i]
                    sumRQ = np.add(sumRQ,rq)
                z = np.subtract(A[:,i],sumRQ.T) #~Qk = Ak - the summation of Rij*Qi
                R[i,j] = np.linalg.norm(z) #Rij is the norm of z (i=j)
                Q[:,i] = z/R[i,j] #Qi is z / norm of Rij (i=j)
                 
                rq = np.zeros(np.shape(rq)) #Set values in rq to zero for dot product purposes 
                sumRQ = np.zeros(np.shape(sumRQ)) #Set values in sumRQ to zero for dot product purposes
                z = np.zeros(np.shape(z)) #Set values in z to zero for dot product purposes             
            
    return Q,R #Q and R, the results of the QR factorization of A 

#This function is used in to generate my report, ran int the Report.py module
def leastSquares(A:np.ndarray, b:np.ndarray):

    (Q,R) = GS_QR(A)
      
    invR = np.linalg.inv(R)

    x = np.dot(invR,(np.dot(Q.T,b)))

    return x

#Test the GS_QR function using the code below
# A = np.array([[1,-1,4],[1,4,-2],[1,4,2],[1,-1,0]]) 
# Q,R = GS_QR(A)
# print(np.dot(Q,R))


