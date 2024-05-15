import numpy as np
from numpy.linalg import norm

def hermitian_eigensystem(H, tolerance):
    
    """ Solves for the eigenvalues and eigenvectors of a hermitian matrix
    
    Args:
        H: Hermitian matrix for which we want to compute eigenvalues and eigenvectors
        
        tolerance: A number that sets the tolerance for the accuracy of the computation.  This number
        is multiplied by the norm of the matrix H to obtain a number delta.  The algorithm successively
        applies (via similarity transformation) Jacobi rotations to the matrix H until the sum of the
        squares of the off-diagonal elements are less than delta.
    
    
    
    Returns:
        d: Numpy array containing eigenvalues of H in non-decreasing order
        
        U: A 2d numpy array whose columns are the eigenvectors corresponding to the computed
        eigenvalues.
        
        
    Checks you might need to do:
        
        H * U[:,k] = d[k] *　U[:,k]      k=0,1,2,...,(n-1)
        
        d[0] <= d[1] <= ... <= d[n-1]     (where n is the dimension of H)
       
        np.transpose(U) * U = U * np.transpose(U) = np.eye(n)
        
    """
    
    # call complex_eigen(H,tolerance)
    # rearrange d and U, so that they are in the non-decreasing order of eigenvalues
    d, U = complex_eigen(H, tolerance)
    
    return d, U


def find_largest_non_diagonal_index(matrix):
    min = np.finfo(float).min
    matrix = np.abs(matrix)
    index = (0, 0)
    for i in range(matrix[0].size):
        for j in range(matrix[0].size):
            if i == j:
                continue
            if matrix[i, j] > min:
                min = matrix[i, j]
                index = (i, j)
    return index

#difficulty: ★★★
def jacobi_rotation(A, j, k):
    a = A[j, j]
    b = A[k, k]
    c = A[j, k]

    theta = 0.5*(np.arctan2(2*c, a-b) )
    
    J = np.diag(np.full(A.shape[0], 1)).astype(float)
    J = np.identity(len(A))
    J[j, j] = np.cos(theta)
    J[j, k] = -np.sin(theta)
    J[k, j] = np.sin(theta)
    J[k, k] = np.cos(theta)
    A = J.transpose() @ A @ J
    
    return A, J

#difficulty: ★
def off(A, tol):
    # calculate the Frobenius norm of A and off(A)
    # to see if we want to stop the call
    A_off = A.copy()
    np.fill_diagonal(A_off, 0)
    off_norm = np.linalg.norm(A_off, 'fro')
    frob_norm = np.linalg.norm(A, 'fro')
    return (off_norm/frob_norm < tol)

#difficulty: ★★★
def real_eigen(A,tolerance):
    J_arr = []

    # iterative process
    while (True):
        j, k = find_largest_non_diagonal_index(A)
        A, J = jacobi_rotation(A, j, k)
        J_arr.append(J)
        if off(A, tolerance):
            break

    # compute result
    R = np.diag(np.full(A.shape[0], 1))

    d = np.diagonal(A)
    for index in range(len(J_arr)):
        R = R @ J_arr[index]
    
    return order_eigensystems(d, R)

def first_occurrences(array):
    first_occ_dict = {}
    for index, element in enumerate(array):
        if element not in first_occ_dict:
            first_occ_dict[element] = index
    
    return first_occ_dict


def complex_to_real_matrix(H):
    real = np.real(H)
    imag = np.imag(H)
    out = np.block([[real, -imag], [imag, real]])

    return out

def real_to_complex_matrix(H):
    # Determine the size of the sub-matrices
    n = H.shape[0] // 2
    
    # Extract A and B from C
    A = H[:n, :n]  # Top left block
    B = H[n:, :n]  # Bottom left block
    
    return A+1j*B

def order_eigensystems(d, U, round = 7):
    d = np.round(d, round)
    U = np.round(U, round)

    indices = np.argsort(d.real)
    d = d[indices]
    U = U[:, indices]
    return d, U

#difficulty: ★★
def complex_eigen(H,tolerance):
    size = H[0].size
    if H.dtype != 'complex128':
        return real_eigen(H, tolerance)
    
    
    H = complex_to_real_matrix(H)
    d, U = real_eigen(H, tolerance)
    n = U[0].size
    top = U[:n//2, :]
    bot = U[n//2:, :]
    U = top + 1j*bot

    unique_eigenvalues = {}
    for i, e_val in enumerate(d):
        if e_val not in unique_eigenvalues:
            unique_eigenvalues[e_val] = i

    d = unique_eigenvalues.keys()
    indices = list(unique_eigenvalues.values())
    temp = U
    U = np.zeros((size, size), dtype='complex')
    for i in range(len(indices)):
        U[:,i] = temp[:,indices[i]]

    d_list = list(d)
    d = np.array(d_list)

    return d, U