# Problem set 2 (35 + 55 + 15 + 28 = 133 pts)


```python
from scipy.sparse import diags # can be used with broadcasting of scalars if desired dimensions are large
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sparse
import scipy
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
import time
```

## Problem 1 (LU decomposition) 35 pts


### 1. LU for band matrices (7 pts)

The complexity to find an LU decomposition of a dense $n\times n$ matrix is $\mathcal{O}(n^3)$.
Significant reduction in complexity can be achieved if the matrix has a certain structure, e.g. it is sparse. 
In the following task we consider an important example of $LU$ for a special type of sparse matrices –– band matrices with the bandwidth $m$ equal to 3 or 5 which called tridiagonal and pentadiagonal respectively.

- (5 pts) Write a function ```band_lu(diag_broadcast, n)``` which computes LU decomposition for tridiagonal or pentadiagonal matrix with given diagonal values. 
For example, input parametres ```(diag_broadcast = [4,-2,1], n = 4)``` mean that we need to find LU decomposition for the triangular matrix of the form:

$$A = \begin{pmatrix}
-2 & 1 & 0 & 0\\
4 & -2 & 1 & 0 \\
0 & 4 & -2 & 1 \\
0 & 0 & 4 & -2 \\
\end{pmatrix}.$$

As an output it is considered to make ```L``` and ```U``` - 2D arrays representing diagonals in factors $L$ (```L[0]``` keeps first lower diagonal, ```L[1]``` keeps second lower, ...), and $U$ (```U[:,0]``` keeps main diagonal, ```U[:,1]``` keeps first upper, ...).
- (2 pts) Compare execution time of the band LU decomposition using standard function from ```scipy```, i.e. which takes the whole matrix and does not know about its special structure, and band decomposition of yours implementation. Comment on the results.


```python


# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;                  add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n):
    # enter your code here
    shift = len(diag_broadcast) // 2
    m = len(diag_broadcast)

    L = np.zeros((shift,n-1))
    U = np.zeros((n,shift+1))

    U[0] = diag_broadcast[shift : ]
    U[: n - shift , -1] = diag_broadcast[-1]

    a = diag_broadcast[shift - 1] / U[0, 0]
    L[0, 0] = a
    for u in range(shift):
        U[1, u] = diag_broadcast[u+shift] - U[0, u+1]*a


    for k in range(1, n - 1):#Eliminate one row
            if shift == 1:
                a = diag_broadcast[0] / U[k, 0]# a = diag_broadcast[l] / U[k, 0]
                L[0, k] = a
                U[k + 1, 0] = diag_broadcast[1] - U[k, 1]*a
                    
            else:
                a = diag_broadcast[0] / U[k - 1, 0]
                L[1, k - 1] = a
                b = diag_broadcast[1] - U[k - 1, 1]*a
                b = b / U[k, 0]
                L[0, k] = b

                U[k + 1, 0] = diag_broadcast[shift] - U[k-1, 2]*a
                U[k + 1, 1] = diag_broadcast[shift + 1]
                if k == (n-2):
                    U[k + 1, 1] = 0
                for u in range(shift):
                    U[k + 1, u] = U[k + 1, u] - U[k, u+1]*b
                ### so hard to automate...
  

    return L, U
```


```python
# Your solution is here
```


```python
n = 5

diag_broadcast = [6, 3, 7, -3, 4]

shift = len(diag_broadcast) // 2

A = np.zeros((n, n))
for k in range(-shift, shift + 1):
    A += np.diag([diag_broadcast[k + shift]] * (n - np.abs(k)), k)

_, L, U = sl.lu(A)
print(L)
print(U)


L, U = band_lu(diag_broadcast = diag_broadcast, n = n)
print(L)
print(U)


```

    [[1.         0.         0.         0.         0.        ]
     [0.42857143 1.         0.         0.         0.        ]
     [0.85714286 0.67241379 1.         0.         0.        ]
     [0.         0.72413793 0.95140665 1.         0.        ]
     [0.         0.         0.89002558 0.84735286 1.        ]]
    [[ 7.         -3.          4.          0.          0.        ]
     [ 0.          8.28571429 -4.71428571  4.          0.        ]
     [ 0.          0.          6.74137931 -5.68965517  4.        ]
     [ 0.          0.          0.          9.51662404 -6.8056266 ]
     [ 0.          0.          0.          0.          9.20666488]]
    [[0.42857143 0.67241379 0.95140665 0.84735286]
     [0.85714286 0.72413793 0.89002558 0.        ]]
    [[ 7.         -3.          4.        ]
     [ 8.28571429 -4.71428571  4.        ]
     [ 6.74137931 -5.68965517  4.        ]
     [ 9.51662404 -6.8056266   0.        ]
     [ 9.20666488  0.          0.        ]]
    

#### Compare execution time of the band LU decomposition using standard function from scipy



```python
diag_broadcast = [6, 3, 7, -3, 4]

band_time = []
sl_time = []

for n in range(1000, 10000, 1000):
    cur_time = time.time()
    band_lu(diag_broadcast = diag_broadcast, n = n)
    band_time.append(time.time() - cur_time)

    shift = len(diag_broadcast) // 2

    A = np.zeros((n, n))
    for k in range(-shift, shift + 1):
        A += np.diag([diag_broadcast[k + shift]] * (n - np.abs(k)), k)

    cur_time = time.time()
    sl.lu(A)
    sl_time.append(time.time() - cur_time)

    print(f'{n} complite',end='\r')

plt.figure(figsize=(9, 9))
plt.plot(np.arange(1000, 10000, 1000), band_time, label='band LU', linewidth=4)
plt.plot(np.arange(1000, 10000, 1000), sl_time, label='scipy LU', linewidth=4)
plt.xlabel('dimension of Matrix', fontsize=18)
plt.ylabel('time', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.legend(fontsize=18)
plt.title('comparasion of methods')
plt.show()


```

    9000 complite


    
![png](output_8_1.png)
    


In general scipy uses the algorithm with complexity $ O (n ^ 3) $. In the case of band matrices, we used an algorithm with complexity $ O (n m) $

### 2. Stability of LU (8 pts)

Let
$A = \begin{pmatrix}
\varepsilon & 1 & 0\\
1 & 1 & 1 \\
0 & 1 & 1
\end{pmatrix}.$ 
* (5 pts) Find analytically LU decomposition with and without pivoting for the matrix $A$.
* (3 pts) Explain, why can the LU decomposition fail to approximate factors $L$ and $U$ for $|\varepsilon|\ll 1$ in computer arithmetic?


```python
# Your solution is here
```

#### Without pivoting

Using Gausioan elimination:

$A = \begin{pmatrix}
\varepsilon & 1 & 0\\
1 & 1 & 1 \\
0 & 1 & 1
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0\\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
\varepsilon & 1 & 0\\
1 & 1 & 1 \\
0 & 1 & 1
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0\\
\frac{1}{\varepsilon} & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
\varepsilon & 1 & 0\\
0 & \frac{\varepsilon - 1}{\varepsilon} & 1 \\
0 & 1 & 1
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0\\
\frac{1}{\varepsilon} & 1 & 0 \\
0 & \frac{\varepsilon}{\varepsilon - 1} & 1
\end{pmatrix}
\begin{pmatrix}
\varepsilon & 1 & 0\\
0 & \frac{\varepsilon - 1}{\varepsilon} & 1 \\
0 & 0 & \frac{1}{1 - \varepsilon}
\end{pmatrix}
= LU$


#### With pivoting

Main goal: exclude $\varepsilon$ as a pivot

Using Gausioan elimination:

$A = \begin{pmatrix}
\varepsilon & 1 & 0\\
1 & 1 & 1 \\
0 & 1 & 1
\end{pmatrix}
=
P_{12}
\begin{pmatrix}
1 & 1 & 1 \\
\varepsilon & 1 & 0\\
0 & 1 & 1
\end{pmatrix}
=
P_{12}
\begin{pmatrix}
1 & 0 & 0\\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 1 & 1 \\
\varepsilon & 1 & 0\\
0 & 1 & 1
\end{pmatrix}
=
P_{12}
\begin{pmatrix}
1 & 0 & 0\\
\varepsilon & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 1 & 1 \\
0 & 1 - \varepsilon & -\varepsilon\\
0 & 1 & 1
\end{pmatrix}
=
P_{12}
\begin{pmatrix}
1 & 0 & 0\\
\varepsilon & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
P_{23}
\begin{pmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 1 - \varepsilon & -\varepsilon
\end{pmatrix}
=
P_{12}
P_{23}
\begin{pmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
\varepsilon & 1 - \varepsilon & 1
\end{pmatrix}
\begin{pmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & -1
\end{pmatrix}
$

$
P_{12}P_{23} =
\begin{pmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{pmatrix}
=
\begin{pmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{pmatrix}
$

$
A = PLU =
\begin{pmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{pmatrix}
\begin{pmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
\varepsilon & 1 - \varepsilon & 1
\end{pmatrix}
\begin{pmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & -1
\end{pmatrix}
$

#### Explain, why can the LU decomposition fail to approximate factors $L$ and $U$ for $|\varepsilon|\ll 1$ in computer arithmetic?

Since we have 2 values with $\frac{1}{\varepsilon}$ and if the ${\varepsilon}$ is close to zero, then these values are undefined(very big).

### 3. Block LU (10 pts) 

Let $A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}$ be a block matrix. The goal is to solve the linear system

$$
     \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} f_1 \\ f_2 \end{bmatrix}.
$$

* (2 pts) Using block elimination find matrix $S$ and right-hand side $\hat{f_2}$ so that $u_2$ can be found from $S u_2 = \hat{f_2}$. Note that the matrix $S$ is called <font color='red'> Schur complement </font> of the block $A_{11}$.
* (4 pts) Using Schur complement properties prove that 

$$\det(X+AB) = \det(X)\det(I+BX^{-1}A), $$


where $X$ - nonsingular square matrix.
* (4 pts) Let matrix $F \in \mathbb{R}^{m \times n}$ and $G \in \mathbb{R}^{n \times m}$. Prove that 

$$\det(I_m - FG) = \det(I_n - GF).$$


```python
# Your solution is here
```

#### Using block elimination find matrix $S$ and right-hand side $\hat{f_2}$ so that $u_2$ can be found from $S u_2 = \hat{f_2}$.

$
\begin{bmatrix}
\begin{array}{cc|c}
A_{11} & A_{12} & f_1\\
A_{21} & A_{22} & f_2
\end{array}
\end{bmatrix}
\to_{II - \frac{A_{21}}{A_{11}}I}
\begin{bmatrix}
\begin{array}{cc|c}
A_{11} & A_{12} & f_1\\
A_{21} - A_{21}A_{11}^{-1}A_{11} & A_{22} - A_{21}A_{11}^{-1}A_{12} & f_2 - A_{21}A_{11}^{-1}f_1
\end{array}
\end{bmatrix}
=
\begin{bmatrix}
\begin{array}{cc|c}
A_{11} & A_{12} & f_1\\
0 & A_{22} - A_{21}A_{11}^{-1}A_{12} & f_2 - A_{21}A_{11}^{-1}f_1
\end{array}
\end{bmatrix}
$

$
(A_{22} - A_{21}A_{11}^{-1}A_{12}) u_2 = f_2 - A_{21}A_{11}^{-1}f_1\\\
S = A_{22} - A_{21}A_{11}^{-1}A_{12}
$

$
\hat{f} = f_2 - A_{21}A_{11}^{-1}f_1
$

#### Using Schur complement properties prove that $\det(X+AB) = \det(X)\det(I+BX^{-1}A)$

$det(X + AB) = det(X + AI^{-1}B) =
det(
\begin{bmatrix}
X & -A\\
B & I
\end{bmatrix}
)=
$

$
=det(
\begin{bmatrix}
X & -A\\
B - BX^{-1}X & I - BX^{-1}(-A)
\end{bmatrix}
)
=
det(
\begin{bmatrix}
X & -A\\
0 & I + BX^{-1}A
\end{bmatrix}
)=
$

$
=det(X(I + BX^{-1}A) - 0) = det(X)det(I + BX^{-1}A)
$

#### Let matrix $F \in \mathbb{R}^{m \times n}$ and $G \in \mathbb{R}^{n \times m}$. Prove that $\det(I_m - FG) = \det(I_n - GF).$

$
det(I_m - FG) = det(I_m - FI_n^{-1}G)= 
det(
\begin{bmatrix}
I_m & F\\
G & I_n
\end{bmatrix}
)=
$

$
=det(
\begin{bmatrix}
I_m - I_mG^{-1}G & F - I_mG^{-1}I_n\\
G & I_n
\end{bmatrix}
)=
det(
\begin{bmatrix}
0 & F - I_mG^{-1}I_n\\
G & I_n
\end{bmatrix}
)=
$

$
=det(0 - G(F - I_mG^{-1}I_n)) = det(-GF + GI_mG^{-1}I_n)) = det(I_n - GF))
$

### 4.  Efficient implementation of LU decomposition (10 pts) 

In the lecture we provide naive implementation of LU factorization with loops and elementwise update of factors. In this subproblem we ask you to provide more efficient implementation of LU factorization and explain how you derive this implementation (main ideas and how you use them in this particular case). 
- (1 pts) Main idea for speed up computation of LU factorization (without using built-in function!) 
- (4 pts) Implement the presented approach to speed up LU 
- (1 pts) Illustrate numerically that your implementation is correct
- (2 pts) Provide the comparison of running time of naive implementation from the lecture, your implementation and NumPy built-in function for range of matrix dimensions. We expect you plot the running time vs matrix dimension for these implementations. So you should get the plot with three lines.
- (2 pts) Discuss the obtained results and explain what other tricks are possible to accelerate computing the LU factorization. 

NumPy or JAX are both ok in this subproblem, but please use the single library for all implementations. 


```python
# Your solution is here
```

#### Main idea for speed up computation of LU factorization

idea: block LU

source: http://www.cs.cornell.edu/~bindel/class/cs6210-f12/notes/lec09.pdf


#### Implement the presented approach to speed up LU


```python
def lecture_lu(matrix):
    a = np.copy(matrix)
    n = a.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for k in range(n): #Eliminate one row
        L[k, k] = 1
        for i in range(k+1, n):
            L[i, k] = a[i, k] / a[k, k]
            for j in range(k+1, n):
                a[i, j] -= L[i, k] * a[k, j]
        for j in range(k, n):
            U[k, j] = a[k, j]
    return L, U


def lu(matrix): # contain L and U in 1 matrix
    A = np.copy(matrix)
    
    n = A.shape[0]
    
    for j in range(n):
        A[j+1:n, j] = A[j+1:n, j] / A[j, j]
        A[j+1:n, j+1:n] = A[j+1:n, j+1:n] -  np.outer(A[j+1:n, j],  A[j, j+1:n])
        
    return A


def block_LU(matrix, number_of_blocks = 2): #contain L and U in 1 natrix
    A = np.copy(matrix)
    n = A.shape[0]
    block_size = n // number_of_blocks
    idx = []
        
    for i in range(number_of_blocks):
        idx.append(block_size * i)
    
    M = len(idx) - 1
    for j in range(M):

        A[idx[j]:idx[j + 1], idx[j]:idx[j + 1]] = lu(A[idx[j]:idx[j + 1], idx[j]:idx[j + 1]]) #lu(A[J][:, J])
        
        L_JJ = np.tril(A[idx[j]:idx[j + 1], idx[j]:idx[j + 1]],-1) + np.eye(idx[j + 1] - idx[j])
        
        U_JJ = np.triu(A[idx[j]:idx[j + 1], idx[j]:idx[j + 1]])
        
        A[idx[j + 1]:n, idx[j]:idx[j + 1]] = A[idx[j + 1]:n, idx[j]:idx[j + 1]] @ np.linalg.inv(U_JJ)
        A[idx[j]:idx[j + 1], idx[j + 1]:n] = np.linalg.inv(L_JJ) @ A[idx[j]:idx[j + 1], idx[j + 1]:n]
        A[idx[j + 1]:n, idx[j + 1]:n] = lu(A[idx[j + 1]:n, idx[j + 1]:n] - A[idx[j + 1]:n, idx[j]:idx[j + 1]] @ A[idx[j]:idx[j + 1], idx[j + 1]:n])
    
    return A
    
```

#### Illustrate numerically that your implementation is correct


```python
#diag_broadcast = [6, 3, 7, -3, 4]

n = 5

# shift = len(diag_broadcast) // 2
# A = np.zeros((n, n))
# for k in range(-shift, shift + 1):
#     A += np.diag([diag_broadcast[k + shift]] * (n - np.abs(k)), k)

A = np.random.random((n, n))
A = A @ A.T

#print(A)
A_lu = lu(A)
L = np.eye(n) + np.tril(A_lu, -1)
U = np.triu(A_lu)
print(np.linalg.norm(A - L @ U))
# print(L)
# print(U)
# print(L @ U)
print()

L, U = lecture_lu(A)
print(np.linalg.norm(A - L @ U))
print()

A_block = block_LU(A)
L = np.eye(n) + np.tril(A_block, -1)
U = np.triu(A_block)
print(np.linalg.norm(A - L @ U))
# print(L)
# print(U)
# print(L @ U)
```

    3.3306690738754696e-16
    
    3.3306690738754696e-16
    
    2.603703785810335e-16
    

#### Provide the comparison of running time of naive implementation from the lecture, your implementation and NumPy(scipy?) built-in function


```python
N = np.linspace(50, 1000, 10, dtype='int')

lecture_time = []
block_time = []
scipy_time = []

for n in N:

    A = np.random.random((n, n))
    A = A @ A.T

    cur_time = time.time()
    lu(A) #lecture_lu(A) #method from lecture take much time
    lecture_time.append(time.time() - cur_time)
    

    cur_time = time.time()
    block_LU(A)
    block_time.append(time.time() - cur_time)

    cur_time = time.time()
    sl.lu(A)
    scipy_time.append(time.time() - cur_time)

    print(f'{n} complited')

plt.figure(figsize=(9, 9))
plt.plot(N, lecture_time, label='lecture LU', linewidth=4)
plt.plot(N, scipy_time, label='scipy LU', linewidth=4)
plt.plot(N, block_time, label='block LU', linewidth=4)
plt.xlabel('dimension of Matrix', fontsize=18)
plt.ylabel('time', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.legend(fontsize=18)
plt.title('comparasion of methods')
plt.show()
```

    50 complited
    155 complited
    261 complited
    366 complited
    472 complited
    577 complited
    683 complited
    788 complited
    894 complited
    1000 complited
    


    
![png](output_28_1.png)
    


#### Discuss the obtained results and explain what other tricks are possible to accelerate computing the LU factorization.

for small $n$ gaussian elimination have close to scipy performance, but time increase fast.

block lu give better result for $n > 400$, but still scipy function better.

other tricks: ... 

## Problem 2 (eigenvalues)  55 pts

### 1. Theoretical tasks (10 pts) 

* (5 pts) Prove that normal matrix is Hermitian iff its eigenvalues are real. Prove that normal matrix is unitary iff its eigenvalues satisfy $|\lambda| = 1$. 

* (5 pts) The following problem illustrates instability of the Jordan form. Find theoretically the eigenvalues of the perturbed Jordan block:

$
    J(\varepsilon) = 
    \begin{bmatrix} 
     \lambda & 1 & & & 0 \\ 
     & \lambda & 1 & & \\ 
     &  & \ddots & \ddots & \\ 
     & & & \lambda & 1 \\ 
     \varepsilon & & & & \lambda  \\ 
    \end{bmatrix}_{n\times n}
$

Comment how eigenvalues of $J(0)$ are perturbed for large $n$.


```python
# Your solution is here
```

#### Prove that normal matrix is Hermitian iff its eigenvalues are real.

A matrix $A$ is normal if and only it is diagonalized by some unitary matrix, i.e., there exists a unitary matrix $U$, such that $A = U^*\Lambda U$

with $\Lambda$ diagonal, containing the eigenvalues of $A$ in the diagonal.

If eigenvalues are real, then $\Lambda^* = \Lambda$ and

$A^* = (U^* \Lambda U)^* = U^*  \Lambda^* U = U^*  \Lambda U = A$

#### Prove that normal matrix is unitary iff its eigenvalues satisfy $|\lambda| = 1$

If $AA^* = A^*A = I$ then $A$ is unitary 

$A = U^*\Lambda U$, $A^* = U^*\Lambda^* U$

$AA^* = U^*\Lambda U U^*\Lambda^* U = U^*\Lambda \Lambda^* U = U^*\Lambda \Lambda U = U^* I U = I$

$A^*A = U^*\Lambda^* U U^*\Lambda U = U^*\Lambda^* \Lambda U = U^*\Lambda \Lambda U = U^* I U = I$

#### Find theoretically the eigenvalues of the perturbed Jordan block:

$
    J(\varepsilon) =
    \begin{bmatrix}
     \lambda & 1 & & & 0 \\
     & \lambda & 1 & & \\
     &  & \ddots & \ddots & \\
     & & & \lambda & 1 \\
     \varepsilon & & & & \lambda  \\
    \end{bmatrix}_{n\times n}
$

We can find eigenvalues from $det(J(\varepsilon) - I\hat{\lambda}) = 0$

$
det(J(\varepsilon) - I\hat{\lambda}) =
\begin{vmatrix}
\lambda - \hat{\lambda} & 1 & & & 0 \\
& \lambda - \hat{\lambda} & 1 & & \\
&  & \ddots & \ddots & \\
& & & \lambda - \hat{\lambda} & 1 \\
\varepsilon & & & & \lambda - \hat{\lambda} \\
\end{vmatrix}
=(\lambda - \hat{\lambda})
\begin{vmatrix}
\lambda - \hat{\lambda} & 1 & & 0\\
& \ddots & \ddots & \\
& & \lambda - \hat{\lambda} & 1 \\
& & & \lambda - \hat{\lambda} \\
\end{vmatrix}
-
\begin{vmatrix}
& 1 & & & 0 \\
&  & \ddots & \ddots & \\
& & & \lambda - \hat{\lambda} & 1 \\
\varepsilon & & & & \lambda - \hat{\lambda} \\
\end{vmatrix}
=
...
=
$

$
=(\lambda - \hat{\lambda})^n - (-1)^n \varepsilon
$

$
(\lambda - \hat{\lambda})^n - (-1)^n \varepsilon = 0 \\\
(\lambda - \hat{\lambda})^n = (-1)^n \varepsilon \\\
(\lambda - \hat{\lambda}) = (-1) \sqrt[n]{\varepsilon} \\\
\hat{\lambda} = \lambda + \sqrt[n]{\varepsilon}
$

#### Comment how eigenvalues of $J(0)$ are perturbed for large $n$.
For $J(0)$ we obtain  $\hat{\lambda} = \lambda + \sqrt[n]{0} = \lambda$

### 2. PageRank (35 pts)


#### Damping factor importance

* (5 pts) Write the function ```pagerank_matrix(G)``` that takes an adjacency matrix $G$ (in both sparse and dense formats) as an input and outputs the corresponding PageRank matrix $A$.


```python


# INPUT:  G - np.ndarray or sparse matrix
# OUTPUT: A - np.ndarray (of size G.shape) or sparse matrix

## G: from i have connection to j
def pagerank_matrix(G):
    # enter your code here

    A = G.copy().astype('float')
    
    s = np.array(np.sum(A, axis=1)).reshape((-1, ))
    
    if sparse.issparse(A):
#         data = A.data
#         columns = A.indices
#         rows = A.indptr \\ rows[1] - rows[0] = number of elements in row 0
        
        for i in range(A.shape[0]):
            if s[i] != 0:
                A.data[A.indptr[i]:A.indptr[i + 1]] /= s[i]
                #SUPER not eficcient
                #how to insert row with 1/shape[0] values?
#             else:
#                 A[i] = np.ones(A.shape[0]) / A.shape[0]
            
            print(i, end='\r')
                
    
    else:
        for i in range(A.shape[0]):
        
            if s[i] > 0:
                A[:, i] /= s[i]
            else:
                A[i] = np.ones(A.shape[0]) / A.shape[0]
    
    return A



```

* (3 pts) Find PageRank matrix $A$ that corresponds to the following graph: <img src="graph.png" width='250'>
What is its largest eigenvalue? What multiplicity does it have?


* (5 pts) Implement the power method for a given matrix $A$, an initial guess $x_0$ and a number of iterations ```num_iter```. It should be organized as a function ```power_method(A, x0, num_iter)``` that outputs approximation to eigenvector $x$, eigenvalue $\lambda$ and history of residuals $\{\|Ax_k - \lambda_k x_k\|_2\}$. Make sure that the method converges to the correct solution on a matrix $\begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$ which is known to have the largest eigenvalue equal to $3$.

#### Find PageRank matrix $A$ that corresponds to the following graph


```python
G = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0]
])

G_crs = sparse.csr_matrix(G)

A = pagerank_matrix(G_crs)

print(A.toarray())
```

    [[0.  1.  0.  0.  0. ]
     [0.  0.  1.  0.  0. ]
     [0.5 0.5 0.  0.  0. ]
     [0.  0.  0.  0.  1. ]
     [0.  0.  0.  1.  0. ]]
    

#### What is its largest eigenvalue? What multiplicity does it have?


```python
eig_val = np.linalg.eigvals(A.toarray())

max_val = eig_val.max()
multiplicity = eig_val[eig_val == max_val].shape[0]

print(f'Max val equal to {abs(max_val)}')
print(f'multiplicity equal to {multiplicity}')
```

    Max val equal to 1.0
    multiplicity equal to 2
    

#### Implement the power method for a given matrix $A$


```python
# INPUT:  A - np.ndarray (2D), x0 - np.ndarray (1D), num_iter - integer (positive)
# OUTPUT: x - np.ndarray (of size x0), l - float, res - np.ndarray (of size num_iter + 1 [include initial guess])
def power_method(A, x0, num_iter, debug=False): # 5 pts
    # enter your code here
    x = x0
    res = []
    l = A.dot(x).dot(x)
    res.append(np.linalg.norm(A.dot(x) - l * x))

    for i in range(1, num_iter + 1):
        x = A.dot(x)
        x /= np.linalg.norm(x)
        l = A.dot(x).dot(x)
        res.append(np.linalg.norm(A.dot(x) - l * x))
        if debug:
            print(f'iteration: {i}', end='\r')
    print()

    res = np.array(res)
    assert len(res) == (num_iter + 1)
    return x, l, res
```


```python
A = np.array([
    [2, -1],
    [-1, 2]
])

x0 = np.random.random(A.shape[0])

x, l, res = power_method(A, x0, 100)

print(l)
```

    
    3.0
    

* (2 pts) Run the power method for the graph presented above and plot residuals $\|Ax_k - \lambda_k x_k\|_2$ as a function of $k$ for ```num_iter=100``` and random initial guess ```x0```.  Explain the absence of convergence. 


* (2 pts) Consider the same graph, but with additional self loop at node 4 (self loop is an edge that connects a vertex with itself). Plot residuals as in the previous task and discuss the convergence. Now, run the power method with ```num_iter=100``` for 10 different initial guesses and print/plot the resulting approximated eigenvectors. Why do they depend on the initial guess?


In order to avoid this problem Larry Page and Sergey Brin [proposed](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) to use the following regularization technique:

$
A_d = dA + \frac{1-d}{N} \begin{pmatrix} 1 & \dots & 1 \\ \vdots & & \vdots \\ 1 & \dots & 1 \end{pmatrix},
$

where $d$ is a small parameter in $[0,1]$ (typically $d=0.85$), which is called **damping factor**, $A$ is of size $N\times N$. Now $A_d$ is the matrix with multiplicity of the largest eigenvalue equal to 1. 
Recall that computing the eigenvector of the PageRank matrix, which corresponds to the largest eigenvalue, has the following interpretation. Consider a person who stays in a random node of a graph (i.e. opens a random web page); at each step s/he follows one of the outcoming edges uniformly at random (i.e. opens one of the links). So the person randomly walks through the graph and the eigenvector we are looking for is exactly his/her stationary distribution â€” for each node it tells you the probability of visiting this particular node. Therefore, if the person has started from a part of the graph which is not connected with the other part, he will never get there.  In the regularized model, the person at each step follows one of the outcoming links with probability $d$ OR teleports to a random node from the whole graph with probability $(1-d)$.

* (2 pts) Now, run the power method with $A_d$ and plot residuals $\|A_d x_k - \lambda_k x_k\|_2$ as a function of $k$ for $d=0.97$, ```num_iter=100``` and a random initial guess ```x0```.

* (5 pts) Find the second largest in the absolute value eigenvalue of the obtained matrix $A_d$. How and why is it connected to the damping factor $d$? What is the convergence rate of the PageRank algorithm when using damping factor?

Usually, graphs that arise in various areas are sparse (social, web, road networks, etc.) and, thus, computation of a matrix-vector product for corresponding PageRank matrix $A$ is much cheaper than $\mathcal{O}(N^2)$. However, if $A_d$ is calculated directly, it becomes dense and, therefore, $\mathcal{O}(N^2)$ cost grows prohibitively large for  big $N$.


* (2 pts) Implement fast matrix-vector product for $A_d$ as a function ```pagerank_matvec(A, d, x)```, which takes a PageRank matrix $A$ (in sparse format, e.g., ```csr_matrix```), damping factor $d$ and a vector $x$ as an input and returns $A_dx$ as an output.

* (1 pts) Generate a random adjacency matrix of size $10000 \times 10000$ with only 100 non-zero elements and compare ```pagerank_matvec``` performance with direct evaluation of $A_dx$.

#### Run the power method for the graph presented above


```python
num_iter = 100
A = pagerank_matrix(G)
x_0 = np.random.random(A.shape[0])
x_0 /= sum(x_0)
x, l, res = power_method(A, x_0, num_iter)

print(f'eigenvector: {x}')
print(f'eigenvalue: {l}')

plt.figure(figsize=(9, 9))
plt.plot(res, linewidth=4)
plt.xlabel('number of itrations', fontsize=18)
plt.ylabel('residuals', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.title('power method for the graph')
plt.show()
```

    
    eigenvector: [0.28086407 0.28086407 0.56172815 0.56762119 0.45221501]
    eigenvalue: 0.9866814136685114
    


    
![png](output_46_1.png)
    



```python
eig_val = np.linalg.eigvals(A)
sorted(eig_val, reverse=True)
```




    [(1.0000000000000004+0j), (1+0j), (-0.5+0.5j), (-0.5-0.5j), (-1+0j)]



we have 2 eigenvalues with absolute value $1$ but with different signs, this fact influence on convergance. 

####  Consider the same graph, but with additional self loop at node 4


```python
G_with_self = np.array([
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1]
])

print()
A_with_self = pagerank_matrix(G_with_self)
x_0 = np.random.random(A_with_self.shape[0])
x, l, res = power_method(A_with_self, x_0, num_iter)

print(f'eigenvector: {x}')
print(f'eigenvalue: {l}')

plt.figure(figsize=(9, 9))
plt.plot(res, linewidth=4)
plt.legend(fontsize=18)
plt.xlabel('number of itrations', fontsize=18)
plt.ylabel('residuals', fontsize=18)
plt.yscale('log')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.title('power method for the updated graph')
plt.show()
```

    No handles with labels found to put in legend.
    

    
    
    eigenvector: [0.39312185 0.7862437  0.39312185 0.12060786 0.24121571]
    eigenvalue: 0.9999999999999997
    


    
![png](output_50_2.png)
    



```python
eig_val = np.linalg.eigvals(A_with_self)
sorted((eig_val), reverse=True)
```




    [(1+0j),
     (1+0j),
     (-0.5+0.49999999999999983j),
     (-0.5+0j),
     (-0.5-0.49999999999999983j)]



Now we have eigenvalues $1$ only with positive sign 


```python
plt.figure(figsize=(9, 9))

plt.xlabel('number of itrations', fontsize=18)
plt.ylabel('residuals', fontsize=18)
plt.yscale('log')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.title('power method for the updated graph')
for i in range(10):
    x, l, res = power_method(A_with_self, np.random.random(A_with_self.shape[0]), num_iter)
    print(f'eigenvector: {x}')
    plt.plot(res, linewidth=4, label=f'set of x {i}')

plt.legend(fontsize=18)
plt.show()
```

    
    eigenvector: [0.28258381 0.56516762 0.28258381 0.32276256 0.64552511]
    
    eigenvector: [0.34372154 0.68744308 0.34372154 0.24130189 0.48260379]
    
    eigenvector: [0.38325965 0.7665193  0.38325965 0.15405989 0.30811978]
    
    eigenvector: [0.33426213 0.66852426 0.33426213 0.25675396 0.51350792]
    
    eigenvector: [0.10657743 0.21315485 0.10657743 0.43170534 0.86341069]
    
    eigenvector: [0.35388713 0.70777426 0.35388713 0.22297237 0.44594475]
    
    eigenvector: [0.21652405 0.4330481  0.21652405 0.37913164 0.75826329]
    
    eigenvector: [0.23692079 0.47384158 0.23692079 0.36420083 0.72840167]
    
    eigenvector: [0.37407166 0.74814332 0.37407166 0.17912139 0.35824278]
    
    eigenvector: [0.40763767 0.81527534 0.40763767 0.0244507  0.04890141]
    


    
![png](output_53_1.png)
    


values of eigenvectors depends on initial distribution, because we have 2 subgraphs which are not connected, in this case Markov chain depends on initial state.

#### Now, run the power method with $A_d$ and plot residuals $\|A_d x_k - \lambda_k x_k\|_2$ as a function of $k$ for $d=0.97$, ```num_iter=100``` and a random initial guess ```x0```


```python
d = 0.97
N = A_with_self.shape[0]

x_0 = np.random.random(N)

A_d = d * A_with_self + ((1 - d) / N) * np.ones((N, N))

x, l, res = power_method(A_d, x_0, num_iter)

print(f'eigenvector: {x}')
print(f'eigenvalue: {l}')

plt.figure(figsize=(9, 9))
plt.plot(res, linewidth=4)
plt.xlabel('number of itrations', fontsize=18)
plt.ylabel('residuals', fontsize=18)
plt.yscale('log')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.title('power method with dumping')
plt.show()

```

    
    eigenvector: [0.33057046 0.65090313 0.32805152 0.27152674 0.5345126 ]
    eigenvalue: 1.0011815087183946
    


    
![png](output_56_1.png)
    


#### Find the second largest in the absolute value eigenvalue of the obtained matrix $A_d$. How and why is it connected to the damping factor $d$? What is the convergence rate of the PageRank algorithm when using damping factor? 


```python
eig_val = np.linalg.eigvals(A_d)
sorted(abs(eig_val), reverse=True)[:2]
```




    [1.0011920943540427, 0.9700000000000009]



$\lambda_2 = d$

Why?...

$q = \frac{\lambda_2}{\lambda_1} = \frac{0.97}{1} = 0.97 = d$


```python

# INPUT:  A - np.ndarray (2D), d - float (from 0.0 to 1.0), x - np.ndarray (1D, size of A.shape[0/1])
# OUTPUT: y - np.ndarray (1D, size of x)
def pagerank_matvec(A, d, x): # 2 pts
    # enter your code here
    #Ad@x = (dA + (1 - d)*np.ones(N, N)/N)@x
    N = A.shape[0]
    y = d * A @ x + (1 - d) * sum(x) / N
    return y
```

#### Generate a random adjacency matrix of size $10000 \times 10000$ with only 100 non-zero elements and compare ```pagerank_matvec``` performance with direct evaluation of $A_dx$.


```python
def generate_matrix(N, nnz):
    M = np.zeros((N, N))
    
    for _ in range(nnz):
        y = np.random.randint(N)
        x = np.random.randint(N)
        
        while M[y][x] == 1:
            y = np.random.randint(N)
            x = np.random.randint(N)
        M[y][x] = 1
    return M
```


```python
A = generate_matrix(10000, 100)
print(sum(sum(A)))
d = 0.97
N = A.shape[0]

x = np.random.random(N)

Ad = d * A + ((1 - d) / N) * np.ones((N, N))
A_scr = sparse.csr_matrix(A)

cur_time = time.time()
res = pagerank_matvec(A_scr, d, x)
print(f'fast: {time.time() - cur_time}')

cur_time = time.time()
res1 = Ad @ x
print(f'direct {time.time() - cur_time}')

print(np.linalg.norm(res - res1))
```

    100.0
    fast: 0.0029954910278320312
    direct 0.10099983215332031
    7.451293627690515e-15
    

#### DBLP: computer science bibliography

Download the dataset from [here](https://goo.gl/oZVxEa), unzip it and put `dblp_authors.npz`  and `dblp_graph.npz` in the same folder with this notebook. Each value (author name) from `dblp_authors.npz` corresponds to the row/column of the matrix from `dblp_graph.npz`. Value at row `i` and column `j` of the matrix from `dblp_graph.npz` corresponds to the number of times author `i` cited papers of the author `j`. Let us now find the most significant scientists according to PageRank model over DBLP data.

* (4 pts) Load the weighted adjacency matrix and the authors list into Python using ```load_dblp(...)``` function. Print its density (fraction of nonzero elements). Find top-10 most cited authors from the weighted adjacency matrix. Now, make all the weights of the adjacency matrix equal to 1 for simplicity (consider only existence of connection between authors, not its weight). Obtain the PageRank matrix $A$ from the adjacency matrix and verify that it is stochastic.
 
 
* (1 pts) In order to provide ```pagerank_matvec``` to your ```power_method``` (without rewriting it) for fast calculation of $A_dx$, you can create a ```LinearOperator```: 
```python
L = scipy.sparse.linalg.LinearOperator(A.shape, matvec=lambda x, A=A, d=d: pagerank_matvec(A, d, x))
```
Calling ```L@x``` or ```L.dot(x)``` will result in calculation of ```pagerank_matvec(A, d, x)``` and, thus, you can plug $L$ instead of the matrix $A$ in the ```power_method``` directly. **Note:** though in the previous subtask graph was very small (so you could disparage fast matvec implementation), here it is very large (but sparse), so that direct evaluation of $A_dx$ will require $\sim 10^{12}$ matrix elements to store - good luck with that (^_<).


* (2 pts) Run the power method starting from the vector of all ones and plot residuals $\|A_dx_k - \lambda_k x_k\|_2$  as a function of $k$ for $d=0.85$.


* (1 pts) Print names of the top-10 authors according to PageRank over DBLP when $d=0.85$. Comment on your findings.


```python
from scipy.sparse import load_npz
import numpy as np
def load_dblp(path_auth, path_graph):
    G = load_npz(path_graph).astype(float)
    with np.load(path_auth) as data: authors = data['authors']
    return G, authors
G, authors = load_dblp('dblp_authors.npz', 'dblp_graph.npz')
```


```python
# Your code is here
```

#### Print its density (fraction of nonzero elements)


```python
print(G.count_nonzero())
print(G.shape[0] ** 2)
print(G.count_nonzero() / (G.shape[0] ** 2))
```

    140388901
    3120688303209
    4.4986518152305776e-05
    

#### Find top-10 most cited authors from the weighted adjacency matrix.


```python
total_cited = np.sum(G, axis=0)
total_cited = np.array(total_cited.tolist()[0])
```


```python
for i in range(10):
    ind = total_cited.argmax()
    print(f'{i} place: {authors[ind]}')
    total_cited[ind] = 0
    
```

    0 place: Scott Shenker
    1 place: Andrew Zisserman
    2 place: Hari Balakrishnan
    3 place: Jiawei Han
    4 place: Anil K. Jain
    5 place: Cordelia Schmid
    6 place: Jitendra Malik
    7 place: Ion Stoica
    8 place: David E. Culler
    9 place: David G. Lowe
    

#### make all the weights of the adjacency matrix equal to 1


```python
new_G = G.copy()
new_G[new_G > 1] = 1
```

#### Obtain the PageRank matrix  $A$


```python
# shape = int(new_G.shape[0] / 100)
# print(shape)
# A = pagerank_matrix(new_G[:shape, :shape], debug=True)
```


```python
A = pagerank_matrix(new_G)
```

    1766546


```python
for i in range(A.shape[0]):
    if np.sum(A[i]) != 1:
        print('not stochastic')
        break

### Oh no! but anyway...
```

    not stochastic
    


```python
A.shape
```




    (1766547, 1766547)




```python
d = 0.85
L = LinearOperator(A.shape, matvec=lambda x, A=A, d=d: pagerank_matvec(A, d, x))
N = A.shape[0]
x = np.ones(N)
x, l, res = power_method(L, x, 100, debug=True)
```

    iteration: 100
    

~10 sec per iteration


```python
print(f'eigenvector: {x}')
print(f'eigenvalue: {l}')

plt.figure(figsize=(9, 9))
plt.plot(res, linewidth=4)
plt.xlabel('number of itrations', fontsize=18)
plt.ylabel('residuals', fontsize=18)
plt.yscale('log')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.title('power method for the big matrix')
plt.show()
```

    eigenvector: [0.00010874 0.00077816 0.00083926 ... 0.00085117 0.00010874 0.00010874]
    eigenvalue: 0.962349497303751
    


    
![png](output_81_1.png)
    



```python
for i in range(10):
    ind = x.argmax()
    print(f'{i} place: {authors[ind]} ({ind})')
    x[ind] = 0
```

    0 place: Abigail Solomon (23996)
    1 place: Andjelka Kovačević (88695)
    2 place: Antal Urmos (118322)
    3 place: Atilim Eser (144046)
    4 place: Ayhan Comert (149719)
    5 place: B. Pődör (155142)
    6 place: Benjamin Dewals (170918)
    7 place: D. Meskauskaite (305443)
    8 place: Dale Chambers (312393)
    9 place: Darko Jevremovic (329431)
    

We lost information about frequency of citing + matrix is not stochastic

### 3. QR algorithm (10 pts)

* Implement QR-algorithm without shifting. Prototype of the function is given below


```python
# INPUT: 
# A_init - square matrix, 
# num_iter - number of iterations for QR algorithm
# OUTPUT: 
# Ak - transformed matrix A_init given by QR algorithm, 
# convergence - numpy array of shape (num_iter, ), 
# where we store the maximal number from the Chebyshev norm 
# of triangular part of the Ak for every iteration
def qr_algorithm(A_init, num_iter): # 3 pts
    # enter your code here
    convergence = []
    Ak = A_init.copy()
    for _ in range(num_iter):
        Q, R = sl.qr(Ak)
        Ak = R @ Q
        convergence.append(np.linalg.norm(Ak, ord = np.inf))
    convergence = np.array(convergence)
    return Ak, convergence
```

#### Symmetric case (3 pts)
- Create symmetric tridiagonal $11 \times 11$ matrix with elements $-1, 2, -1$ on sub-, main- and upper diagonal respectively without using loops.
- Run $400$ iterations of the QR algorithm for this matrix.
- Plot the output matrix with function ```plt.spy(Ak, precision=1e-7)```.
- Plot convergence of QR-algorithm.


```python
# Your solution is here
```


```python
A = np.zeros((11, 11))
A += np.diag([-1] * 10, -1) + np.diag([2] * 11, 0) + np.diag([-1] * 10, 1)
print(A)
```

    [[ 2. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [-1.  2. -1.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0. -1.  2. -1.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0. -1.  2. -1.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0. -1.  2. -1.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0. -1.  2. -1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0. -1.  2. -1.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0. -1.  2. -1.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0. -1.  2. -1.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0. -1.  2. -1.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  2.]]
    


```python
Ak, convergence = qr_algorithm(A, 400)
print(np.round(Ak, 2))
```

    [[ 3.93 -0.    0.    0.    0.   -0.   -0.   -0.   -0.    0.   -0.  ]
     [-0.    3.73 -0.    0.    0.    0.   -0.   -0.   -0.   -0.    0.  ]
     [ 0.   -0.    3.41 -0.   -0.   -0.    0.    0.   -0.   -0.   -0.  ]
     [ 0.    0.   -0.    3.   -0.   -0.   -0.    0.    0.    0.   -0.  ]
     [ 0.    0.    0.   -0.    2.52 -0.   -0.   -0.   -0.    0.    0.  ]
     [ 0.    0.    0.    0.   -0.    2.   -0.    0.   -0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.   -0.    1.48 -0.    0.   -0.    0.  ]
     [ 0.    0.    0.    0.    0.    0.   -0.    1.   -0.    0.   -0.  ]
     [ 0.    0.    0.    0.    0.    0.    0.   -0.    0.59  0.    0.  ]
     [ 0.    0.    0.    0.    0.    0.    0.    0.   -0.    0.27 -0.  ]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    0.   -0.    0.07]]
    


```python
plt.spy(Ak, precision=1e-7)
```




    <matplotlib.image.AxesImage at 0x192820b5a90>




    
![png](output_90_1.png)
    



```python
plt.figure(figsize=(9, 9))
plt.plot(convergence, linewidth=4)
plt.legend(fontsize=18)
plt.xlabel('number of itrations', fontsize=18)
plt.ylabel('convergence', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.title('QR algorithm')
plt.show()
```

    No handles with labels found to put in legend.
    


    
![png](output_91_1.png)
    


#### Nonsymmetric case (4 pts)

- Create nonsymmetric tridiagonal $11 \times 11$ matrix with elements $5, 3, -2$ on sub-, main- and upper diagonal respectively without using loops.
- Run $250$ iterations of the QR algorithm for this matrix.
- Plot the result matrix with function ```plt.spy(Ak, precision=1e-7)```. Is this matrix lower triangular? How does this correspond to the claim about convergence of the QR algorithm?


```python
# Your solution is here
```


```python
A = np.zeros((11, 11))
A += np.diag([5] * 10, -1) + np.diag([3] * 11, 0) + np.diag([-2] * 10, 1)
print(A)
```

    [[ 3. -2.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 5.  3. -2.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  5.  3. -2.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  5.  3. -2.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  5.  3. -2.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  5.  3. -2.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  5.  3. -2.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  5.  3. -2.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  5.  3. -2.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  5.  3. -2.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  5.  3.]]
    


```python
Ak, convergence = qr_algorithm(A, 250)
print(np.round(Ak, 2))
```

    [[ 3.26 -6.   -1.9   0.15 -1.04 -0.21 -0.2  -0.63 -0.19 -0.08  0.04]
     [ 6.23  2.74  0.79 -1.21  0.68 -0.2   0.26  0.34  0.12  0.05 -0.02]
     [ 0.    0.    2.72 -5.8  -1.13  1.2  -0.16 -0.31 -0.22  0.04 -0.04]
     [ 0.    0.    5.18  3.28 -2.75 -0.63 -0.36 -0.84 -0.43 -0.23 -0.1 ]
     [ 0.    0.    0.    0.    3.07 -5.2  -1.29  0.78  0.13 -0.27 -0.03]
     [ 0.    0.    0.    0.    3.85  2.93 -1.56 -3.39 -0.24 -0.12  0.48]
     [ 0.    0.    0.    0.    0.    0.    3.81 -2.59  3.19  1.94  0.83]
     [ 0.    0.    0.    0.    0.    0.    4.11  2.19 -1.96  0.51 -0.36]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    4.15 -3.4  -1.6 ]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    1.18  1.85  3.75]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    3.  ]]
    


```python
plt.spy(Ak, precision=1e-7)
```




    <matplotlib.image.AxesImage at 0x1928226be10>




    
![png](output_96_1.png)
    



```python
plt.figure(figsize=(9, 9))
plt.plot(convergence, linewidth=4)
plt.legend(fontsize=18)
plt.xlabel('number of itrations', fontsize=18)
plt.ylabel('convergence', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.title('QR algorithm')
plt.show()
```

    No handles with labels found to put in legend.
    


    
![png](output_97_1.png)
    


It is almost upper triangular matrix. $A_k$ should be in diagonal form for qr algorithm convergence

## Problem 3. (Pseudo-Schur decomposition) 15 pts
Let's redefine scalar product $ \forall x, y \in \mathbb{C}^n$ in a following way:

$$ [x,y]_J = y^{*}Jx, \text{s.t.}\ J = \text{diag}(j_{11}, j_{22}, \dots, j_{nn})\ \text{and}\ j_{ii} = \pm1\ \forall i \in [1,n].$$

Denote rows of matrix $V \in \mathbb{C}^{n \times n}$ as $v_1, v_2, \dots, v_n$. Then $V$ is called $\textbf{J-orthonormal}$ iff 

$$[v_i, v_k]_J = \pm \delta_{ik}.$$

We will call matrix $T \in \mathbb{C}^{n \times n}$ $\textbf{almost triangular}$ iff $T$ is upper triangular with diagonal blocks of order $1$ or $2$.

Matrix $A \in \mathbb{C}^{n \times n}$ is said to be $\textbf{J-decomposable}$ if exist J-orthonormal matrix $V$ and upper triangular matrix $T$ such that 


$$A = V T V^{-1}.$$

Matrix $A \in \mathbb{C}^{n \times n}$ is said to have $\textbf{pseudoschur J-decomposition}$ if exist J-orthonormal matrix $V$ and almost triangular matrix $T$ such that


$$A = V T V^{-1}.$$

This problem is to get familiar with the fact that two abovementioned decompositions exist not for any square matrix with complex entries.




- (2 pts) $A$ has pseudoschur J-decomposition $A = V T V^{-1}, \ \text{where}\ V = \begin{bmatrix} v_1 & v_2 &  \ldots & v_n \end{bmatrix}, \ T = \begin{bmatrix} T_{ij} \end{bmatrix}$ and $v_1$ is $\textbf{not}$ an eigenvector of $A$. Show that $T_{21} \ne 0$. 



- (5 pts) Given $J = \text{diag}(1, -1)$ and $A = \begin{bmatrix}
3   \ -1\\
-1\   \ 3\\
\end{bmatrix}$, prove that $A$ is not J-decomposable.


- (8 pts) Given that $A \in \mathbb{C}^{n \times n}$ is diagonalizable, show that it has pseudoschur J-decomposition for any $J$ of form $J=\text{diag}(\pm 1, \dots, \pm 1)$.
Note that in order to solve that subproblem you should firstly prove the following fact:

    Let $S \in \mathbb{C}^{m \times n},\ m \ge n,\ J = \text{diag}(\pm 1).$ If $A = S^{*}JS$ and $det(A) \ne 0$, then exists QR decomposition of $S$ with respect to $J$: 
    $$S = P_1 QR P_2^{*} = P_1 Q \begin{bmatrix} R_1 \\ 0 \end{bmatrix} P_2^{*}, \ Q^{*} J^{'}Q = J^{'}, \ J^{'} = P_1^{*}JP_1,$$ where $P_1$ and $P_2$ are permutation matrices, $Q$ is called $J^{'}$- unitary and $R_1$ is almost triangular.



```python
# Your solutuion is here
```

#### Show that $T_{21} \ne 0$.


```python
$
A = VTV^{-1} \\
AV = VT
$

$
AV = 
\begin{bmatrix} 
Av_1 & Av_2 & ... & Av_n 
\end{bmatrix}
$

$
vT = 
\begin{bmatrix} 
v_1 & v_2 & ... & v_n 
\end{bmatrix}
\begin{bmatrix} 
T_{11} & T_{12} & ... & T_{1n} \\
T_{21} & T_{22} & ... & ... \\
0 & T_{32} & ... & ... \\
... & ... & ... & ... \\
0 & 0 & ... & T_{nn} \\
\end{bmatrix}
$
=
$
\begin{bmatrix} 
v_1 T_{11} + v_2 T_{21} + 0  & v_1 T_{12} + v_2 T_{22} + ... & ... 
\end{bmatrix}
$

$
Av_1 = v_1 T_{11} + v_2 T_{21}
$

if $T_{21} = 0 => Av_1 = v_1 T_{11} => v_1$ - eigenvector and $T_{11}$ - eigenvalue, but $v_1$ is not eigenvector of $A => T_{21} \ne 0$ 

```

#### prove that  𝐴  is not J-decomposable.

$J =
\begin{bmatrix}
1 & 0 \\ 
0 & -1 
\end{bmatrix}
$


## Problem 4. (Skeleton decomposition) 28 pts

The application that we are particularly interested in is
the approximation of a given matrix by a low-rank matrix:

$$ A \approx UV^T, A \in \mathbb{R}^{m \times n}, U \in \mathbb{R}^{m \times r}, V \in \mathbb{R}^{n \times r}.$$

It is well known that the best (in any unitary invariant norm) low-rank approximation can be computed via singular value decomposition (SVD). As an alternative, we can consider skeleton decompostion of the form:

$$A \approx A_r = A(: , \mathcal{J})A(\mathcal{I} , \mathcal{J})^{-1}A(\mathcal{I} , :),$$
where $\mathcal{I,J}$ are some index sets of length $r$.

Below we consider the matrix $N \times N$ derived from the following function discretization in the uniform grid in square $[0, 1] \times [0, 1]$:
$ f(x, y) = \exp(-\sqrt{x^2 + y^2})$.
It means that $A = [a_{ij}]$, where $a_{ij} = f(x_i, x_j)$ and $x_i = i / (N-1)$, $x_j = j / (N-1)$, where $i, j = 0,\ldots, N-1$.

- (2 pts) compose this matrix for $N = 5000$ without loops (Hint: function ```np.meshgrid``` can help you)



```python
# Your solution is here
```


```python
def f(x, y):
    return np.exp(-np.sqrt(x**2 + y**2))

N = 5000

xx, yy = np.meshgrid(np.arange(N)/(N-1), np.arange(N)/(N-1))

A = f(xx, yy)

print(A[:2, :2])
```

    [[1.         0.99979998]
     [0.99979998 0.99971714]]
    

- (3 pts) Compute Skeleton approximation with random selection of rows and columns indices for $r = 5$ (check that submatrix in the intersection of rows and columns is nonsingular). Average the relative error $\frac{\|A - A_r \|_F}{\|A\|_F}$ over $M$ samples of column/row indices. Check that $M$ is sufficiently large to provide stable mean.

 Note: extracting submatrices should be done according to ```numpy```  e.g. ```A[index_set, :]``` to extract selected rows, ```A[:, index_set]``` to extract selected columns etc..



```python
# Your solution is here
```


```python
r = 5
M = 1000
er = 0
count = 0
for _ in range(M): 
    J = np.random.choice(N, r)
    L = np.random.choice(N, r)

    A1 = A[:, J]
    A2 = A[L, :][:, J]
    if np.linalg.det(A2) > 0:
        A2 = np.linalg.inv(A2)
    else:
        continue
    A3 = A[L, :]
    
    As = A1 @ A2 @ A3
    count += 1
    er += (np.linalg.norm(A - As) / np.linalg.norm(A))

print(er / count)  


```

    0.16967791863429288
    

As you should know from the lecture, if $A$ is of rank $r$ and $\hat{A} = A(\mathcal{I} , \mathcal{J})$ is nonsingular, then the exact equality holds. In the approximate case, however, the quality of the approximation depends on the volume of the submatrix $\hat{A}$: 

**Theorem**

*If $\hat{A} = A_{max}$ has maximal in modulus determinant among all $r \times r$ submatrices of $A$, the following error etimate holds:*

$$ \|A - A_r\|_1 \leq (r+1)\sigma_{r+1}.$$


And the question is how to choose a good submatrix of nearly maximal volume in practice.

**Definition**: *We call $r \times r$ submatrix $A_{dom}$ of rectangular $n \times r$ matrix $A$ of
full rank dominant, if all the entries of $AA_{dom}^{-1}$ are not greater than $1$ in
modulus.*

The crucial theoretical result behind the scene is that the volume of any dominant submatrix $A_{dom}$ can not be very much smaller than the maximum volume submatrix $A_{max}$ (without proof).

We provide the following algorithm for constructing dominant submatrix of a tall matrix.

**Algorithm 1**: 
    
Given matrix $A$ of size $n \times r$ finds dominant submatrix of size $r \times r$

__step 0.__ Start with arbitrary nonsingular $r \times r$ submatrix $A_{dom}$. Reorder rows in $A$ so that $A_{dom}$ occupies first $r$ rows in $A$.

__step 1.__ Compute $B = AA_{dom}^{-1}$ and find its maximum in module  entry $b_{ij}$.

__step 2.__ **If $|b_{ij}| > 1 + \delta$, then**:

Swap rows $i$ and $j$ in $B$ (accrodignly in A). By swapping the rows we have increased the volume of the upper submatrix in $B$, as well as in $A$ (why?). Let $A_{dom}$ be the new upper submatrix of $A$ and go to __step 1__.

**elseif $|b_{ij}| < 1 + \delta$**:

return $A_{dom}$.

Note: $\delta = 10^{-2}$ seems to be a good practical choice.

- (10 pts) Implement algorithm 1 according to the following signature, where the function returns ```row_indices``` array which can be used as ```A[row_indices, :]``` to extract selected submatrix.

Note that matrix inverse $A_{dom}^{-1}$ in step 3 has to be updated efficiently using [Shermann-Morison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula) (inverse of rank-1 update).


```python
def dominant_submatrix_search(A):
    # Your code is here
    A_ = A.copy()
    n, r = A_.shape
    
    eps = 0.01
    
    #step 0
    row_indices = np.random.choice(n, r, replace=False)
    Adom = A_[row_indices, :]
    
    rest_indeces = []
    for i in range(n):
        if i not in row_indices:
            rest_indeces.append(i)
    rest_indeces = np.array(rest_indeces)
    
    all_indeces = np.concatenate((row_indices, rest_indeces), axis=None)
    
    A_[r: ,:] = A_[rest_indeces, :]        
    A_[:r, :] = Adom
    
    while(True):
        #step 1
        B = A_ @ np.linalg.inv(Adom)

        b = np.max(np.abs(B))
        i_, j_ = np.where(np.abs(B) == b)
        

        #step 2
        if b > (1 + eps):
            B[i_, :], B[j_, :] = B[j_, :], B[i_, :]
            A_[i_, :], A_[j_, :] = A_[j_, :], A_[i_, :]
            all_indeces[i_], all_indeces[j_] = all_indeces[j_], all_indeces[i_]

            Adom = A_[:r, :]
        else:
            break
    print()        
    row_indices = all_indeces[:r]
    return row_indices
```


```python
print(dominant_submatrix_search(np.random.rand(10, 3)))
```

    
    [7 1 6]
    


```python
# check the convergence of your implementation on random data
A = np.random.rand(5000, 10)
row_indices = dominant_submatrix_search(A)
```

    
    


```python
A[row_indices, :].shape
```




    (10, 10)




```python
A1 = A[row_indices, :]
```

- (10 pts) Propose the method to construct Skeleton approximation using Algorithm 1. The signature of the function is provided below.  Compare your algorithm with previously considered random selection on the matrix $A$.

Hint: 

1) start with random selection of columns 

2) if you transpose the matrix for which you find rows, then Algorithm 1 will give you update for columns in the initial matrix


```python
def skeleton(A,  r):
    # Your code is here
    
    return row_indices, column_indices
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
