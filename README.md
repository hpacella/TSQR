# Parallel TSQR

The tall, skinny QR (TSQR) decomposition of a matrix A developed by Demmel et al. [1] is a communication-optimal QR decomposition for matrices with many more rows than columns. 

The matrix A is divided into _i_ block rows that satisfy m<sub>_i_</sub>&ge;n, with each of these blocks located on a different processor. The algorithm starts by finding the QR decomposition of these individual blocks. Each processor then shares their resulting (local) R matrix with its neighbors; these matrices are combined and their QR decomposition is found. This process is repeated until the final level of the processor tree. The Q matrices are locally stored on the individal processors.

The final R factor is taken to be the R factor of A = QR, and Q is stored implicitly as a product of block orthogonal matrices. An example of this for 4 processors with a binary tree is shown below [1].

<p align="center">
 <img src="images/example.png" width = "470" height = "213">
</p>

The complete algorithm is shown below [1].

<p align="center">
 <img src="images/algorithm.png" width="545" height = "330">
</p>

### Interfacing with BLAS
A special type of Regent index space was used so that BLAS subroutines could be called for the computations of Q and R. 
  * R is calculated using the subroutine [dgeqrf](http://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html).
  * Q is calculated using the subroutine [dorgqr](http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga14b45f7374dc8654073aa06879c1c459.html).
  
### Implementations
This decomposition is implemented in several different ways, depending on the desired final result (R on one processor, R on all processors, or R on all processors and Q implicity stored). They are contained in the following folders:

  * **TSQR_one_proc_R** : This implementation computes only R. There is no redundant work done in this algorithm; the final R matrix is only contained on one processor. Any number of initial processors can be used, and each processor is considered to have at most one neighbor. When there are an even number of processors at a level, the processors form pairs, with one processor sending its R matrix to the other processor in the pair. When there is an odd number of processors at a level, the first (N-1) processors are again paired off, but the remaining processor does not perform any computations, instead "carrying" its current R matrix through to the next level. An example of the tree for an odd number of initial processors is shown below. 
  
 <p align="center">
 <img src="images/TSQR_one_proc.png" width="436" height = "192">
</p>
  
  * **TSQR_all_proc_R** : This implementation also computes only R, but the final R matrix is computed on every processor. This is accomplished via redundant work. The tree used is a butterfly all-reduction, shown below [2]. Because of this communication pattern, the initial number of processors used to subdivide the matrix must be a power of 2. 
  
  <p align="center">
 <img src="images/butterfly_all_reduction.png" width="423" height = "319">
</p>
  
  * **TSQR_complete_QR** : This implementation is very similar to that in the TSQR_all_proc_R folder, but it also implicitly calculates Q. Q is made up of a series of orthogonal matrices locally computed and stored on each processor.
  
### File Inputs

There are several inputs that must be provided to run the TSQR code:
  * **m** : The number of rows in the matrix A.
  * **n** : The number of columns in the matrix A.
  * **P** : The number of "processors" that that the matrix A is divided into.
  * **m_vec (optional)**  : The row dimensions of the submatrices that the matrix A is divided into. If no dimensions are provided, a uniform row dimension of (A/P) will be used.
  * **matrix input file** : An HDF5 file that contains the entries of the matrix A. If no input file is provided, then the matrix is initialized using a full-rank block matrix (for testing purposes).

These inputs are contained in the file config.rg and are stored in a terra struct. 

### References

[1] Demmel, James, et al. “Communication-Optimal Parallel and Sequential QR and LU Factorizations.” SIAM Journal on Scientific Computing, vol. 34, no. 1, 2 Feb. 2012, pp. A206–A239., doi:10.1137/080731992.

[2] Demmel, James, et al. Communication-Optimal Parallel and Sequential QR and LU Factorizations. Communication-Optimal Parallel and Sequential QR and LU Factorizations.
