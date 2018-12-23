import "regent"

local c = regentlib.c
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
local cstr = terralib.includec("string.h")

--fortran-order 2D index space
struct __f2d { y : int, x : int}
f2d = regentlib.index_type(__f2d, "f2d")
rectf2d = regentlib.rect_type(f2d)

local blas = require("BLAS_functions_TSQR")
local h = require("helper_functions_TSQR")
require("config")

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~                MAIN                    ~~~~
----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

task main()

  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  --~~~~  input arguments/domain construction  ~~~~
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  var config : Config
  config:initialize()
  config:checks()

  var P : int = config.P --number of "processors" the matrix will be divided across
  var m : int = config.m --matrix rows
  var n : int = config.n --matrix columns

  --build A matrix
  var matrix_dim : f2d = {x = n, y = m}
  var matrix = region(ispace(f2d, matrix_dim), double)
 
  --used for partitioning of the matrix
  var blocks : f2d = {x = 1, y = P}
  var mat_colors = ispace(f2d, blocks)

  --submatrix dimensions
  var m_vec = region(ispace(int1d, P), int) 
  var q_dims = region(ispace(int1d, P), int)
  fill(q_dims, n) --column dimension remains constant for this implementation
  var r_vec = region(ispace(int1d, P), int)
  fill(r_vec, 2*n)

  if config.nonuniform_mat_vec then
    --h.initialize_mat_vec(P, m_vec, mat_vec_file)
  else
    fill(m_vec, m/P) --assume equal division if no submatrix dimensions given
  end

  --color the matrix subregions
  var matrix_part = h.color_matrix(m_vec, q_dims, matrix, mat_colors)

  --initialize the matrix
  if config.read_input_matrix_file then
    if not h.file_exists(config.input_matrix_file) then
      c.printf("Input matrix file does not exist!\n")
      c.abort()
    end
    --h.initialize_from_file(matrix, input_matrix_file) 
  else
    --[[
    for i in mat_colors do
      h.initialize(i.y, m_vec[i.y], matrix_part[i])
    end
    ]]--
    --random, full rank matrixfor verification
    matrix[f2d {x = 0, y = 0}] = 3
    matrix[f2d {x = 0, y = 1}] = 8
    matrix[f2d {x = 0, y = 2}] = 5
    matrix[f2d {x = 0, y = 3}] = 5
    matrix[f2d {x = 0, y = 4}] = 9
    matrix[f2d {x = 0, y = 5}] = 2
    matrix[f2d {x = 0, y = 6}] = 7
    matrix[f2d {x = 0, y = 7}] = 7
    matrix[f2d {x = 0, y = 8}] = 3
    matrix[f2d {x = 0, y = 9}] = 5
    matrix[f2d {x = 1, y = 0}] = 0
    matrix[f2d {x = 1, y = 1}] = 0
    matrix[f2d {x = 1, y = 2}] = 5
    matrix[f2d {x = 1, y = 3}] = 7
    matrix[f2d {x = 1, y = 4}] = 9
    matrix[f2d {x = 1, y = 5}] = 1
    matrix[f2d {x = 1, y = 6}] = 5
    matrix[f2d {x = 1, y = 7}] = 4
    matrix[f2d {x = 1, y = 8}] = 0
    matrix[f2d {x = 1, y = 9}] = 3
    matrix[f2d {x = 2, y = 0}] = 1
    matrix[f2d {x = 2, y = 1}] = 7
    matrix[f2d {x = 2, y = 2}] = 3
    matrix[f2d {x = 2, y = 3}] = 5
    matrix[f2d {x = 2, y = 4}] = 1
    matrix[f2d {x = 2, y = 5}] = 6
    matrix[f2d {x = 2, y = 6}] = 2
    matrix[f2d {x = 2, y = 7}] = 6
    matrix[f2d {x = 2, y = 8}] = 6
    matrix[f2d {x = 2, y = 9}] = 7
    matrix[f2d {x = 3, y = 0}] = 4
    matrix[f2d {x = 3, y = 1}] = 0
    matrix[f2d {x = 3, y = 2}] = 2
    matrix[f2d {x = 3, y = 3}] = 9
    matrix[f2d {x = 3, y = 4}] = 1
    matrix[f2d {x = 3, y = 5}] = 8
    matrix[f2d {x = 3, y = 6}] = 5
    matrix[f2d {x = 3, y = 7}] = 9
    matrix[f2d {x = 3, y = 8}] = 0
    matrix[f2d {x = 3, y = 9}] = 4

  end
  
  --build R matrix
  var R_matrix_dim : f2d = {x = n, y = P*n}
  var R_matrix = region(ispace(f2d, R_matrix_dim), double)
  fill(R_matrix, 0.0)

  --partition the R matrix
  var r_blocks = {x = 1, y = P}
  var R_matrix_part = partition(equal, R_matrix, ispace(f2d, r_blocks))

  --temporary R matrix regions
  var R_matrix_temp_dim : f2d = {x = n, y = 2*P*n}
  var R_matrix_temp = region(ispace(f2d, R_matrix_temp_dim), double)
  fill(R_matrix_temp, 0.0)
  --partition the temporary R matrix
  var r_temp_blocks = {x = 1, y = P}
  var R_matrix_temp_part = partition(equal, R_matrix_temp, ispace(f2d, r_temp_blocks))

  --tau region (used to find Q matrices)
  var tau_dim : f2d = {x = P, y = m}
  var tau = region(ispace(f2d, tau_dim), double)
  var tau_blocks : f2d = {x = P, y = 1}
  var tau_part = partition(equal, tau, ispace(f2d, tau_blocks)) 
            
  --work region (used as scratch by BLAS subroutines)
  var work_dim : f2d = {x = P, y = m*n}
  var work = region(ispace(f2d, work_dim), double)
  var work_blocks : f2d = {x = P, y = 1}
  var work_part = partition(equal, work, ispace(f2d, work_blocks)) 
  
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  --~~~~           parallel TSQR               ~~~~
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  --number of levels in the tree
  var L : int = cmath.ceil(cmath.log2(P)) 
  var P_dest = region(ispace(int1d, P), int)
  
  --Region to store Q matrices
  var Q_dim : f2d
  var Q_mat_blocks : f2d
  if L == 0 then --creates dummy index, Q region actually won't be used (when P = 1)
    Q_dim = {x = n, y = P*2*n}
    Q_mat_blocks = {x = 1, y = P}
  else 
    Q_dim = {x = L*n, y = P*2*n}
    Q_mat_blocks = {x = L, y = P}
  end

  var Q_mat = region(ispace(f2d, Q_dim), double)
  var Q_mat_part = partition(equal, Q_mat, ispace(f2d, Q_mat_blocks))
  
  var procs = ispace(int1d, P)
  
  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()
  
  for k = 0, (L+1) do --successive levels of the tree
 
    if k ~= 0 then
 
      --Steps (not necessarily sequential):
      --1. Each processor shares its current R matrix with its neighbor (butterfly all-reduction pattern).
      --2. Combine adjacent (n x n) R matrices into (2n x n) matrices.
      --3. Find the QR factorization of these new matrices.
      --4. Extract R values.
      --5. Store local implicit Q matrix.

      --1.find destination processors for butterfly all reduction
      var neighbor : int = cmath.pow(2,(k-1))
      var step : int = cmath.pow(2,k)
     
      for i = 0, P, step do
        for j = i, (i + neighbor) do          
          P_dest[j] = j + neighbor
          P_dest[j + neighbor] = j
        end
      end
 
      --1./2. Share R matrices with neighbor and make (2n x n) matrices
      for y in procs do

        --figure out position in shared matrix
        var temp_loc : int

        if [int](y) > P_dest[y] then temp_loc = 1
        else temp_loc = 0 end

        h.copy_function(R_matrix_part[f2d {x = 0, y = y}], R_matrix_temp_part[{x = 0, y = P_dest[y]}], y, P_dest[y], temp_loc, n)
        h.copy_function(R_matrix_part[f2d {x = 0, y = y}], R_matrix_temp_part[{x = 0, y = y}], y, y, temp_loc, n)

      end

      --3. Find QR factorization for the combined R blocks on each processor
      for y in procs do  
        blas.dgeqrf(y, m, R_matrix_temp_part[f2d {x = 0, y = y}], tau_part[f2d {x = y, y = 0}], work_part[f2d {x = y, y = 0}])
      end

      --4. Recover R on each processor (using original R matrix partition)
      for y in procs do
        h.get_R_matrix(y, r_vec, n, R_matrix_temp_part[f2d {x = 0, y = y}], R_matrix_part[f2d {x = 0, y = y}])
      end

      --5. Calculate Q matrices for this level of the tree
      for y in procs do
        blas.dorgqr(y, m, R_matrix_temp_part[f2d {x = 0, y = y}], tau_part[f2d {x = y, y = 0}], work_part[f2d {x = y, y = 0}])
        h.get_Q_matrix((k-1), n, Q_mat_part[f2d {x = (k-1), y = y}], R_matrix_temp_part[f2d {x = 0, y =y}])
      end

    else  --initial QR factorization

      --find Q vectors, R entries 
      for y in procs do
        blas.dgeqrf(y, m, matrix_part[f2d {x = 0, y = y}], tau_part[f2d {x = y, y = 0}], work_part[f2d {x = y, y = 0}])
      end
      
      --copy values to R matrix
      for y in procs do
        h.get_R_matrix(y, m_vec, n, matrix_part[f2d {x = 0, y = y}], R_matrix_part[f2d {x = 0, y = y}])
      end

      --find Q matrices
      for y in procs do
        blas.dorgqr(y, m, matrix_part[f2d {x = 0, y = y}], tau_part[f2d {x = y, y = 0}], work_part[f2d {x = y, y = 0}])
      end

    end

  end  --end tree levels

  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Total Time : %f s\n", 1e-3*(ts_end - ts_start))
 
  if config.print_solution then
 
    --Print out final R solution
    c.printf("R_final:\n")
    var R_final = R_matrix_part[f2d {x = 0, y = 0}]
    for i in R_final do
      c.printf("entry = (%d, %d) value = %f\n", i.x, i.y, R_final[i])
    end
  
    --Print out final Q entries
    c.printf("Q_final matrices:\n")
    for l = 0, (L + 1) do
      for p = 0, P do

        c.printf("L = %d, P = %d\n", l, p)
        if l == 0 then
          var current_matrix = matrix_part[f2d {x = 0, y = p}]
          for i in current_matrix do
            c.printf("entry = (%d, %d) value = %f\n", i.x, i.y, current_matrix[i])
          end
 
        else
          var current_matrix = Q_mat_part[f2d {x = (l - 1), y = p}] 
          for i in current_matrix do
            c.printf("entry = (%d, %d) value = %f\n", i.x, i.y, current_matrix[i])
          end
        end

      end
    end

  end
  
end --end main task

regentlib.start(main)

