import "regent"

local c = regentlib.c
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
local cstr = terralib.includec("string.h")

local blas = require("BLAS_functions_TSQR")
local h = require("helper_functions_TSQR")
require("config")

task get_power(a :int)
  var exp = cmath.pow(2,a)
  return exp
end

task get_levels(P : int)
  var l = cmath.ceil(cmath.log2(P)) 
  return l
end

task print_time(t_micro : double)
  c.printf("Total Time : %f s\n", 1e-6*(t_micro))
end

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~                MAIN                    ~~~~
----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--__demand(__replicable)
task main()

  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  --~~~~  input arguments/domain construction  ~~~~
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  --var config : Config
  --config:initialize()
  --config:checks()

  var P : int = 64 --config.P --number of "processors" the matrix will be divided across
  var m : int = 64000 --config.m --matrix rows
  var n : int = 50 --config.n --matrix columns

  --build A matrix
  var matrix_dim : int2d = {x = m, y = n}
  var matrix = region(ispace(int2d, matrix_dim), double)
 
  --used for partitioning of the matrix
  var blocks = P
  var mat_procs = ispace(int1d, blocks)

  --submatrix dimensions
  var m_vec = region(ispace(int1d, P), int) 
  var q_dims = region(ispace(int1d, P), int)
  fill(q_dims, n) --column dimension remains constant for this implementation
  var r_vec = region(ispace(int1d, P), int)
  fill(r_vec, 2*n)

  --if config.nonuniform_mat_vec then
    --h.initialize_mat_vec(P, m_vec, mat_vec_file)
  --else
    fill(m_vec, m/P) --assume equal division if no submatrix dimensions given
  --end

  --color the matrix subregions
  var matrix_part = h.color_matrix(m_vec, q_dims, matrix, mat_procs)

  --initialize the matrix
  --[[
  if config.read_input_matrix_file then
    if not h.file_exists(config.input_matrix_file) then
      c.printf("Input matrix file does not exist!\n")
      c.abort()
    end
    --h.initialize_from_file(matrix, input_matrix_file) 
  else
    ]]--
    for p in mat_procs do
      h.initialize(p, m_vec[p], matrix_part[p])
    end
  --end
 
  --build R matrix
  var R_matrix_dim : int2d = {x = P*n, y = n}
  var R_matrix = region(ispace(int2d, R_matrix_dim), double)
  fill(R_matrix, 0.0)

  --partition the R matrix
  var r_blocks = P
  var R_matrix_part = h.equal_matrix_part(n, R_matrix, ispace(int1d, r_blocks))  

  --temporary R matrix regions
  var R_matrix_temp_dim : int2d = {x = 2*P*n, y = n}
  var R_matrix_temp = region(ispace(int2d, R_matrix_temp_dim), double)
  fill(R_matrix_temp, 0.0)

  --partition the temporary R matrix
  var r_temp_blocks = P
  var R_matrix_temp_part = h.equal_matrix_part(2*n, R_matrix_temp, ispace(int1d, r_temp_blocks))
  var R_matrix_temp_copy_0_part = h.copy_matrix_part(2*n, 0, R_matrix_temp, ispace(int1d, P))
  var R_matrix_temp_copy_1_part = h.copy_matrix_part(2*n, n, R_matrix_temp, ispace(int1d, P))

  --tau region (used to find Q matrices)
  var tau_dim = m*P
  var tau = region(ispace(int1d, tau_dim), double)
  var tau_blocks = P
  var tau_part = partition(equal, tau, ispace(int1d, tau_blocks)) 
            
  --work region (used as scratch by BLAS subroutines)
  var work_dim = m*n*P
  var work = region(ispace(int1d, work_dim), double)
  var work_blocks = P
  var work_part = partition(equal, work, ispace(int1d, work_blocks)) 
  
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  --~~~~           parallel TSQR               ~~~~
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  --number of levels in the tree
  var L : int = get_levels(P) 
  
  --Region to store Q matrices
  var Q_dim : int2d
  var Q_mat_blocks : int

  var L_Q : int
  if L == 0 then L_Q = 1 
  else L_Q = L end 

  Q_dim = {x = P*L_Q*2*n, y = n}
  var Q_mat = region(ispace(int2d, Q_dim), double)
  --partition by levels
  var Q_mat_part_levels = h.equal_matrix_part(P*2*n, Q_mat, ispace(int1d, L_Q))
  --partition by processors
  var Q_mat_part_procs = h.processor_matrix_part(2*n, L_Q, Q_mat, ispace(int1d, P))
  --cross product partition
  var Q_mat_prod = cross_product(Q_mat_part_levels, Q_mat_part_procs)

  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()

  --Initial QR factorization:
  --find Q vectors, R entries
  __demand(__parallel)
  for p in mat_procs do
    blas.dgeqrf(p, m, matrix_part[p], tau_part[p], work_part[p])
  end
      
  --copy values to R matrix
  __demand(__parallel)
  for p in mat_procs do
    h.get_R_matrix(p, m_vec, n, matrix_part[p], R_matrix_part[p])
  end

  --find Q matrices
  __demand(__parallel)
  for p in mat_procs do
    blas.dorgqr(p, m, matrix_part[p], tau_part[p], work_part[p])
  end
 
  for k = 1, (L+1) do --successive levels of the tree
    
    --Steps (not necessarily sequential):
    --1. Each processor shares its current R matrix with its neighbor (butterfly all-reduction pattern).
    --2. Combine adjacent (n x n) R matrices into (2n x n) matrices.
    --3. Find the QR factorization of these new matrices.
    --4. Extract R values.
    --5. Store local implicit Q matrix.

    --1.find destination processors for butterfly all reduction
    var neighbor : int = get_power(k-1)
    var step : int = get_power(k)

    --1./2. Share R matrices with neighbor and make (2n x n) matrices
    --__demand(__parallel)
    for p in mat_procs do
      h.copy_function(R_matrix_part[p - (p % step) + (p % neighbor)], R_matrix_temp_copy_0_part[p])
    end
    
    --__demand(__parallel)
    for p in mat_procs do
      h.copy_function(R_matrix_part[p], R_matrix_temp_copy_1_part[p])
    end 

    --3. Find QR factorization for the combined R blocks on each processor
    __demand(__parallel)
    for p in mat_procs do  
      blas.dgeqrf(p, m, R_matrix_temp_part[p], tau_part[p], work_part[p])
    end

    --4. Recover R on each processor (using original R matrix partition)
    __demand(__parallel)
    for p in mat_procs do
      h.get_R_matrix(p, r_vec, n, R_matrix_temp_part[p], R_matrix_part[p])
    end

    --5. Calculate Q matrices for this level of the tree
    __demand(__parallel)
    for p in mat_procs do
      blas.dorgqr(p, m, R_matrix_temp_part[p], tau_part[p], work_part[p])
    end

    var Q_level = Q_mat_prod[(k - 1)]
    __demand(__parallel)
    for p in mat_procs do
      h.get_Q_matrix(Q_level[p], R_matrix_temp_part[p])
    end

  end  --end tree levels

  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  print_time(ts_end - ts_start)

end --end main task

regentlib.start(main)

    --[[
    --random, full rank matrix for verification
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
    ]]--
  --[[
  c.printf("initial matrix\n")
  for i in matrix do
    c.printf("entry (%d, %d) = %f\n", i.x, i.y, matrix[i])
  end 
 
  if config.print_solution then
 
    --Print out final R solution
    c.printf("R_final:\n")
    var R_final = R_matrix_part[0]
    for i in R_final do
      c.printf("entry = (%d, %d) value = %f\n", i.x, i.y, R_final[i])
    end
  
    --Print out final Q entries
    c.printf("Q_final matrices:\n")
    for l = 0, (L + 1) do
      for p = 0, P do

        c.printf("L = %d, P = %d\n", l, p)
        if l == 0 then
          var current_matrix = matrix_part[p]
          for i in current_matrix do
            c.printf("entry = (%d, %d) value = %f\n", i.x, i.y, current_matrix[i])
          end
 
        else
          var current_matrix = Q_mat_prod[(l - 1)][p] 
          for i in current_matrix do
            c.printf("entry = (%d, %d) value = %f\n", i.x, i.y, current_matrix[i])
          end
        end

      end
    end

  end
  ]]--
