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
--~~~~                 MAIN                   ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

task main()

  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  --~~~~   input arguments/domain construction  ~~~~
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
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
    for i in mat_colors do
      h.initialize(i.y, m_vec[i.y], matrix_part[i])
    end
  end

  --build R matrix
  var R_matrix_dim : f2d = {x = n, y = P*n}
  var R_matrix = region(ispace(f2d, R_matrix_dim), double)
  fill(R_matrix, 0.0)

  --partition the R matrix
  var r_blocks = {x = 1, y = P}
  var R_matrix_part = partition(equal, R_matrix, ispace(f2d, r_blocks))

  --tau region (used to find Q matrices)
  var tau_dim : f2d = {x = m, y = P}
  var tau = region(ispace(f2d, tau_dim), double)
  var tau_blocks : f2d = {x = 1, y = P}
  var tau_part = partition(equal, tau, ispace(f2d, tau_blocks)) 
            
  --work region (used as scratch by BLAS subroutines)
  var work_dim : f2d = {x = m*n, y = P}
  var work = region(ispace(f2d, work_dim), double)
  var work_blocks : f2d = {x = 1, y = P}
  var work_part = partition(equal, work, ispace(f2d, work_blocks)) 

  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  --~~~         parallel TSQR                  ~~~~
  --~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  --number of levels in the tree
  var L : int = cmath.ceil(cmath.log2(P)) 
  var P_new : int = P
  var P_prev : int = P 

  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()

  for k = 0, (L+1) do --successive levels of the tree
 
    if k ~= 0 then

      P_new = cmath.ceil([double](P_new)/2)
      
      var r_colors_new = ispace(f2d, {x = 1, y = P_new})

      --Paritioning depends on if P_new is odd or even. Steps (not necessarily sequential):
      --1. Repartition to combine adjacent (n x n) R matrices into (2n x n) matrices.
      --2. Find the QR factorization of these new matrices.
      --3. Extract R values, fill in necessary zeros.
      --4. Shift R values upward.

      if (P_prev) % 2 == 0 and P_new ~= 1 then

        --1. Repartition to form new R matrices
        var tree_part = partition(equal, R_matrix, r_colors_new)
   
        --2. Find QR factorization for the R blocks of the tree_part partition
        for y = 0, P_new do  
          blas.dgeqrf(y, m, tree_part[f2d {x = 0, y = y}], tau_part[f2d {x = 0, y = y}], work_part[f2d {x = 0, y = y}])
        end

        --3. Recover R (using original R matrix partition)
        for y = 0, 2*P_new, 2 do
          h.R_matrix_recovery(y, n, R_matrix_part[f2d {x = 0, y = y}])
        end

        --4. shift R matrices into position for next partitioning
        var shift_val : int = 1
        for y = 2, P, 2 do
          h.copy_function((R_matrix_part[f2d {x = 0, y = y}]), (R_matrix_part[f2d {x = 0, y = (y - shift_val)}]), y, shift_val, 0, n)
          shift_val += 1
        end
        
      else
        --1. Repartition to form new R matrices
        var tree_part = h.odd_partition(R_matrix, P_new, n, r_colors_new)
        
        --2. Find QR factorization for the R blocks of the tree_part partition
        var P_bound : int
        if P_new == 1 then P_bound = P_new
        else P_bound = P_new - 1 end

        for y = 0, P_bound do 
          blas.dgeqrf(y, m, tree_part[f2d {x = 0, y = y}], tau_part[f2d {x = 0, y = y}], work_part[f2d {x = 0, y = y}])
        end 

        --3. Recover R (using original R matrix partition)
        for y = 0, P_new, 2 do
          h.R_matrix_recovery(y, n, R_matrix_part[f2d {x = 0, y = y}])
        end

        --4. shift R matrices into position for next partitioning
        if P_new ~= 1 then
          var shift_val : int = 1
          for y = 2, (2*P_new-1), 2 do
            h.copy_function((R_matrix_part[f2d {x = 0, y = y}]), (R_matrix_part[f2d {x = 0, y = (y - shift_val)}]), y, shift_val, 0, n)
            shift_val += 1
          end
        end

      end

      P_prev = P_new

    else  --initial QR factorization

      --find Q vectors, R entries 
      for y = 0,P do
        blas.dgeqrf(y, m, matrix_part[f2d {x = 0, y = y}], tau_part[f2d {x = 0, y = y}], work_part[f2d {x = 0, y = y}])
      end
      
      --copy values to R matrix
      for y = 0,P do
        h.get_R_matrix(y, m_vec, n, matrix_part[f2d {x = 0, y = y}], R_matrix_part[f2d {x = 0, y = y}])
      end

    end

  end  --end tree levels

  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Total Time : %f s\n", 1e-3*(ts_end - ts_start))

  
  --Print out final R solution
  c.printf("R_final:\n")
  var R_final = R_matrix_part[f2d {x = 0, y = 0}]
  for i in R_final do
    c.printf("entry = (%d, %d) value = %f\n", i.x, i.y, R_final[i])
  end
  

end --end main task
regentlib.start(main)
