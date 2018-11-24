import "regent"

local c = regentlib.c

local helper_exp = {}

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~        input arguments                 ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--function to check if a provided input file is valid
terra helper_exp.file_exists(filename : rawstring)
  var file = c.fopen(filename, "r")
  if file == nil then return false end
  c.fclose(file)
  return true
end

terra initialize_mat_vec_(P : int, m_vec : region(ispace(int1d), int), mat_vec_file : rawstring)
  --[[
  var count : uint
  var line = c.malloc(P)
  var i : int = 0
  
  var m_vec_file = c.fopen(mat_vec_file, "r")
   
  while c.getline(&line, &count, m_vec_file) ~= -1 do
    while count > 0 do
      c.sscanf(line, "%d", &m_vec[i])
      i = i + 1
      count = count - 1
    end
  end 
  
  c.fclose(mat_vec_file)
  ]]--
end

--task to retrieve the dimensions of the initial matrix subregion 
task helper_exp.initialize_mat_vec(P : int, m_vec : region(ispace(int1d), int), mat_vec_file : rawstring)
where reads writes(m_vec)
do
  --initialize_mat_vec_(P, m_vec, mat_vec_file)
end

--task to initialize the matrix region if no input file is provided
task helper_exp.initialize(y : int, m : int, matrix : region(ispace(f2d), double))
where reads(matrix), writes(matrix)
do

  fill(matrix, 1.0)

  for i in matrix do
    var y_shift = i.y - y*m
    if i.x == y_shift then
      matrix[i] += i.x
    end
  end

end

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~        domain coloring                 ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--task to partition the matrix for matrix multiplication
task helper_exp.color_matrix(m_vec : region(ispace(int1d), int), q_dim : region(ispace(int1d), int), matrix : region(ispace(f2d), double), 
                  mat_colors : ispace(f2d))

where reads(m_vec, q_dim)
do

  var start_point = 0

  var matrix_coloring = c.legion_domain_point_coloring_create()

  --color the matrix
  for i in mat_colors do

   var length : int = m_vec[i.y]
 
   c.legion_domain_point_coloring_color_domain(matrix_coloring, i, [rectf2d]{lo = [f2d] {x = 0, y = start_point}, hi = [f2d] {x = (q_dim[i.y] - 1), y = (start_point + length - 1)}})

   start_point += length
 
  end

  var mat_part = partition(disjoint, matrix, matrix_coloring, mat_colors)
  c.legion_domain_point_coloring_destroy(matrix_coloring)
  
  return mat_part

end

--task to partition the matrix when there is an odd number of submatrices in the level
task helper_exp.odd_partition(matrix : region(ispace(f2d), double), P_new : int, n : int, mat_colors : ispace(f2d))

  var start_point : int = 0
  var length : int

  var matrix_coloring = c.legion_domain_point_coloring_create()

  --color the matrix
  for i in mat_colors do
 
    if i.y ~= (P_new - 1) or P_new == 1 then
      length = 2*n 
    else 
      length = n 
    end
   
   c.legion_domain_point_coloring_color_domain(matrix_coloring, i, [rectf2d] {lo = [f2d] {x = 0, y = start_point},hi = [f2d] {x = (n - 1), y = (start_point + length - 1)}})

   start_point += length
  end

  var odd_part = partition(disjoint, matrix, matrix_coloring, mat_colors)
 
  c.legion_domain_point_coloring_destroy(matrix_coloring)
  
  return odd_part

end

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~         matrix extractions             ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--task to copy the upper diagonal results of the dgeqrf BLAS subroutine to corresponding R region
task helper_exp.get_R_matrix(y : int, m_vec : region(ispace(int1d), int), n : int, matrix : region(ispace(f2d), double), R_matrix : region(ispace(f2d), double))

where reads(matrix, m_vec), writes(R_matrix)
do
  var r_point : f2d
  var y_shift : int
  var offset : int = 0
  
  for j = 0, y do
    offset += m_vec[j]
  end

  for i in matrix do
    y_shift = i.y - offset
    if i.x >= y_shift then
      r_point = {x = i.x, y = i.y - offset + y*n}
      R_matrix[r_point] = matrix[i]
    end
  end
end

--task to copy the computed Q matrix from the dorgqr BLAS subroutine to the processor's "unique" region
task helper_exp.get_Q_matrix(x : int, n : int, Q_matrix : region(ispace(f2d), double), temp_matrix : region(ispace(f2d), double))

where reads(temp_matrix), writes(Q_matrix)
do
  var x_shift : int

  for i in temp_matrix do
    x_shift = i.x + x*n
    Q_matrix[f2d {x = x_shift, y = i.y}] = temp_matrix[i]
  end
end

--task to extract R from the results of the dgeqrf BLAS subroutine
task helper_exp.R_matrix_recovery(y : int, n : int, R_matrix : region(ispace(f2d), double))

where reads(R_matrix), writes(R_matrix)
do
  var y_shift : int

  for i in R_matrix do
    y_shift = i.y - y*(2*n)
    if i.x < y_shift then
      R_matrix[i] = 0
    end
  end

end

--task to copy a matrix from one region to another
task helper_exp.copy_function(source_region : region(ispace(f2d), double), destination_region : region(ispace(f2d), double), 
                   source_y : int, dest_y : int, temp_loc : int, n : int)
where reads(source_region, destination_region), writes(destination_region)
do

  var y_shift : int 
  var dest_lo_y = destination_region.bounds.lo.y + n*temp_loc

  for i in source_region do
    y_shift = (i.y - n*source_y) + dest_lo_y
    destination_region[f2d {x = i.x, y = y_shift}] = source_region[i]
  end

end

return helper_exp
