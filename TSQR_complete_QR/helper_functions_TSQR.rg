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
--[[
task helper_exp.initialize_mat_vec(P : int, m_vec : region(ispace(int1d), int), mat_vec_file : rawstring)
where reads writes(m_vec)
do
  --initialize_mat_vec_(P, m_vec, mat_vec_file)
end
]]--

--task to initialize the matrix region if no input file is provided
task helper_exp.initialize(x : int, m : int, matrix : region(ispace(int2d), double))
where reads(matrix), writes(matrix)
do

  fill(matrix, 1.0)

  for i in matrix do
    var x_shift = i.x - x*m
    if i.y == x_shift then
      matrix[i] += i.y
    end
  end

end

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~        domain coloring                 ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--task to partition the matrix for matrix multiplication
task helper_exp.color_matrix(m_vec : region(ispace(int1d), int), q_dim : region(ispace(int1d), int), matrix : region(ispace(int2d), double), 
                  mat_colors : ispace(int1d))

where reads(m_vec, q_dim)
do

  var start_point = 0

  var matrix_coloring = c.legion_domain_point_coloring_create()

  --color the matrix
  for i in mat_colors do

   var length : int = m_vec[i]
 
   c.legion_domain_point_coloring_color_domain(matrix_coloring, i, rect2d {lo = {x = start_point, y = 0}, 
                                                                   hi = {x = (start_point + length - 1), y = q_dim[i] - 1}})

   start_point += length
 
  end

  var mat_part = partition(disjoint, matrix, matrix_coloring, mat_colors)
  c.legion_domain_point_coloring_destroy(matrix_coloring)
  
  return mat_part

end

--task to partition a 2D region by processor id
task helper_exp.processor_matrix_part(dim_x : int, levels : int, matrix : region(ispace(int2d), double), mat_colors : ispace(int1d))

  var start_point = matrix.bounds.lo.x
  var length : int = dim_x

  var matrix_coloring = c.legion_multi_domain_point_coloring_create()

  --color the matrix
  for j = 0,levels do

    for i in mat_colors do

      c.legion_multi_domain_point_coloring_color_domain(matrix_coloring, i, rect2d {lo = {x = start_point, y = matrix.bounds.lo.y}, 
                                                                   hi = {x = (start_point + length - 1), y = matrix.bounds.hi.y}})

      start_point += length

    end
 
  end

  var mat_part = partition(disjoint, matrix, matrix_coloring, mat_colors)
  c.legion_multi_domain_point_coloring_destroy(matrix_coloring)
  
  return mat_part

end

--task to create a 1D partition of a 2D region
task helper_exp.equal_matrix_part(dim_x : int, matrix : region(ispace(int2d), double), mat_colors : ispace(int1d))

  var start_point = matrix.bounds.lo.x
  var length : int = dim_x

  var matrix_coloring = c.legion_domain_point_coloring_create()

  --color the matrix
  for i in mat_colors do
    
    c.legion_domain_point_coloring_color_domain(matrix_coloring, i, rect2d {lo = {x = start_point, y = matrix.bounds.lo.y}, 
                                                                   hi = {x = (start_point + length - 1), y = matrix.bounds.hi.y}})

    start_point += length
 
  end

  var mat_part = partition(disjoint, matrix, matrix_coloring, mat_colors)
  c.legion_domain_point_coloring_destroy(matrix_coloring)
  
  return mat_part

end

--task to create a 1D partition of a 2D region
task helper_exp.copy_matrix_part(dim_x : int, offset : int, matrix : region(ispace(int2d), double), mat_colors : ispace(int1d))

  var start_point = matrix.bounds.lo.x + offset
  var length : int = dim_x

  var matrix_coloring = c.legion_domain_point_coloring_create()

  --color the matrix
  for i in mat_colors do
   
    c.legion_domain_point_coloring_color_domain(matrix_coloring, i, rect2d {lo = {x = start_point, y = matrix.bounds.lo.y}, 
                                                                   hi = {x = (start_point + length/2 - 1), y = matrix.bounds.hi.y}})

    start_point += length
 
  end

  var mat_part = partition(disjoint, matrix, matrix_coloring, mat_colors)
  c.legion_domain_point_coloring_destroy(matrix_coloring)
  
  return mat_part

end

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~         matrix extractions             ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--task to copy the upper diagonal results of the dgeqrf BLAS subroutine to corresponding R region
task helper_exp.get_R_matrix(p : int, m_vec : region(ispace(int1d), int), n : int, matrix : region(ispace(int2d), double), R_matrix : region(ispace(int2d), double))

where reads(matrix, m_vec), writes(R_matrix)
do
  var r_point : int2d
  var x_shift : int
  var offset : int = 0
  
  for j = 0, p do
    offset += m_vec[j]
  end

  for i in matrix do
    x_shift = i.x - offset
    if i.y >= x_shift then
      r_point = {x = i.x - offset + p*n, y = i.y}
      R_matrix[r_point] = matrix[i]
    end
  end
end

--task to copy the computed Q matrix from the dorgqr BLAS subroutine to the processor's "unique" region
task helper_exp.get_Q_matrix(Q_matrix : region(ispace(int2d), double), temp_matrix : region(ispace(int2d), double))

where reads(temp_matrix), writes(Q_matrix)
do
  var x_shift : int = Q_matrix.bounds.lo.x - temp_matrix.bounds.lo.x

  for i in temp_matrix do
    Q_matrix[{x = i.x + x_shift, y = i.y}] = temp_matrix[i]
  end
end

--task to copy a matrix from one region to another
task helper_exp.copy_function(source_region : region(ispace(int2d), double), destination_region : region(ispace(int2d), double))
where reads(source_region), writes(destination_region)
do

  var x_shift : int = destination_region.bounds.lo.x - source_region.bounds.lo.x 

  for i in source_region do
    destination_region[{x = i.x + x_shift, y = i.y}] = source_region[i]
  end

end

return helper_exp
