import "regent"
local c = regentlib.c
local cmath = terralib.includec("math.h")

--Terra struct that contains all necessary information about the initial matrix A

struct Config {

  --matrix dimensions
  m : int, --number of rows
  n : int,  --number of columns
  P : int, --number of processors

  --matrix initialization
  read_input_matrix_file : bool,
  input_matrix_file : &int8,
  nonuniform_mat_vec : bool,
  --mat_vec_array : int[P],
  
  --options for printing solution
  print_solution : bool,
}

terra Config:initialize()

  --matrix dimensions
  self.m = 10
  self.n = 4
  self.P = 1

  --matrix initialization
  self.read_input_matrix_file = false
  --self.input_matrix_file = 

  self.nonuniform_mat_vec = false
  --self.mat_vec_file = array()

  --printing solution 
  self.print_solution = false
end

terra Config:checks()

  --check provided P value
  if cmath.floor(cmath.pow(2,self.P)) ~= cmath.ceil(cmath.pow(2, self.P)) then
    c.printf("The total number of processors P must be a power of 2.\n")
    c.abort()
  end

end
  
