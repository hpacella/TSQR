import "regent"

local c = regentlib.c
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
local cstr = terralib.includec("string.h")

--BLAS info
local blas_lib = terralib.includecstring [[

  extern void dgeqrf_(int* M, int* N, double* A, int* lda, double* tau, double* work, int* lwork, int* info);

  extern void dorgqr_(int* M, int* N, int* K, double* A, int* lda, double* tau, double* work, 
                      int* lwork, int* info);

]]

terralib.linklibrary("libblas.so")
terralib.linklibrary("liblapack.so")

local blas_exp = {}

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~      BLAS/LAPACK functions              ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

terra min(a : int, b : int) : int
if a < b then return a
else return b end
end

terra max(a : int, b : int) : int
if a > b then return a
else return b end
end 

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end
local raw_ptr = raw_ptr_factory(double)

--function to retrieve a pointer for a region
terra get_raw_ptr(y : int, x : int, ny : int, nx : int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = y * ny
  rect.lo.x[1] = x * nx
  rect.hi.x[0] = (y + 1) * ny - 1
  rect.hi.x[1] = (x + 1) * nx - 1
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dgeqrf_terra(y: int, m_dim : int, n_dim : int, m_overall : int, mat_block_p : c.legion_physical_region_t, mat_block_f : c.legion_field_id_t,
                  tau_block_p : c.legion_physical_region_t, tau_block_f : c.legion_field_id_t,
                  work_block_p : c.legion_physical_region_t, work_block_f : c.legion_field_id_t)

  var n_ = array(n_dim)
  var m_ = array(m_dim)

  var raw_mat = get_raw_ptr(y, 0, m_dim, n_dim, mat_block_p, mat_block_f)
 
  var raw_tau = get_raw_ptr(y, 0, m_overall, 1, tau_block_p, tau_block_f)
 
  var lwork = array(m_overall*n_dim) 
  var raw_work = get_raw_ptr(y, 0, m_overall*n_dim, 1, work_block_p, work_block_f)

  var info : int[1]

  blas_lib.dgeqrf_(m_, n_, raw_mat.ptr, &(raw_mat.offset), raw_tau.ptr, raw_work.ptr, lwork, info) 

end 

--task to execute the BLAS dgeqrf subroutine on a region
task blas_exp.dgeqrf(y : int, m_overall : int, mat_block : region(ispace(f2d), double), tau_block : region(ispace(f2d), double), 
            work_block : region(ispace(f2d), double))

where reads writes(mat_block, tau_block, work_block)
do

  var bounds = mat_block.bounds
  var n_dim : int = bounds.hi.x - bounds.lo.x + 1
  var m_dim : int = bounds.hi.y - bounds.lo.y + 1
  
  dgeqrf_terra(y, m_dim, n_dim, m_overall, __physical(mat_block)[0], __fields(mat_block)[0], __physical(tau_block)[0], __fields(tau_block)[0], 
               __physical(work_block)[0], __fields(work_block)[0])

end

terra dorgqr_terra(y : int, m_dim : int, n_dim : int, m_overall : int, mat_block_p : c.legion_physical_region_t, mat_block_f : c.legion_field_id_t,
                  tau_block_p : c.legion_physical_region_t, tau_block_f : c.legion_field_id_t,
                  work_block_p : c.legion_physical_region_t, work_block_f : c.legion_field_id_t)

  var n_ = array(n_dim)
  var m_ = array(m_dim)
  var raw_mat = get_raw_ptr(y, 0, m_dim, n_dim, mat_block_p, mat_block_f)

  var raw_tau = get_raw_ptr(y, 0, m_overall, 1, tau_block_p, tau_block_f)

  var lwork = array(m_overall*n_dim)
  var raw_work = get_raw_ptr(y, 0, m_overall*n_dim, 1, work_block_p, work_block_f)

  var info : int[1]
  
  blas_lib.dorgqr_(m_, n_, n_, raw_mat.ptr, &(raw_mat.offset), raw_tau.ptr, raw_work.ptr, lwork, info)
 
end 

--task to execute the BLAS dorgqr subroutine on a region
task blas_exp.dorgqr(y : int, m_overall : int,  mat_block : region(ispace(f2d), double), tau_block : region(ispace(f2d), double), work_block : region(ispace(f2d), double))
where reads writes(mat_block, tau_block, work_block)
do

  var bounds = mat_block.bounds
  var n_dim : int = bounds.hi.x - bounds.lo.x + 1
  var m_dim : int = bounds.hi.y - bounds.lo.y + 1

  dorgqr_terra(y, m_dim, n_dim, m_overall, __physical(mat_block)[0], __fields(mat_block)[0], __physical(tau_block)[0], __fields(tau_block)[0], 
               __physical(work_block)[0], __fields(work_block)[0])

end

return blas_exp
