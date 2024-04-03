# numC
Porting Numpy to C while maintaining similar pythonic syntax and acceralating on GPU.


## Functions Implemented:

currently only 2D and 1D arrays supported.

### npArray
* constructor() -> default, parameter, deep copy
* reshape()
* size()
* copyFromCPU(), copyFromGPU()
* print()
* overloaded cout
* T()
* at() -> idx, (r, c), (ArrayIdx), (ArrayRows, ArrayCols)
* set() -> (idx, val), (r, c, val), (ArrayIdx, valArray), (ArrayRow, ArrayCol, ArrayVal)
* dot(), Tdot(), dotT()
* operators :
   - arithmetic +, -, /, * 
   - unary negative -
   - assignment (=, deep copy)
   - comparison <, >, <=, =>, ==, != 
   - sum, max, min, argmax, argmin (axis).

      all operators support broadcasting
* destructor

### npFunctions
* Ones
* Zeros
* arange
* Maximum, gets element wise maximum (broadcasting supported)
* Minimum
* exp()
* log()
* square()
* sqrt()
* pow() 
* shuffle() => shuffles an array's values randomly. (permutations)
* array_split(ar, num_parts) => divides array into num_parts, even if parts are unequal. returns vector of arrays.
### Random
* Randn -> normal distribution
* Rand -> default [0, 1], [lo, hi]
 
