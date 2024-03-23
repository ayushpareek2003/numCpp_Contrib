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
   - arithmetic +, -, /, * (broadcasting supported)
   - unary negative -
   - assignment (=, deep copy)
   - comparison <, >, <=, =>, ==, != (broadcasting operator)
   - sum, max, min, argmax, argmin (axis), (broadcasting)
* destructor

### npFUnctions
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

### Random
* Randn -> normal distribution
* Rand -> default [0, 1], [lo, hi]
 