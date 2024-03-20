# numC
Porting Numpy to C while maintaining similar pythonic syntax and acceralating on GPU.


Functions Implemented:

currently only 2D and 1D arrays supported.

* Ones
* Zeros
* arange
* Reshape
* getter setter functions. (may be improved in a later update.)
* getter setter for list access also -> a[ np.arange(10), np.arange(10 ] (fetch diag elements.)
* Random
   - Randn
   - Rand
* Maximum -> np.maximum(gets element wise maximum
* arithmetic operators ( +, -, *, / )
* dot -> also added Tdot and dotT, to do A.T @ B without transposing overhead.
* T() -> returns transpose
* exp()
* log()
* comparison operators (>, >=, <=, <) -> returns an array of 1s and 0s, depending on whether the condition is satisfied.
* sum -> with axis argument
* max, min, argmax, argmin -> with axis argument
*  sort, argsort -> next update.
* sqrt, square, pow 