Dependencies :
------------------
	1) C++11 Compiler (suggested)

Build
-----------------
g++ multi-threaded-matrix-mul.cpp -std=c++11 -pthread -o excecutableName

Run
------------------
Mandatory params

	M1 = # of rows in Matrix1
	N1 = # of columns in Matrix1

	M2 = # of rows in Matrix2
	N2 = # of columns in Matrix2

	Op ->  0 : Transpose
	       1 : Matrix Multiplication

Optional params

	test-> 0 : Test With Random number generator
       	       1 : Test with all values in matrices = 1 , Easy to debug	
1) Transpose
	1) ./executableName M1 N1 0

2) Matrix Multiplication
	1)  ./executableName M1 N1 M2 N2 1 (Will generate a matrix Inputs with random numbers)
	2)  ./executableName M1 N1 M2 N2 1 1 (Will generate a matrix Inputs with all ones)

Restrictions
	None as long as N1==M2 any matrix multiplication is allowed and memory is available to 	
	store the array
