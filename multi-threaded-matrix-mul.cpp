/*

Reference for this code is taken from two different sources as mention below
a) https://www.researchgate.net/publication/312523647_OPTIMIZING_MATRIX_MULTIPLICATION_USING_MULTITHREADING

b) http://bitsploit.blogspot.com/2017/07/optimizing-matrix-multiplication-using.html


1) No multi-threading is applied on transpose logic. Only inplace approach is used if the input 
   matrix is square.

2) Idea of multi-threading has been taken from (a). 
   Current logic is to only create as many number of threads that can run on the system parallely 
   on each core and deviding the workload on these threads. Also ensure there are fairly high 
   amount of work load exists on each thread. Eg: my current system supports max 4 threads  
   overall. Some systems with better configuration will have more ability to create threads, hence 
   workload will be devided effectively. Also, the logic ensures equal amount of work is done by  
   each thread. Based on the number of rows say M number of threads created is 
   M/MAX_SUPPORTED_THREAD in that system. However, if the division is not integer a GCD is 
   calculated to effectively distribute jobs accross threads.

3) Code for Strassen logic for matrix multiplication which has order O(n^log2(7)) has been shared 
   from the author "Malsha Ranawaka" link given in (b). The entire implementation of her code is 
   done after few bug fixes. In her code the final matrix multiplication logic uses standard 
   matrix multication technique with complexity O(n^3). In this code the logic remains same except 
   for the part that the power of multi-core parallel excecution using threads is done. Some of  
   the results found shown below is interesting.


	shubham@shubham-ubuntu:~/Desktop/BrainCorp$ diff file1.txt file2.txt 
	2054c2054
	< Average time taken to execute matrices of size - 2048 : 81
	---
	> Average time taken to execute matrices of size - 2048 : 46

	file1.txt and file2.txt store the result of matrix multiplication.

	1st shows exceution time of Malsha's code with a square matrix of size 2048. 
	2nd shows exceution time of Malsha's code with multi-thread approach implemented by me.


   A significant time difference has been observed. Though it's not at par with fastest matrix    
   multiplication logic done by "Coppersmith–Winograd algorithm" and others. It is certainly    
   improved version over traditional logic. Also, any order of matrix muliplication is supported 
   with this code as long as N1==M2. 

 
*/



#include <bits/stdc++.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <math.h>
#include <ctime>



std::mutex mtx;


#define Threshold 128


struct threadOp{
	int numThreads;
	int operationperThread;
};

int getNextPowerOfTwo(int n)
{
	return pow(2, int(ceil(log2(n))));
}


void padZeros(std::vector< std::vector<double> > &newA, 
	       std::vector< std::vector<double> > &a, int M1,int N1)
{
  
	for (int i=0; i<M1; i++){
		for (int j=0; j<N1; j++){
			newA[i][j] = a[i][j];
		}
	}
}


void MatrixTransposeInplace(std::vector< std::vector<double> > &a, 
		int n)
{
	
	for(int i=0;i<n;i++)
	{
		for(int j=i+1;j<n;j++)
		{
			double temp = a[i][j];
			a[i][j] = a[j][i];
			a[j][i] = temp;
		}
	}
}
 

void MatrixTranspose(std::vector< std::vector<double> > &a, 
		std::vector< std::vector<double> > &c, 
		int M1,int N1)
{
	for(int i=0;i<M1;i++)
	{
		for(int j=0;j<N1;j++)
		{
			c[j][i] = a[i][j];
		}
	}
}
 


void MatrixMutiplication(const std::vector < std::vector <double> > &a, 
		const std::vector < std::vector <double> > &b,
		std::vector < std::vector <double> > &c,
		int A_start,int A_end,int B_start,int B_end,int C_start,
		int C_end)
 
{
	/*	
	mtx.lock();
	std::cout<<A_start<<A_end<<B_start<<B_end<<C_start<<C_end<<std::endl;
	mtx.unlock();
	*/	
	
	for(int i=A_start;i<A_end;i++){
		for(int j = B_start;j<B_end;j++){
			double sum = 0;
			for(int k=C_start;k<C_end;k++){
				sum += a[i][k]*b[k][j];
			}
		c[i][j] = sum; 
		}	
	}
		
}

int GCD(int num1,int num2){
	if (num2==0)
		return num1;
	else
		return GCD(num2,num1%num2);
}

threadOp OperationPerThread(int M,int numThreads)
{
 
	threadOp threadInfo;

	if(M%numThreads)
	{
		threadInfo.numThreads = numThreads;
		threadInfo.operationperThread = M/numThreads;
	}
	else
	{
		int numreqThreads = GCD(M,numThreads);
		threadInfo.numThreads = numreqThreads;
		threadInfo.operationperThread = M/numreqThreads;
	}
	return threadInfo;

}

int getMaxOrder(int M1 ,int N2 , int M2){
	int maxorder ;
	maxorder = (M1>N2)?M1:N2;
	maxorder = (maxorder>M2)?maxorder:M2;
	return maxorder;
}
	

void multiplyMatStandard(std::vector< std::vector<double> > &a, 
		std::vector< std::vector<double> > &b,
		std::vector< std::vector<double> > &c, 
		int M1,int N2,int M2)
{
	int MAXTHREADS = std::thread::hardware_concurrency();
	int n = getMaxOrder(M1,N2,M2);
	int newSize = getNextPowerOfTwo(n);
	//std::cout<<"New Size"<<newSize<<"\n";

	
	std::vector< std::vector<double> > 
			newA(newSize, std::vector<double>(newSize)), 
			newB(newSize, std::vector<double>(newSize)), 
			newC(newSize, std::vector<double>(newSize));
	
	if(M1==M2 && M1==N2 && M1==newSize)
	{   
		newA = a;
		newB = b;
	}
	else
	{
		padZeros(newA,a,M1,M2);
		padZeros(newB,b,M2,N2);
	}

	
	threadOp threadInfo = OperationPerThread(newSize,MAXTHREADS);
	int numThreads = threadInfo.numThreads;
	int operationperthread = threadInfo.operationperThread;
	//std::cout<<"Num of threads "<<numThreads<<std::endl;
	//std::cout<<"Num of operations per thread "<<operationperthread<<std::endl;
	int count = 0;

	std::vector<std::thread> workers;	
		
	if (operationperthread<1)
		MatrixMutiplication(std::ref(a),
			std::ref(b),
			std::ref(c),
			0,M1,0,N2,0,M2);
	
	while(count<newSize && operationperthread>=1){
	   for(int i=0;i<numThreads;i++)
		{
	  		int countbegin = count;
			int countend = count + operationperthread;
			count = count + operationperthread;
							
			if (countend>newSize){
			    countend = newSize;
			    }

			//std::cout<<countbegin<<"\t"<<countend<<std::endl;
			
				
			workers.push_back(std::thread(MatrixMutiplication,
			std::ref(newA),
			std::ref(newB),
			std::ref(newC),
			countbegin,
			countend,
			0,newSize,0,newSize));

			if(count>=newSize)
			   break;
			
			
		}
		
		
	}
	
	std::for_each(workers.begin(), workers.end(), [](std::thread &t) 
 	{
        	t.join();
    	});
	
	
	/*
	
	for(int i=0; i<newSize; i++){
		for(int j=0; j<newSize; j++){
			std::cout<<newC[i][j]<<"\t";
		}
		std::cout<<"\n";
	}
	*/

	for(int i=0; i<M1; i++){
			for(int j=0; j<N2; j++){
				c[i][j] = newC[i][j];
			}
		}
		
	
	
}


void initMat(std::vector< std::vector<double> > &a, int M1 , int N1 ,int test)
{
		// initialize matrices and fill them with random values
		for (int i = 0; i < M1; ++i) {
			for (int j = 0; j < N1; ++j) {
				if (!test)
					a[i][j] = (double)rand()/RAND_MAX*10;
				else
					a[i][j] = 1;
					
			}
		}
}


void add(std::vector< std::vector<double> > &a, std::vector< std::vector<double> > &b,
	 std::vector< std::vector<double> > &resultMatrix, int n)
{
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] + b[i][j];
		}
	}
}

void subtract(std::vector< std::vector<double> > &a, std::vector< std::vector<double> > &b,
	      std::vector< std::vector<double> > &resultMatrix, int n)
{
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] - b[i][j];
		}
	}
}

void multiplyStrassen(std::vector< std::vector<double> > &a,
	std::vector< std::vector<double> > &b, 
	std::vector< std::vector<double> > &c, int M1 , int N2=0 , int M2=0)
{	
	if(M1<=Threshold){
		multiplyMatStandard(a, b, c, M1 , M1 , M1);
	}
	else{
		//std::cout<<M1<<N2<<M2<<"\n";
		int n = getMaxOrder(M1,N2,M2);
		//std::cout<<"Order of n"<<n;
		int newSize = getNextPowerOfTwo(n);
		//std::cout<<"Order of newSize"<<newSize;
		
		std::vector< std::vector<double> > 
			newA(newSize, std::vector<double>(newSize)), 
			newB(newSize, std::vector<double>(newSize)), 
			newC(newSize, std::vector<double>(newSize));
		if(n==newSize){   //matrix size is already a power of two
			newA = a;
			newB = b;
		}
		else{
			padZeros(newA,a,M1,M2);
			padZeros(newB,b,M2,N2);
		}
		
	
		int blockSize = newSize/2;  //size for a partition matrix
		
		std::vector<double> block (blockSize);
		std::vector< std::vector<double> > 
			//partitions of newA//
			a11(blockSize, block), a12(blockSize, block), 
			a21(blockSize, block), a22(blockSize, block), 
			//partitions of newB//
			b11(blockSize, block), b12(blockSize, block), 
			b21(blockSize, block), b22(blockSize, block), 
			//partitions of newC//
			c11(blockSize, block), c12(blockSize, block), 
			c21(blockSize, block), c22(blockSize, block), 
			//matrices storing intermediate results//
			aBlock(blockSize, block), bBlock(blockSize, block),
			//set of submatrices derived from partitions//
			m1(blockSize, block), m2(blockSize, block), 
			m3(blockSize, block), m4(blockSize, block),  
			m5(blockSize, block), m6(blockSize, block), 
			m7(blockSize, block);  
		
	

		//partition matrices
		for (int i=0; i<blockSize; i++){
			for (int j=0; j<blockSize; j++){
				a11[i][j] = newA[i][j];
				a12[i][j] = newA[i][j+blockSize];
				a21[i][j] = newA[i+blockSize][j];
				a22[i][j] = newA[i+blockSize][j+blockSize];
				b11[i][j] = newB[i][j];
				b12[i][j] = newB[i][j+blockSize];
				b21[i][j] = newB[i+blockSize][j];
				b22[i][j] = newB[i+blockSize][j+blockSize];
			}
		}
	    
		//compute submatrices
		//m1 = (a11+a22)(b11+b22)
		add(a11, a22, aBlock, blockSize);
		add(b11, b22, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m1, blockSize,blockSize,blockSize);
		
		//m2 = (a21+a22)b11
		add(a21, a22, aBlock, blockSize);
		multiplyStrassen(aBlock, b11, m2, blockSize,blockSize,blockSize);
		
		//m3 = a11(b12-b22)
		subtract(b12, b22, bBlock, blockSize);
		multiplyStrassen(a11, bBlock, m3, blockSize,blockSize,blockSize);
		
		//m4 = a22(b21-b11)
		subtract(b21, b11, bBlock, blockSize);
		multiplyStrassen(a22, bBlock, m4, blockSize,blockSize,blockSize);
		
		//m5 = (a11+a12)b22
		add(a11, a12, aBlock, blockSize);
		multiplyStrassen(aBlock, b22, m5, blockSize,blockSize,blockSize);
		
		//m6 = (a21-a11)(b11+b12)
		subtract(a21, a11, aBlock, blockSize);
		add(b11, b12, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m6, blockSize,blockSize,blockSize);
		
		//m7 = (a12-a22)(b12+b22)
		subtract(a12, a22, aBlock, blockSize);
		add(b12, b22, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m7, blockSize,blockSize,blockSize);
		
		//calculate result submatrices
		//c11 = m1+m4-m5+m7
		add(m1, m4, aBlock, blockSize);
		subtract(aBlock, m5, bBlock, blockSize);
		add(bBlock, m7, c11, blockSize);
		
		//c12 = m3+m5
		add(m3, m5, c12, blockSize);
		
		//c21 = m2+m4
		add(m2, m4, c21, blockSize);
		
		//c22 = m1-m2+m3+m6
		subtract(m1, m2, aBlock, blockSize);
		add(aBlock, m3, bBlock, blockSize);
		add(bBlock, m6, c22, blockSize);
		
		//calculate final result matrix
		for(int i=0; i<blockSize; i++){
			for(int j=0; j<blockSize; j++){
				newC[i][j] = c11[i][j];
				newC[i][blockSize+j] = c12[i][j];
				newC[blockSize+i][j] = c21[i][j];
				newC[blockSize+i][blockSize+j] = c22[i][j];
			}
		}
		
		//remove additional values from expanded matrix
		for(int i=0; i<M1; i++){
			for(int j=0; j<N2; j++){
				c[i][j] = newC[i][j];
			}
		}
		
	}
	
}

int main(int argc, char *argv[])
{
	int M1 = 17;
	int N1 = 17;
	int M2 = 17;
	int N2 = 17;
	int op = 0;
	int test = 0;
	
	
	if(argc>1){
		if(argc==4)
			{
			M1 = atoi(argv[1]);
			N1 = atoi(argv[2]);
			op = atoi(argv[3]);
			}
		else if(argc==6 or argc==7)
			{
			M1 = atoi(argv[1]);
			N1 = atoi(argv[2]);
			M2 = atoi(argv[3]);
			N2 = atoi(argv[4]);
			op = atoi(argv[5]);
			if (argc==7)
				test = atoi(argv[6]);
			}
		else	{
			std::cout<<" Please provide appropiriate num of arguments\n Refer ReadMe file for more Info\n";
			return -1;
		    	}
	}
	else{
		std::cout<<"Please provide Input\n";
		return -1;
	    }

	std::vector < std::vector <double> > matrix1(M1 , std::vector<double>(N1 , 0.0));
	std::vector < std::vector <double> > matrix2(M2 , std::vector<double>(N2 , 0.0));
	//std::cout<<M1<<N1<<M2<<N2<<op<<test<<"\n";

	initMat(matrix1,M1,N1,test);
	initMat(matrix2,M2,N2,test);
	double startTime;
	double elapsedTime;

			
	if(op==1)
		{
		if(N1!=M2)
			{
		std::cout<<"Matrix multiplication condition not statified ensure N1==M2\n";	
			return -1;
		}	
		
		startTime = 0;			
		startTime = time(0);
		
		
		
		std::vector < std::vector <double> > matrix3(M1 , std::vector<double>(N2 , 0.0));
		if (M1<Threshold)
			multiplyMatStandard(matrix1,matrix2,matrix3,M1,N2,M2);

		if (M1>=Threshold)
			multiplyStrassen(matrix1,matrix2,matrix3,M1,N2,M2);
		
		elapsedTime = time(0) - startTime;
		std::cout << "Average time taken to execute matrices of size M*N " 
		<<M1<<" "<<N1<<" "<<M2<<" "<<N2<<" "<<elapsedTime<<std::endl;
			std::cout<<"Matrix multiplication result\n";
			
			for(int i=0; i<M1; i++){
				for(int j=0; j<N2; j++){
					std::cout<<matrix3[i][j]<<"\t";
				}
				std::cout<<"\n";
			}
		
		

	}
	else if (op==0)
	{
		// This section of code will generate matrix transpose 
		std::cout<<"Transpose of a Matrix\n";
		if (M1==N1)
		{
		
			MatrixTransposeInplace(matrix1,M1);
			
			for(int i=0; i<M1; i++){
				for(int j=0; j<M1; j++){
					std::cout<<matrix1[i][j]<<"\t";
				}
				std::cout<<"\n";
				}
			
		}
		else
		{
		std::vector < std::vector <double> > matrix3(N1 , std::vector<double>(M1 , 0.0));
		MatrixTranspose(matrix1,matrix3,M1,N1);

		for(int i=0; i<N1; i++){
				for(int j=0; j<M1; j++){
					std::cout<<matrix3[i][j]<<"\t";
				}
				std::cout<<"\n";
			}
		}

	}
	else
	    std::cout<<"Please specify correct operation\n";
		
		
	return 0;
}

