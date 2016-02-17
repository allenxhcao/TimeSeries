package MatrixFactorization;

import DataStructures.Matrix;

/*
 * A prediction link in the context C = prediction_link(A,B), 
 * where C is decomposed into A and B, 
 * ex: a typical prediction link is the dot product C = AB'
 */

public abstract class Kernel 
{
	Matrix A, B;
	
	// constructor
	public Kernel()
	{
		A = B = null;
	}
	
	// set the decomposed matrixes
	public void Setup(Matrix a, Matrix b)
	{
		A = a;
		B = b;
	}
	
	// predict the Value of the cell having coordinates i,j in the original matrix 
	public abstract double Value(int i, int j);
	
	// get the gradient of the Value of A(i,k) for a stochastic j
	public abstract double Gradient_A(int i, int j, int k);
	// get the gradient of the Value of B(k,j) for a stochastic i
	public abstract double Gradient_B(int i, int j, int k);
}
