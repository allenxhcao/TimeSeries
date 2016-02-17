package MatrixFactorization;

import DataStructures.Matrix;

/*
 * A loss function sceleton for reconstructing matrix C based on input kernel(A,B) 
 */
public abstract class LossFunction 
{
	Matrix C;
	Kernel kernel; 
	
	/*
	 * Set up the neccessary data for the loss function, i.e the 
	 * X and the prediction_link(U,Psi_i') 
	 */
	public void Setup(Matrix c, Kernel pl)
	{
		C = c;
		kernel = pl;
	}
	
	// get the loss of the predicted compared with observed 
	public abstract double Loss(int i, int j);
	
	// get the gradient of the value of U(i,k) for a stochastic j
	public abstract double Gradient_A(int i, int j, int k);
	// get the gradient of the value of Psi_i(k,j) for a stochastic i
	public abstract double Gradient_B(int i, int j, int k);
	
}
