package MatrixFactorization;

import Utilities.GlobalValues;
import DataStructures.Matrix;

/*
 * A square loss error, with some member methods dedicated to the gradients of matrix factorizations
 */
public class SquareErrorLoss extends LossFunction 
{

	// the loss of a square error is the square of their differences
	@Override
	public double Loss(int i, int j) 
	{
            // if the cell is not observed the loss is zero
            if( C.get(i, j) != GlobalValues.MISSING_VALUE )
		return Math.pow( C.get(i, j) - kernel.Value(i, j), 2);
            else 
                return 0;
	}

	// the gradient of u_ik at a stochastic j is equal to 
	// -2 * (X-PL(U,Psi_i))* grad_U( PL(U,Psi_i) )

	@Override
	public double Gradient_A(int i, int j, int k) 
	{
		//double error_ij = C.getMeanValue() + C.get(i, j) - kernel.Value(i, j);
		double error_ij = C.get(i, j) - kernel.Value(i, j);
		
		return -2 * error_ij * kernel.Gradient_A(i, j, k);
	}

	// the gradient of u_ik at a stochastic j is equal to 
	// -2 * (X-PL(U,Psi_i))* grad_V( PL(U,Psi_i) )

	@Override
	public double Gradient_B(int i, int j, int k) 
	{
		//double error_ij = C.getMeanValue() + C.get(i, j) - kernel.Value(i, j);
		double error_ij = C.get(i, j) - kernel.Value(i, j);
		
		return -2 * error_ij * kernel.Gradient_B(i, j, k);
	}

}
