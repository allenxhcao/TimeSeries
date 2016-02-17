package MatrixFactorization;

import DataStructures.Matrix;

/*
 * A linear prediction link of a cell i,j of the original matrix to be reconstructed, 
 * is the dot product of row i and column j in the latent represantation matrixes
 */

public class LinearKernel extends Kernel {


	// in the linear prediction link the prediction is the dot product 
	// of the i'th column by the j'th row, <u,v>=SUM_k ( U(i,k)*Psi_i(k,j) )
	@Override
	public double Value( int i, int j) {
		
		return MatrixUtilities.getRowByColumnProduct(A, i, B, j); 
	}

	// known <a,b>=SUM_k ( U(i,k)*Psi_i(k,j) ), of A(i,k) is simply B(k,j) 
	// since the derivative of u is 1
	@Override
	public double Gradient_A(int i, int j, int k) {
		
		return B.get(k, j); 
	}

	// known <a,b>=SUM_k ( A(i,k)*B(k,j) ), gradient of B(k,j) is simply A(i,k) 
	// since the derivative of v is 1
	@Override
	public double Gradient_B(int i, int j, int k) {
		return A.get(i, k);
	}

}
