package MatrixFactorization;

import DataStructures.DataSet;
import java.util.List; 

import DataStructures.Matrix;

/*
 * An abstract class that should provide the basic model interface 
 * for a matrix factorization model X = UV' and also for the supervised 
 * label decomposition Y = UW'
 * 
 * We assume that the model will described a unified model. 
 * 
 */
public abstract class MatrixFactorizationModel
{
	/*
	 * The matrix X and reduced matrixes U & Psi_i, s.t approximately 
	 * X = UxV' and Y = UW'
	 */
	protected Matrix X, Y, U, V, W;
	
	protected DataSet trainSet, testSet;
	 
	/*
	 * Loss weights matrix
	 */
	Matrix Weights;
	
	public enum LossType {
	    Logistic, Linear, SmoothHinge 
	}

	protected LossType labelsLossType;
        
	/*
	 * The number of dimensions of the latent factors
	 * @param M_i matrix to decompose
	 * @param L the label matrix for the rows of M_i 
	 */
	public int latentDim;
	
	public MatrixFactorizationModel(int noLatentFactors)
	{
		latentDim = noLatentFactors;
		labelsLossType = LossType.Logistic;
	}
	
	
	/*
	 * Train the decomposition by using the training set matrixes X and Y
	 * return the error rate of the label relation over the test set
	 */
	public abstract double Factorize( 
	        DataSet trainSet, DataSet testSet, 
	        DataSet latentTrainSet, DataSet latentTestSet );

    /* factorize a dataset in an unsupervized fashion and return 
     * the latent factorization
     */
    public abstract DataSet Factorize( DataSet dataset );
	
	/*
	 * Get U & Psi_i & W matrix, typically after solve
	 */
	public Matrix getU()
	{
		return U;
	}
	
	public Matrix getV()
	{
		return V;
	}
	
	public Matrix getW()
	{
		return W;
	}
	
        
	
	//private Matrix UTrainingTemp
	
	
}
