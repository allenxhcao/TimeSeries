package Regression;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import Classification.Kernel;
import Classification.Kernel.KernelType;
import DataStructures.Matrix;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class LSSVM 
{
	// the train predictors and train targets
	public Matrix trainPredictors;
	public double [] trainTargets;
	// the index of the target variable
	public int targetVariableIndex;
	
	// the gamma parameter of the svm
	public double lambda;
	public double beta;
	
	// the alphas of the dual solution
	public double [] alphas;
	// the bias parameter 
	public double b;
	
	// the kernel
	public Kernel kernel;

	// the gaussian kernel parameter, the sigmasquare
	public double sig2;
	// the polynomial degree
	public int d;

	// store the linear systems of equations matrices
	// for not having to initialize them in repeated train calls
	RealMatrix A, B;
	
	public LSSVM()
	{
		// initializr a polynomial kernel
		kernel = new Kernel(Kernel.KernelType.Polynomial);
		kernel.degree = d; 
		
		beta = 1.0;
		
		// set the predictors to null
		trainPredictors = null;
		trainTargets = null; 
	}
	
	
	/*
	 * Train the Least Square SVM with fully observed targets
	 */
	public void Train(Matrix trainPredictors, double [] trainTargets) 
	{
		
		int [] trainTargetIndices = new int[trainTargets.length];
		for(int i = 0; i < trainTargets.length; i++)
			trainTargetIndices[i] = i; 
		
		Train(trainPredictors, trainTargets, trainTargetIndices);
	}
	
	/*
	 * Train the Least Square SVM, with missing values
	 */
	
	public void Train(Matrix trainPredictors, double [] trainTargets, int [] observableIndices) 
	{
		if( trainPredictors == null || trainTargets == null )
		{
			Logging.println("LeastSquareSVM::Train - Training data null", LogLevel.PRODUCTION_LOG);
			return;
		}
		else if( kernel == null )
		{
			Logging.println("LeastSquareSVM::Train - Kernel is null", LogLevel.PRODUCTION_LOG);
			return; 
		} 
		
		// store the predictors
		this.trainPredictors = trainPredictors;
		this.trainTargets = trainTargets;
		
		// the number of train instances
		int N = observableIndices.length; 
	
		// quit if no train instance, or if dimensions dont match
		if( N < 1 )
		{
			return;
		}
		
		alphas = new double[N];
		b = 0;
		
		// the linear coefficients
		if( A == null || A.getColumnDimension() != N+1 || A.getRowDimension() != N+1)
			A = new Array2DRowRealMatrix(N+1, N+1);
		
		//Logging.println( "LSSVM: Num Observed=" + N , LogLevel.DEBUGGING_LOG); 
		
		int i, j;  
		
		// create A =
		//  -----------------------
		//  | 0  |   1^T          |
		//  |----|----------------|
		//  | 1  |  K + gamma^-1 I |
		//  -----------------------
		
		// set zero the cell A(0,0)
		A.setEntry(0, 0, 0.0); 
				
		// set 1 in the first diagonal and the first row
		for (int index = 1; index < N+1; index++) 
		{ 
			A.setEntry(0, index, 1.0); 
			A.setEntry(index, 0, 1.0);  
		} 
		
		//double val = lambda / beta; 
		double val = lambda;
		
		// add the (K + 1/gamma I) in the range A(1,1) to A(M_i+1,M_i+1)
		for ( i = 1; i < N+1; i++) 
			for ( j = 1; j < N+1; j++) 
					A.setEntry(i, j, kernel.K(trainPredictors, 
											observableIndices[i-1], 
											observableIndices[j-1]) 
									+ ( i==j ? val : 0) );
		
		// create B =
		//  -----
		//  | 0 |
		//  |---|
		//  | Y |
		//  -----
		
		// create a column vector holding the row of i, B = X_i
		RealMatrix B = new Array2DRowRealMatrix(N+1, 1);
		
		B.setEntry(0, 0, 0.0 ); 
		// set the target values 
		for (i = 1; i < N+1; i++) 
			B.setEntry(i, 0, trainTargets[observableIndices[i-1]] ); 
		
		// solve the linear system of equation
		QRDecomposition decomposer = new QRDecomposition(A);
		//CholeskyDecomposition decomposer = new CholeskyDecomposition(A);
		//LUDecomposition decomposer = new LUDecomposition(A);
		
		RealMatrix solution = decomposer.getSolver().solve(B);
		
		// the first element of the solved parameter is the value of the beta
		b = solution.getEntry(0, 0); 
		
		// the rest are the values of the alphas parameters
		for (i = 0; i < N; i++) 
			alphas[i] = solution.getEntry(i+1, 0);
	}
	
	
	/*
	 * Classify the test set and return the RMSE
	 */
	public double PredictTestSet( Matrix testPredictors, double [] testTargets )
	{
		double mse = 0;
		
		int Nt = testPredictors.getDimRows();
		
		double [] estimatedTestTarget = new double[Nt];
		
		// compute the test targets from the predictors
		for(int i = 0; i < Nt; i++)
			estimatedTestTarget[i] = PredictInstance(testPredictors.getRow(i));
		
		for(int i = 0; i < Nt; i++)
		{
			double diff = estimatedTestTarget[i]-testTargets[i];
			mse += diff*diff;
		}
		
		
		return mse / Nt;
	}
	
	/*
	 * Classify the test set and return the RMSE
	 */
	public double PredictTrainSet( )
	{
		double mse = 0;
		
		int N = trainPredictors.getDimRows(); 
		
		double [] estimatedTrainTarget = new double[N];
		
		// compute the test targets from the predictors
		for(int i = 0; i < N; i++)
		{
			double val = 0;
			for(int j = 0; j < N; j++)
				val += alphas[j]*
						kernel.K(trainPredictors.getRow(i), trainPredictors.getRow(j));

			estimatedTrainTarget[i] = val + b;
		}
		
		for(int i = 0; i < N; i++)
		{
			double diff = estimatedTrainTarget[i]-trainTargets[i];
			mse += diff*diff;
		}
		
		return mse / N;
	}
	
	// predict an instance for sparse datasets, specify observed instances	
	public double PredictInstance(double [] instance, int [] observedIndices)
	{
		double val = 0;
		
		// assert if the number of alphas correspond to the observed instances
		if( alphas.length != observedIndices.length )
		{
			Logging.println("LSSVM::PredictTestInstance : size of alphas doesn't match size of observed indices", 
					LogLevel.ERROR_LOG);
			return val;
		}
		
		for(int i = 0; i < observedIndices.length; i++)
			val += alphas[i] * kernel.K(instance, trainPredictors.getRow(observedIndices[i]));
		val += b;
		
		return val;
	}
	
	// predict an instance for dense datasets
	public double PredictInstance(double [] testInstance)
	{
		double val = 0;
		
		for(int i = 0; i < trainPredictors.getDimRows(); i++) 
			val += alphas[i] * kernel.K(testInstance, trainPredictors.getRow(i));
		val += b;
		
		return val;
	}
}
