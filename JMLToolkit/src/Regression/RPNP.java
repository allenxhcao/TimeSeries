package Regression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.commons.math3.linear.RealMatrix;

import MatrixFactorization.BiasedMatrixFactorization;
import MatrixFactorization.UnbiasedMatrixFactorization;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import Classification.Kernel;
import Classification.Kernel.KernelType;
import DataStructures.Matrix;
import DataStructures.Tripple;
import DataStructures.Tripples;

// SRNP: Supervised Regression Through Nonlinear Projection

public class RPNP 
{
	// the regression models, one per each predictor
	LSSVM [] modelsPredictors;
	
	// the observed indices of the predictor cells
	int [][] observedPredictorsIndices;
	
	Matrix X;
	// the observation data
	
	// the low-rank data
	Matrix U;
	// the dimensionality of the latent data
	public int D;
	// number of instances and predictors
	int N, M;
	// the learn rate
	public double eta;
	// the kernel used for the 
	public Kernel kernel;
	// the gamma parameter of each regression
	public double lambdaU, lambdaV;
	// the maximum number of epocs
	public int maxEpocs;
	

	List<Integer> rowIds, colIds;

	
	// default constructor
	public RPNP()
	{
		kernel = new Kernel(KernelType.Polynomial);
		kernel.type = KernelType.Polynomial;
		kernel.degree = 2;
		
		maxEpocs = 1000;
	}
	
	
	// initialize the regression models 
	public void InitializeRegressionsModels( Tripples trainSet, Tripples testSet )
	{
		FixIndices(trainSet, testSet);
		
		/*
		UnbiasedMatrixFactorization umf = new UnbiasedMatrixFactorization(); 
		umf.lambdaU = lambdaU; 
		umf.lambdaV = lambdaV; 
		umf.eta = eta; 
		umf.maxEpocs = maxEpocs; 
		umf.K = D_i; 
		umf.trainData = trainSet; 
		umf.testData = testSet; 
		umf.N = N;
		umf.M = M_i;
		umf.Initialize();
		umf.Decompose();
		*/
		// store a reference to the data
		X = new Matrix(N, M); 
		
		for(Tripple trp : trainSet.cells) 
			X.set(trp.row, trp.col, trp.value);
		
		// initialize matrix U
		U = new Matrix(N, D);
		U.RandomlyInitializeCells(GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON); 
		
		// initialize the array of the regression models
		modelsPredictors = new LSSVM[M];
		observedPredictorsIndices = new int[M][];
		
		// initialize the predictor models and record the observed instances 
		for(int j = 0; j < M; j++)
		{
			modelsPredictors[j] = new LSSVM();
			modelsPredictors[j].kernel = kernel; 
			modelsPredictors[j].lambda = lambdaV;
			
			List<Integer> indices = new ArrayList<Integer>();
			
			for(int i = 0; i < N; i++)
				if( X.get(i,j) != GlobalValues.MISSING_VALUE )
					indices.add(i);
					
			observedPredictorsIndices[j] = new int[indices.size()];
			for( int i = 0; i < indices.size(); i++)
				observedPredictorsIndices[j][i] = indices.get(i);		
		}
		
	}
	
	/*
	 * The method that trains the supervised multi-regression
	 * nonlinear factorization
	 */
	public void Train( Tripples trainSet, Tripples testSet )
	{
		List<Double> historyMSE = new ArrayList<Double>(); 
		historyMSE.add( Double.MAX_VALUE );
		
		// assert some conditions
		if( trainSet == null )
		{
			Logging.println("SRNP::Train : Train data null", LogLevel.ERROR_LOG);
			return;
		}
		else if( kernel == null)
		{
			Logging.println("SRNP::Train : Kernel not initialized", LogLevel.ERROR_LOG);
			return;
		}
		
		// initialize the models
		InitializeRegressionsModels(trainSet, testSet);
		
		// iterate the gradient descent learning algorithm for a defined periods
		for(int epoc = 0; epoc < maxEpocs; epoc++)
		{
			// train the predictor models first
			for(int j = 0; j < M; j++)
			{
				modelsPredictors[j].Train(U, X.getCol(j), observedPredictorsIndices[j]);
				UpdateUPredictor(j);
			}
			
			if( Logging.currentLogLevel != LogLevel.PRODUCTION_LOG )
			{
				double trainMAE = Predict(trainSet); 
				double testMAE = Predict(testSet);
				Logging.println("Epoc=" + epoc + ", trainMAE=" + trainMAE + ", testMAE=" + testMAE, LogLevel.DEBUGGING_LOG);
			}
			
		}
		
		 
	}
	
	
	// update the low rank projection U, based on the solution of the target variable 
	
	public void UpdateUPredictor(int j)
	{
		int i=0,l=0;
		double grad = 0, kernelGrad = 0;
		
		for(int iIndex = 0; iIndex < observedPredictorsIndices[j].length; iIndex++)
		{
			for(int lIndex = 0; lIndex < observedPredictorsIndices[j].length; lIndex++)
			{
				i = observedPredictorsIndices[j][iIndex];
				l = observedPredictorsIndices[j][lIndex];
				
				for(int k = 0; k < D; k++)
				{
					kernelGrad = ComputeKernelGradient(i, l, i, k);
					
					if( kernelGrad != 0)
					{
						grad = modelsPredictors[j].alphas[iIndex] 
								* modelsPredictors[j].alphas[lIndex]
										* kernelGrad;
						
						U.set(i, k,  U.get(i,k) + eta*grad);
					}
					
					kernelGrad = ComputeKernelGradient(i, l, l, k);
					
					if( kernelGrad != 0)
					{
						grad = modelsPredictors[j].alphas[iIndex] 
								* modelsPredictors[j].alphas[lIndex]
										* kernelGrad;
						
						U.set(l, k,  U.get(l,k) + eta*grad);
					}
				}
			}
		}
	}
	
	// compute the kernel gradient { d K(i,l) / d U(r,k) } 
	public double ComputeKernelGradient(int i, int l, int r, int k)
	{
		double grad = 0;
		
		if( kernel.type == KernelType.Linear)
		{
			if( r == i )
				grad = U.get(l, k);
			else if(r == l)
				grad = U.get(i, k);	
		}
		else if( kernel.type == KernelType.Polynomial)
		{
			if( r == i || r == l )
			{
				grad = kernel.degree * 
						Utilities.StatisticalUtilities.Power(U.RowDotProduct(i, l)+1, kernel.degree-1); 
				
				if( r == i )
					grad *= U.get(l, k);
				else if(r == l)
					grad *= U.get(i, k);
			}			
		}
		
		return grad;
	}
	
	public double Predict(Tripples set)
	{
		int numObserved = set.cells.size();
		double squareError = 0;
		
		for( Tripple trp : set.cells )
		{
			double error = trp.value - Predict(trp.row, trp.col);
			squareError += error*error;
		}
		
		return Math.sqrt(squareError/numObserved);  
	}
	
	public double Predict(int i, int j)
	{
		return modelsPredictors[j].PredictInstance(U.getRow(i), observedPredictorsIndices[j]);
	}
	
	
	public void FixIndices(Tripples trainSet, Tripples testSet)
	{
		// fist merge elements in a set
		SortedSet<Integer> rowIdsSet = new TreeSet<Integer>();
		SortedSet<Integer> colIdsSet = new TreeSet<Integer>();
		
		// merge the row and col ids of the train and test sets
		rowIdsSet.addAll( trainSet.rowIds ); 
		rowIdsSet.addAll( testSet.rowIds );
		
		colIdsSet.addAll( trainSet.colIds );
		colIdsSet.addAll( testSet.colIds );
		
		// move the sets to plain arrays to 
		rowIds = new ArrayList<Integer>(); 
		rowIds.addAll(rowIdsSet); 
		
		colIds = new ArrayList<Integer>(); 
		colIds.addAll(colIdsSet);
		
		// convert the indices to the index of the ordered set
		for( Tripple trp : trainSet.cells )
		{
			trp.row = rowIds.indexOf( trp.row );
			trp.col = colIds.indexOf( trp.col );
		}
		for( Tripple trp : testSet.cells )
		{
			trp.row = rowIds.indexOf( trp.row );
			trp.col = colIds.indexOf( trp.col );
		}
		
		// randomly shuffle the cells
		Collections.shuffle( trainSet.cells ); 
		Collections.shuffle( testSet.cells );
		
    	N = rowIds.size();
    	M = colIds.size();
    	
    	Logging.println("N=" + N + ", M_i=" + M, LogLevel.DEBUGGING_LOG);
	}
	
}
