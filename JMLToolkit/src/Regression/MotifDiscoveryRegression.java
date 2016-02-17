package Regression;

import java.util.ArrayList;
import java.util.List;

import TimeSeries.BagOfPatterns;
import TimeSeries.BagOfPatterns.RepresentationType;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import Classification.Kernel;
import Classification.Kernel.KernelType;
import DataStructures.Matrix;

// variable selection in a greedy forward selection search
public class MotifDiscoveryRegression 
{
	// the test and training data
	public Matrix trainPredictors, testPredictors;
	public double [] trainLabels;
	public double [] testLabels;
	
	// the bag of pattern 
	public BagOfPatterns bop;
	
	// parameters of the bag of patterns
	public int n, alpha, bopPolyDegree, innerDimension;
	// parameters of the ls-svm
	public double C;
	public int svmPolyDegree;
	
	public MotifDiscoveryRegression()
	{

	}
	
	// select the best numVariables-many variables which improve classification rate 
	public int [] SelectVariables(int numVariables)
	{
		int [] bestSubset = null;
		
		int numPredictors = trainPredictors.getDimColumns(); 
		
		for(int k = 1; k <= numVariables; k++)
		{
			int [] candidateSubset = new int[k];
			// load the k-1 previous variables
			for(int p = 0; p < k-1; p++)
				candidateSubset[p] = bestSubset[p];
			
			int bestAdditionalPredictor = 0;
			double smallestError = Double.MAX_VALUE;
			
			for(int j = 0; j < numPredictors; j++)
			{
				// make sure it is not previously set
				boolean previouslySelected = false;
				for(int l = 0; l < k-1; l++)
				{
					if( candidateSubset[l] == j )
					{
						previouslySelected = true;
						break;
					}
				}
				
				if(previouslySelected)
					continue;
				
				// set the last candidate to j, in order to tests
				candidateSubset[k-1] = j;
				
				double error = EvaluateCandidate(candidateSubset);
				
				//Logging.println(candidateSubset, LogLevel.DEBUGGING_LOG);
				//Logging.println("Error=" + error, LogLevel.DEBUGGING_LOG); 				
				
				if(error < smallestError)
				{
					bestAdditionalPredictor = j;
					smallestError = error;
				}				
			}
			
			// set the best additional predictor
			candidateSubset[k-1] = bestAdditionalPredictor;
			// dump the best candidate so far
			bestSubset = candidateSubset.clone();
			
			
			Logging.println("----------------------------", LogLevel.DEBUGGING_LOG);
			Logging.println("Best so far:", LogLevel.DEBUGGING_LOG);
			Logging.println(bestSubset, LogLevel.DEBUGGING_LOG); 
			Logging.println("Best error=" + smallestError, LogLevel.DEBUGGING_LOG);
			
			//Logging.println(k + ", " + smallestError, LogLevel.DEBUGGING_LOG);
			
		}
		
		return bestSubset;
	}
	
	// evaluate a candidate subset using the ls-svm classifier and return the mse 
	public double EvaluateCandidate(int [] candidataSubset)
	{

		Matrix newPredictorsTrain = new Matrix( trainPredictors.getDimRows(), candidataSubset.length );
		
		// create a dataset for the train predictors containing only the predictors subset
		for( int i = 0; i < trainPredictors.getDimRows(); i++)
		{
			for( int jIndex = 0; jIndex < candidataSubset.length; jIndex++ )
			{
				int j = candidataSubset[jIndex];				
				newPredictorsTrain.set(i, jIndex, trainPredictors.get(i, j) ); 
			}
		}
		
		Matrix newPredictorsTest = new Matrix( testPredictors.getDimRows(), candidataSubset.length );
		
		// create a dataset for the test predictors containing only the predictors subset
		for( int i = 0; i < testPredictors.getDimRows(); i++)
		{
			for( int jIndex = 0; jIndex < candidataSubset.length; jIndex++ )
			{
				int j = candidataSubset[jIndex];				
				newPredictorsTest.set(i, jIndex, testPredictors.get(i, j) ); 
			}
		}
		
		// test the parameters
		LSSVM lssvm = new LSSVM();
		lssvm.kernel = new Kernel();
		lssvm.kernel.type = KernelType.Polynomial;
	
		// set lambda 1 and degree 3
		lssvm.lambda = C;
		lssvm.kernel.degree = svmPolyDegree;
		
		// train the lssvm
		lssvm.Train(newPredictorsTrain, trainLabels);
		// predict the test values and record the MSE error
		double error = lssvm.PredictTestSet(newPredictorsTest, testLabels); 
		
		return error;
	}
	
	// create a bag of patterns histogram from the data
	public void CreateHistogram(Matrix predTr, Matrix predTe, double [] labTr, double [] labTe)
	{
		// merge the predictors
		Matrix X = new Matrix(predTr);
        X.AppendMatrix(predTe);
        
        // create a bag of patterns representation
        bop = new BagOfPatterns();
        bop.slidingWindowSize = n;
        bop.representationType = RepresentationType.Polynomial;
        //bop.representationType = RepresentationType.SAX;
        bop.alphabetSize = alpha;
        bop.polyDegree = bopPolyDegree;
        bop.innerDimension = innerDimension; 
        
        
        // create the histogram
        Matrix H = bop.CreateWordFrequenciesMatrix(X);
        
        System.out.println("25-th word is " + bop.dictionary.get(25) );
        
        H.SaveToFile("C:\\Users\\josif\\Desktop\\hist.csv"); 
        
        // set the new train and test predictors
        trainPredictors = new Matrix( predTr.getDimRows(), H.getDimColumns());
        for(int r = 0; r < predTr.getDimRows(); r++)
        	trainPredictors.SetRow(r, H.getRow(r));
        
        testPredictors = new Matrix( predTe.getDimRows(), H.getDimColumns());
        for(int r = 0; r < predTe.getDimRows(); r++)
        	testPredictors.SetRow(r, H.getRow(r + predTr.getDimRows() ));
        
        // set the labels
        trainLabels = labTr;
        testLabels = labTe;
        
        // regress the full histogram
        LSSVM lssvm = new LSSVM();
		lssvm.kernel = new Kernel();
		lssvm.kernel.type = KernelType.Polynomial;
		lssvm.lambda = C;
		lssvm.kernel.degree = svmPolyDegree;
		
		lssvm.Train(trainPredictors, trainLabels);
		
		double error = lssvm.PredictTestSet(testPredictors, testLabels);
		
		Logging.println("Error full histogram: " + error, LogLevel.ERROR_LOG); 
        
	}
	
	
}

