package Classification;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import Regression.LSSVM;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import Classification.Kernel.KernelType;
import DataStructures.DataSet;
import DataStructures.Matrix;

public class VariableSubsetSelection 
{
	Matrix predictors;
	
	int numTrain;
	int numTest;
	
	Matrix target;
	Matrix trainTarget;
	Matrix testTarget;
	
	public double svmC, svmDegree;
	
	Random rand = new Random();
	
	public VariableSubsetSelection()
	{
		svmC = 1.0;
		svmDegree = 3.0;
	}
	
	
	public void Init(Matrix preds, Matrix labs, int nTrain, int nTest)
	{
		predictors = preds;
		target = labs;
		
		numTrain = nTrain;
		numTest = nTest;
		
		trainTarget = new Matrix(numTrain, 1);
		for( int i = 0; i < numTrain; i++)
			trainTarget.set(i, 0, labs.get(i));
		
		testTarget = new Matrix(numTest, 1);
		for( int i = numTrain; i < numTrain+numTest; i++)
			testTarget.set(i-numTrain, 0, labs.get(i));  
	}
	
	public int [] SelectVariables( )
	{
		int [] bestSubsequentSubset = null;
		int [] bestSubset = null;
		double bestMCR = 1.0;
		
		int numVariables = predictors.getDimColumns();  
		
		for(int k = 1; k <= numVariables; k++)
		{
			int [] candidateSubset = new int[k];
			// load the k-1 previous variables
			for(int p = 0; p < k-1; p++)
				candidateSubset[p] = bestSubsequentSubset[p];
			
			int bestSubsequentPredictor = 0;
			double smallestSubsequentMcr = Double.MAX_VALUE;
			
			for(int j = 0; j < numVariables; j++)
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
				
				double mcr = EvaluateCandidate(candidateSubset);
				//double mcr = CrossEvaluateCandidate(candidateSubset, 10);
				
				//Logging.println(candidateSubset, LogLevel.DEBUGGING_LOG);
				//Logging.println("Error=" + mcr, LogLevel.DEBUGGING_LOG); 				
				
				if(mcr < smallestSubsequentMcr)
				{
					bestSubsequentPredictor = j;
					smallestSubsequentMcr = mcr;
				}				
			}
			
			// set the best additional predictor
			candidateSubset[k-1] = bestSubsequentPredictor;
			// dump the best candidate so far
			bestSubsequentSubset = candidateSubset.clone();
			
			// check if the subsequent subset is an improvement
			if(smallestSubsequentMcr < bestMCR) 
			{
				bestMCR = smallestSubsequentMcr;
				bestSubset = bestSubsequentSubset.clone();
			}
			else // otherwise check if it is time to stop
			{
				if( bestSubset != null )
				{
					// quit if 5 subsequent iterations without improvement
					if( bestSubsequentSubset.length - bestSubset.length > 10 )
						break;
				}
			}
			
			Logging.println("----------------------------", LogLevel.DEBUGGING_LOG);
			Logging.println("Current Subset:", LogLevel.DEBUGGING_LOG);
			Logging.println(bestSubsequentSubset, LogLevel.DEBUGGING_LOG); 
			Logging.println("Current MCR=" + smallestSubsequentMcr, LogLevel.DEBUGGING_LOG);
			
			//Logging.println(k + ", " + smallestError, LogLevel.DEBUGGING_LOG);
			
		}
		
		Logging.println("----------------------------", LogLevel.DEBUGGING_LOG);
		Logging.println("Best Subset:", LogLevel.DEBUGGING_LOG);
		Logging.println(bestSubset, LogLevel.DEBUGGING_LOG); 
		Logging.println("Best MCR=" + bestMCR, LogLevel.DEBUGGING_LOG);
		Logging.println("----------------------------", LogLevel.DEBUGGING_LOG);
		
		
		return bestSubset; 
	}
	
	// evaluate a candidate subset using the classifier and return the mcr 
	public double EvaluateCandidate(int [] candidateSubset)
	{
		// create a dataset for the train predictors containing only the predictors subset
		Matrix newPredictorsTrain = new Matrix( numTrain, candidateSubset.length );
		for( int i = 0; i < numTrain; i++)
			for( int jIndex = 0; jIndex < candidateSubset.length; jIndex++ )
			{
				int j = candidateSubset[jIndex];				
				newPredictorsTrain.set(i, jIndex, predictors.get(i, j) ); 
			}
		
		// create a dataset for the test predictors containing only the predictors subset
		Matrix newPredictorsTest = new Matrix(numTest, candidateSubset.length );
		for( int i = numTrain; i < numTrain+numTest; i++)
			for( int jIndex = 0; jIndex < candidateSubset.length; jIndex++ )
			{
				int j = candidateSubset[jIndex];				
				newPredictorsTest.set(i-numTrain, jIndex, predictors.get(i, j) ); 
			}
		
		// create datasets
		DataSet trainSet = new DataSet();
	    trainSet.LoadMatrixes(newPredictorsTrain, trainTarget);
	    DataSet testSet = new DataSet();
	    testSet.LoadMatrixes(newPredictorsTest, testTarget); 
	    
	    NearestNeighbour nn = new NearestNeighbour("euclidean");
	    
		return nn.Classify(trainSet, testSet);
	}
	
	
	// evaluate a candidate subset using the classifier and return the mcr 
		public double CrossEvaluateCandidate(int [] candidateSubset, int numCrossValidations)
		{
			double mcrSum = 0.0;
			
			int numTestCV = (numTrain+numTest)/numCrossValidations;
			int numTrainCV = numTrain+numTest - numTestCV; 
			
			for(int fold = 0; fold < numCrossValidations; fold++)
			{
				// pick randomply 
				Matrix newPredictorsTrain = new Matrix( numTrainCV, candidateSubset.length );
				Matrix newTargetTrain = new Matrix( numTrainCV, 1 );
				for( int iIdx = 0; iIdx < numTrainCV; iIdx++)
				{
					int i = rand.nextInt(numTrain+numTest);
							
					for( int jIndex = 0; jIndex < candidateSubset.length; jIndex++ )
					{
						int j = candidateSubset[jIndex];				
						newPredictorsTrain.set(iIdx, jIndex, predictors.get(i, j) ); 
					}
					
					newTargetTrain.set(iIdx, 0, target.get(i));
				}
				
				// create a dataset for the test predictors containing only the predictors subset
				Matrix newPredictorsTest = new Matrix(numTestCV, candidateSubset.length );
				Matrix newTargetTest = new Matrix( numTestCV, 1 );
				for( int iIdx = 0; iIdx < numTestCV; iIdx++)
				{
					int i = rand.nextInt(numTrain+numTest);
					
					for( int jIndex = 0; jIndex < candidateSubset.length; jIndex++ )
					{
						int j = candidateSubset[jIndex];				
						newPredictorsTest.set(iIdx, jIndex, predictors.get(i, j) ); 
					}
					
					newTargetTest.set(iIdx, 0, target.get(i));
				}
				
				// create datasets
				DataSet trainSet = new DataSet();
			    trainSet.LoadMatrixes(newPredictorsTrain, newTargetTrain);
			    DataSet testSet = new DataSet();
			    testSet.LoadMatrixes(newPredictorsTest, newTargetTest); 
			    
			    // compute accuracy
			    NearestNeighbour nn = new NearestNeighbour("euclidean");
				mcrSum += nn.Classify(trainSet, testSet);
				
			    /*
				Instances trainWeka = trainSet.ToWekaInstances();
				Instances testWeka = testSet.ToWekaInstances();

				SMO svm = WekaClassifierInterface.getPolySvmClassifier(svmC, svmDegree);
				Evaluation eval = null;
				
				try
				{
					svm.buildClassifier(trainWeka);
					eval = new Evaluation(trainWeka);
					eval.evaluateModel(svm, testWeka);
				}
				catch(Exception exc) 
				{
					Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
				}
								
				mcrSum += eval.errorRate();
				*/
			}
			
			return mcrSum/numCrossValidations;
		}
	
}
