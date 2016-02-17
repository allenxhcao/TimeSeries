package TimeSeries;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Classification.NearestNeighbour;
import Classification.WekaClassifierInterface;
import DataStructures.DataSet;
import DataStructures.Matrix;

public class InvariantConvolution 
{
	public int minWindowSize;
	public int maxWindowSize;
	public int increment;
	
	// number of dimensions
	public int K;
	// the delta T increment
	public int deltaT;
	
	public  int maxScale;
	
	// the data predictors and labels
	public Matrix T;
	public Matrix Y;
	// size of the data, train and test instances
	public int NTrain, NTest;
	// dimensionality of the data
	public int M;
	
	// the maximum number of iterations
	public int maxIter;
	
	// the regularization parameter of the pattern
	public double lambdaP;
	
	// the complexity and degree of polynomial kernel
	public double svmC, svmDegree;
	
	public InvariantConvolution()
	{
		svmC = 1.0;
		svmDegree = 2.0;
	}
	
	// learn and return the test error and the leave one out train error
	public double [] Learn()
	{
		//int numIncrements = ((M_i-minWindowSize)/increment);
		//int maxScaleIdx = 4;
		Logging.println("maxScaleIdx="+maxScale+", FeatureSize="+K*maxScale, LogLevel.DEBUGGING_LOG); 
		
		// initialize a convolutional factorization
		Matrix F = new Matrix(NTrain+NTest, K*maxScale);
		
		for(int scaleIdx=1; scaleIdx <= maxScale ; scaleIdx++) 
		{
			ConvolutionLocalPatterns clp = new ConvolutionLocalPatterns();
			//ConvolutionLocalPatternsFoldIn clp = new ConvolutionLocalPatternsFoldIn();
			clp.T = T;
			clp.Y = Y;
			clp.K = K;
			clp.deltaT = deltaT;
			clp.NTrain = NTrain;
			clp.NTest = NTest;
			clp.M = M; 
			clp.maxIter = maxIter; 
			clp.L = minWindowSize*scaleIdx;
			clp.lambdaP = lambdaP; 
			
			Logging.println("Scale="+scaleIdx+ ", L="+ clp.L, LogLevel.DEBUGGING_LOG); 
			double mcr = clp.Learn();
			Logging.println("Scale="+scaleIdx +", MCR="+mcr, LogLevel.DEBUGGING_LOG); 
			
			double f_ik = 0;
			for(int i=0; i < NTrain+NTest; i++)
			{
				for(int k = 0; k < K; k++)
				{
					f_ik = 0;
					for(int j = 0; j < clp.NSegments; j++)
						f_ik += clp.D[i][j][k];
					
					// set it to the cummulative features
					int kp = k + (scaleIdx-1)*K; 
					
					F.set(i, kp, f_ik); 
				} 
			} 
			
			clp = null;
			System.gc();
		} 
		 
		return ClassifyHistogram(F);
		//return ClassifyHistogramNearestNeighbor(F); 
	}
	
	public double [] LearnFoldIn()
	{
		Logging.println("maxScaleIdx="+maxScale+", FeatureSize="+K*maxScale, LogLevel.DEBUGGING_LOG); 
		
		// initialize a convolutional factorization
		Matrix F = new Matrix(NTrain+NTest, K*maxScale);
		
		for(int scaleIdx=1; scaleIdx <= maxScale ; scaleIdx++) 
		{
			ConvolutionLocalPatternsFoldIn clp = new ConvolutionLocalPatternsFoldIn();
			clp.T = T;
			clp.Y = Y;
			clp.K = K;
			clp.deltaT = deltaT;
			clp.NTrain = NTrain;
			clp.NTest = NTest;
			clp.M = M; 
			clp.maxIter = maxIter; 
			clp.L = minWindowSize*scaleIdx;  
			clp.lambdaP = lambdaP; 
			
			Logging.println("Scale="+scaleIdx+ ", L="+ clp.L, LogLevel.DEBUGGING_LOG); 
			double mcr = clp.Learn();
			Logging.println("Scale="+scaleIdx +", MCR="+mcr, LogLevel.DEBUGGING_LOG); 
			
			double f_ik = 0;
			for(int i=0; i < NTrain; i++)
			{
				for(int k = 0; k < K; k++)
				{
					f_ik = 0;
					for(int j = 0; j < clp.NSegments; j++)
						f_ik += clp.DTrain[i][j][k];
					
					// set it to the cummulative features
					int kp = k + (scaleIdx-1)*K; 
					
					F.set(i, kp, f_ik); 
				} 
			} 
			
			for(int i=0; i < NTest; i++)
			{
				for(int k = 0; k < K; k++)
				{
					f_ik = 0;
					for(int j = 0; j < clp.NSegments; j++)
						f_ik += clp.DTest[i][j][k];
					
					// set it to the cummulative features
					int kp = k + (scaleIdx-1)*K; 
					
					F.set(i+NTrain, kp, f_ik); 
				} 
			} 
			
			clp = null;
			System.gc();
		} 
		 
		return ClassifyHistogram(F);
	}
	
	
	public double [] ClassifyHistogram(Matrix F) 
	{
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y, 0, NTrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(F, Y, NTrain, NTrain+NTest); 
	    
	    // print the training instances
	    //Logging.print(F.cells, System.out, LogLevel.DEBUGGING_LOG ); 
	     
	    
	    
	    
	    Instances trainWeka = trainSetHist.ToWekaInstances();
		Instances testWeka = testSetHist.ToWekaInstances();
		
		SMO svm = WekaClassifierInterface.getPolySvmClassifier(svmC, svmDegree);
		
		// first evaluate the leave one out cross-validation 
		Evaluation eval = null;
		Random rand = new Random();
		
		try
		{
			svm.buildClassifier(trainWeka);
			eval = new Evaluation(trainWeka);
			
			// for leave one out cross validation, set the number of folds to the number of instances
			eval.crossValidateModel(svm, trainWeka, NTrain, rand ); 
			
		}
		catch(Exception exc) 
		{
			Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
		}
		
		double trainError = eval.errorRate();
		
		// measure the test error
		eval = null;
		
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
						
		double testError = eval.errorRate();
		
		// return the test and train errors
		return new double[]{trainError, testError};
	}
	
	public double [] ClassifyHistogramNearestNeighbor(Matrix F)
	{
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y, 0, NTrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(F, Y, NTrain, NTrain+NTest); 
	    
	    Instances trainWeka = trainSetHist.ToWekaInstances();
		Instances testWeka = testSetHist.ToWekaInstances();

		IBk nn = new IBk(1);
		
		// first evaluate the leave one out cross-validation 
		Evaluation eval = null;
		Random rand = new Random();
		
		try
		{
			eval = new Evaluation(trainWeka);
			eval.crossValidateModel(nn, trainWeka, NTrain, rand ); 
		}
		catch(Exception exc) 
		{
			Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
		}
		
		double trainError = eval.errorRate();
		
		// measure the test error
		eval = null;
		
		try
		{
			eval = new Evaluation(trainWeka);
			eval.evaluateModel(nn, testWeka);
		}
		catch(Exception exc) 
		{
			Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
		}
						
		double testError = eval.errorRate();
		
		// return the test and train errors
		return new double[]{trainError, testError};
	}
}
