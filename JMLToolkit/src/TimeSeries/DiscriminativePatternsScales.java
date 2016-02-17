package TimeSeries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import TimeSeries.BagOfPatterns.RepresentationType;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;
import Classification.NearestNeighbour;
import Classification.WekaClassifierInterface;
import Clustering.KMeans;
import DataStructures.Coordinate;
import DataStructures.DataSet;
import DataStructures.Matrix;


public class DiscriminativePatternsScales 
{	 
	// number of training and testing instances
	public int ITrain, ITest;
	// length of a time-series 
	public int Q;
	// length of segments at different scales
	public int initL;
	public int L[];
	
	// magnitudes of the segment scales 
	public int M;
	
	// number of latent patterns
	public int K;
	
	// number local segments
	int J[];
	
	double S[][][][];
	
	// latent patterns
	double P[][][];
	
	// degrees of membership
	double D[][][][];
	
	// classification weights
	double W[][];
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y;
		
	// the number of iterations
	public int maxIter;
	// the learning rate
	public double eta; 
	
	// the regularization parameters
	public double lambdaD, lambdaP, lambdaW;
	
	// the impact switch
	public double beta;
	
	public double cR, cA;
	
	// the delta increment between segments of the sliding window
	public int deltaT;

	boolean isUnsupervised = false;
	
	Random rand = new Random();
	
	// constructor
	public DiscriminativePatternsScales()
	{
		deltaT = 1;
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", initL="+initL + ", maxMagnitudeScale="+M, LogLevel.DEBUGGING_LOG);
		Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lambdaD="+ lambdaD + ", lambdaF="+ lambdaP + ", lamdaW="+lambdaW + ", beta="+beta, LogLevel.DEBUGGING_LOG);
		
		// avoid zero/negative sliding window increments, 
		// or increments greater than the window size
		if( deltaT < 1) deltaT = 1; 
		else if(deltaT > initL ) deltaT = initL; 
		Logging.println("deltaT="+ deltaT, LogLevel.DEBUGGING_LOG); 
		

		SegmentTimeSeriesDataset();		
		InitializePatternsProbabilityDistance(); 
		InitializeHardMembershipsToClosestPattern(); 
		
		cR = beta;  
		cA = (1.0-beta);  
		
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		for(int i = 0; i < ITrain+ITest; i++)
			if(Y.get(i) != 1.0)
				Y.set(i, 0, 0.0);
		
		InitializeWeights();
	
		
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	
	// initialize the patterns from random segments
	public void InitializePatternsProbabilityDistance()
	{
		P = new double[M][K][];
		
		for(int m=0; m < M; m++)
		{
			double [][] segmentsM = new double[(ITrain + ITest)*J[m]][L[m]];
			
			for(int i= 0; i < ITrain + ITest; i++) 
				for(int j= 0; j < J[m]; j++) 
					for(int l = 0; l < L[m]; l++)
						segmentsM[i*J[m] + j][l] = S[i][m][j][l];
			
			KMeans kmeans = new KMeans();
			P[m] = kmeans.InitializeKMeansPP(segmentsM, K);
			
			if( P[m] == null)
				System.out.println("P not set");
		}
	}
	
	
	// initialize the degree of membership to the closest pattern
	public void InitializeHardMembershipsToClosestPattern()
	{
		// initialize the degree of membership to 1 for the closest initial pattern
		// and 0 for all the others
		D = new double[ITrain+ITest][M][][];
				
		for(int i = 0; i < ITrain+ITest; i++)
		{
			for(int m = 0; m < M; m++)
			{
				D[i][m] = new double [J[m]][K]; 
				
				for(int j = 0; j < J[m]; j++)
				{
					// compute the distance between the i,j-th segment and the k-th pattern
					double minDist = Double.MAX_VALUE;
					int closestPattern = 0;
					
					for(int k = 0; k < K; k++)
					{					
						double dist = 0;
						for(int l = 0; l < L[m]; l++)
						{
							double err = S[i][m][j][l] - P[m][k][l];
							dist += err*err;
						}
						
						if(dist < minDist)
						{
							minDist = dist;
							closestPattern = k;
						}
					}	
					
					for(int k = 0; k < K; k++)
						if( k == closestPattern)
							D[i][m][j][k] = 1.0;
						else
							D[i][m][j][k] = 0.0;
				}
			}
		}
	}
	
	// initialize the classification weights W
	public void InitializeWeights()
	{
		W = new double[M][K];
		
		for(int m = 0; m < M; m++) 
			for(int k = 0; k < K; k++) 
				W[m][k] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON; 
	}
	
	// partition the time series into segments
	public void SegmentTimeSeriesDataset() 
	{
		// initialize the segments, the last two indices will be initialized via dynamic sizes
		S = new double[ITrain+ITest][M][][];
		// initialize the vector of the numbers of vectors segment at each scale
		J = new int[M];
		// initialize the length of sliding window at each scale
		L = new int[M];
		
		for(int i = 0; i < ITrain+ITest; i++)
		{
			for(int m = 0; m < M; m++)
			{
				// set the segment size and the number of segments at scale m
				//L[m] = StatisticalUtilities.PowerInt(2, m)*initL; 
				L[m] = (m+1)*initL; 
				J[m] = (Q - L[m]) / deltaT;
				// initialize the segments storage at scale m
				//System.out.println( "J[m]="+J[m]+", L[m]="+L[m] );
				
				S[i][m] = new double[J[m]][L[m]];
								
				for(int j = 0; j < J[m]; j++)
				{
					for(int l = 0; l < L[m]; l++)
						S[i][m][j][l] = T.get(i, (j*deltaT) + l); 
					
					// normalize the segment 
					double [] normalizedSegment = StatisticalUtilities.Normalize(S[i][m][j]);
					for(int l = 0; l < L[m]; l++)
						S[i][m][j][l] = normalizedSegment[l];
				}
			}
		}
		
		Logging.println("Partion to Normalized Segments Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// compute the MCR on the test set
	private double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			double label_i = Sigmoid.Calculate(Predict(i));
			
			if( (Y.get(i) == 1 && label_i < 0.5) || (Y.get(i) == 0 && label_i >= 0.5) )
						numErrors++;
		}
		
		return (double)numErrors/(double)ITrain;
	}
	
	// compute the MCR on the test set
	private double GetMCRTestSet() 
	{
		int numErrors = 0;
		
		for(int i = ITrain; i < ITrain+ITest; i++)
		{
			double label_i = Sigmoid.Calculate(Predict(i));
			
			if( (Y.get(i) == 1 && label_i < 0.5) || (Y.get(i) == 0 && label_i >= 0.5) )
						numErrors++;
		}
		
		return (double)numErrors/(double)ITest;
	}
	
	// predict the label value vartheta_i
	public double Predict(int i)
	{
		double Y_hat_i = 0;

		double F_imk = 0;
		
		for(int m = 0; m < M; m++)
		{
			for(int k = 0; k < K; k++)
			{
				F_imk = 0;
				for(int j = 0; j < J[m]; j++)
					F_imk += D[i][m][j][k];
				
				Y_hat_i += F_imk * W[m][k];
			}
		}
		return Y_hat_i;
	}
	
	
	
	// reconstruct the point l of the j-th segment of the i-th time series
	public double Reconstruct(int i, int m, int j, int l)
	{
		double S_imjl = 0;
		
		// apply a convolution of k-many patterns and their degrees of membership
		// use the sum of the l-th points from each pattern P
		for(int k = 0; k < K; k++)
			S_imjl += D[i][m][j][k] * P[m][k][l];
		
		return S_imjl;
	}
	
	// measure the reconstruction loss
	public double MeasureRecontructionLoss()
	{
		double reconstructionLoss = 0, e_imjl = 0;
		
		// iterate through all the time series
		for(int i = 0; i < ITrain+ITest; i++)
		{
			for(int m = 0; m < M; m++)
			{
				for(int j = 0; j < J[m]; j++)
				{
					for(int l = 0; l < L[m]; l++)
					{
						e_imjl = S[i][m][j][l] - Reconstruct(i, m, j, l);
						reconstructionLoss += cR*e_imjl*e_imjl; 					
					} 
				}
			}
		}
		
		return reconstructionLoss;
	}
	
	// compute the accuracy loss of instance i according to the 
	// smooth hinge loss 
	public double AccuracyLoss(int i)
	{
		double Y_hat_i = Predict(i);
		double sig_y_i = Sigmoid.Calculate(Y_hat_i);
		
		return cA*(- Y.get(i)*Math.log( sig_y_i ) - (1-Y.get(i))*Math.log(1-sig_y_i)); 
	}
	
	// compute the accuracy loss of the train set
	public double AccuracyLossTrainSet()
	{
		double accuracyLoss = 0;
		
		for(int i = 0; i < ITrain; i++)
			accuracyLoss += AccuracyLoss(i);
		
		return accuracyLoss;
	}
	// compute the accuracy loss of the train set
	public double AccuracyLossTestSet()
	{
		double accuracyLoss = 0;
		
		for(int i = ITrain; i < ITrain+ITest; i++)
			accuracyLoss += AccuracyLoss(i);
		
		return accuracyLoss;
	}
	
	
	public void LearnLR()
	{
		double e_imjl = 0;
		
		double regDConst = 0, regPConst = 0; 
		
		for(int i = 0; i < ITrain+ITest; i++) 
		{
			for(int m = 0; m < M; m++)
			{
				regDConst = lambdaD/L[m];
				regPConst = (2*lambdaP)/((ITrain+ITest)*J[m]);
				
				for(int j = 0; j < J[m]; j++) 
				{
					for(int l = 0; l < L[m]; l++) 
					{
						e_imjl = S[i][m][j][l] - Reconstruct(i, m, j, l); 
						
						for(int k = 0; k < K; k++) 
						{
							D[i][m][j][k] -=  eta*(-2*cR*e_imjl*P[m][k][l] + regDConst*D[i][m][j][k]); 
							P[m][k][l] -= eta*(-2*cR*e_imjl*D[i][m][j][k] + regPConst*P[m][k][l]); 
						}					
					}
					
				}
			}
		}
	}
	
	
	public void LearnLA()
	{
		for(int i = 0; i < ITrain; i++)
		{
			LearnLA(i); 
		}
		
	}
	
	public void LearnLA(int i)
	{
		double e_i = 0, F_imk = 0;
		double regWConst = (2*lambdaW)/ITrain;
		
		e_i = Y.get(i) - Sigmoid.Calculate(Predict(i));
		
		for(int m = 0; m < M; m++)
		{
			for(int k = 0; k < K; k++)
			{
				F_imk = 0;
			
				for(int j = 0; j < J[m]; j++)
				{
					D[i][m][j][k] -= eta*(cA*-e_i*W[m][k] + lambdaD*D[i][m][j][k]); 						
					F_imk += D[i][m][j][k];
				}
					
				W[m][k] -= eta*(cA*-e_i*F_imk + regWConst*W[m][k]); 						
			}
		}
	
	}
		
	public void LearnLAWeights()
	{
		double e_i = 0, F_imk = 0;
		double regWConst = (2*lambdaW)/ITrain;
		
		for(int iter = 0; iter < maxIter; iter++)
		{
			for(int i = 0; i < ITrain; i++)
			{ 
				e_i = Y.get(i) - Sigmoid.Calculate(Predict(i));
				
				for(int m = 0; m < M; m++)
				{
					for(int k = 0; k < K; k++)
					{
						F_imk = 0;
						for(int j = 0; j < J[m]; j++)
							F_imk += D[i][m][j][k];
							
						W[m][k] -= eta*(cA*-e_i*F_imk + regWConst*W[m][k]); 						
					}
				}
			}
		}
		
	}
	
	// optimize the objective function
	public double Learn()
	{
		// initialize the data structures
		Initialize();
		
		List<Double> lossHistory = new ArrayList<Double>();
		lossHistory.add(Double.MIN_VALUE);
		
		// apply the stochastic gradient descent in a series of iterations
		for(int iter = 0; iter < maxIter; iter++)
		{
			// measure the loss
			double LR = MeasureRecontructionLoss();
			double mcrTrain = GetMCRTrainSet();
			double mcrTest = GetMCRTestSet();
			double LATrain = AccuracyLossTrainSet();
			double LATest = AccuracyLossTestSet();
			
			Logging.println("It=" + iter + ", LR="+LR+", LATrain="+ LATrain + ", LATest="+ LATest  +
							", MCRTrain=" +mcrTrain + ", MCRTest=" +mcrTest, LogLevel.DEBUGGING_LOG);
			
			// learn the latent matrices
			LearnLR();
			
			if(!isUnsupervised)
				LearnLA();
			
			lossHistory.add(LR+LATrain);
			
			if( lossHistory.size() > 10 )
			{
				//if( LR+LATrain > lossHistory.get( lossHistory.size() - 9  )  )
					//break;
			}
		}
		
		if(isUnsupervised)
			LearnLAWeights(); 
		
		return GetMCRTestSet(); 
	}
	
	

}
