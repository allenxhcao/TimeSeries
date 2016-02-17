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


public class DiscriminativePatterns 
{
	// number of training and testing instances
	public int NTrain, NTest;
	// length of a time-series
	public int M;
	// length of a segment
	public int L;
	// number of latent patterns
	public int K;
	
	// local segments
	int NSegments;
	double S[][][];
	
	// latent patterns
	double P[][];
	
	// degrees of membership
	double D[][][];
	
	// classification weights
	double W[];
	
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

	
	Random rand = new Random();
	
	// constructor
	public DiscriminativePatterns()
	{
		deltaT = 1;
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		Logging.println("NTrain="+NTrain + ", NTest="+NTest + ", M_i="+M, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L="+L, LogLevel.DEBUGGING_LOG);
		Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lambdaD="+ lambdaD + ", lambdaF="+ lambdaP + ", lamdaW="+lambdaW + ", beta="+beta, LogLevel.DEBUGGING_LOG);
		
		// avoid zero/negative sliding window increments, 
		// or increments greater than the window size
		if( deltaT < 1) deltaT = 1;
		else if(deltaT > L ) deltaT = L;
		
		NSegments = (M-L)/deltaT; 
		
		Logging.println("deltaT="+ deltaT + ", NSegments="+ NSegments, LogLevel.DEBUGGING_LOG);
		
		cR = beta/(NSegments*L);
		cA = (1-beta);
		
		SegmentTimeSeriesDataset();	
		InitializePatternsProbabilityDistance();
		InitializeHardMembershipsToClosestPattern();
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		for(int i = 0; i < NTrain+NTest; i++)
			if(Y.get(i) != 1.0)
				Y.set(i, 0, 0.0);
		
		InitializeWeights();
	
		
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
	}
	
		// initialize the patterns from random segments

	// initialize the patterns from random segments
	public void InitializePatternsProbabilityDistance()
	{
	
		double [][] segments = new double[(NTrain + NTest)*NSegments][L];
		for(int i= 0; i < NTrain + NTest; i++) 
			for(int j= 0; j < NSegments; j++) 
				for(int l = 0; l < L; l++)
					segments[i*NSegments + j][l] = S[i][j][l];
		
		KMeans kmeans = new KMeans();
		P = kmeans.InitializeKMeansPP(segments, K);
		
		if( P == null)
			System.out.println("P not set");
	}
	
	// initialize the degree of membership to the closest pattern
	public void InitializeHardMembershipsToClosestPattern()
	{
		// initialize the degree of membership to 1 for the closest initial pattern
		// and 0 for all the others
		D = new double[NTrain+NTest][NSegments][K];
				
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
			{
				// compute the distance between the i,j-th segment and the k-th pattern
				double minDist = Double.MAX_VALUE;
				int closestPattern = 0;
				
				for(int k = 0; k < K; k++)
				{					
					double dist = 0;
					for(int l = 0; l < L; l++)
					{
						double err = S[i][j][l] - P[k][l];
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
						D[i][j][k] = 1.0;
					else
						D[i][j][k] = 0.0;
			}
	}
	
	// initialize the classification weights W
	public void InitializeWeights()
	{
		W = new double[K];
		for(int k = 0; k < K; k++) 
			W[k] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON; 
		
		double LR = MeasureRecontructionLoss();
		double mcrTrain = GetMCRTestSet();
		double mcrTest = GetMCRTestSet();
		double LATrain = AccuracyLossTrainSet();
		double LATest = AccuracyLossTrainSet();
		
		Logging.println("LR="+LR+", LATrain="+ LATrain + ", LATest="+ LATest  +
						", MCRTrain=" +mcrTrain + ", MCRTest=" +mcrTest, LogLevel.DEBUGGING_LOG);
	
		
	}
	
	// partition the time series into segments
	public void SegmentTimeSeriesDataset()
	{
		S = new double[NTrain+NTest][NSegments][L]; 
		
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int l = 0; l < L; l++)
					S[i][j][l] = T.get(i, (j*deltaT) + l); 
				
				// normalize the segment 
				double [] normalizedSegment = StatisticalUtilities.Normalize(S[i][j]);
				for(int l = 0; l < L; l++)
					S[i][j][l] = normalizedSegment[l];
			}
		}
		
		Logging.println("Partion to Normalized Segments Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// compute the MCR on the test set
	private double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < NTrain; i++)
		{
			double label_i = Sigmoid.Calculate(Predict(i));
			
			if( (Y.get(i) == 1 && label_i < 0.5) || (Y.get(i) == 0 && label_i >= 0.5) )
						numErrors++;
		}
		
		return (double)numErrors/(double)NTrain;
	}
	
	// compute the MCR on the test set
	private double GetMCRTestSet() 
	{
		int numErrors = 0;
		
		for(int i = NTrain; i < NTrain+NTest; i++)
		{
			double label_i = Sigmoid.Calculate(Predict(i));
			
			if( (Y.get(i) == 1 && label_i < 0.5) || (Y.get(i) == 0 && label_i >= 0.5) )
						numErrors++;
		}
		
		return (double)numErrors/(double)NTest;
	}
	
	// predict the label value vartheta_i
	public double Predict(int i)
	{
		double Y_hat_i = 0;

		double F_ik = 0;
		for(int k = 0; k < K; k++)
		{
			F_ik = 0;
			for(int j = 0; j < NSegments; j++)
				F_ik += D[i][j][k];
			
			Y_hat_i += F_ik * W[k];
		}
		return Y_hat_i;
	}
	
	
	
	// reconstruct the point l of the j-th segment of the i-th time series
	public double Reconstruct(int i, int j, int l)
	{
		double S_ijl = 0;
		
		// apply a convolution of k-many patterns and their degrees of membership
		// use the sum of the l-th points from each pattern P
		for(int k = 0; k < K; k++)
			S_ijl += D[i][j][k] * P[k][l];
		
		return S_ijl;
	}
	
	// measure the reconstruction loss
	public double MeasureRecontructionLoss()
	{
		double reconstructionLoss = 0, e_ijl = 0;
		
		// iterate through all the time series
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int l = 0; l < L; l++)
				{
					e_ijl = S[i][j][l] - Reconstruct(i, j, l);
					reconstructionLoss += e_ijl*e_ijl; 					
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
		
		return - Y.get(i)*Math.log( sig_y_i ) - (1-Y.get(i))*Math.log(1-sig_y_i); 
	}
	
	// compute the accuracy loss of the train set
	public double AccuracyLossTrainSet()
	{
		double accuracyLoss = 0;
		
		for(int i = 0; i < NTrain; i++)
			accuracyLoss += AccuracyLoss(i);
		
		return accuracyLoss;
	}
	// compute the accuracy loss of the train set
	public double AccuracyLossTestSet()
	{
		double accuracyLoss = 0;
		
		for(int i = NTrain; i < NTrain+NTest; i++)
			accuracyLoss += AccuracyLoss(i);
		
		return accuracyLoss;
	}
	
	
	public void LearnLR()
	{
		double e_ijl = 0;
		
		double regDConst = lambdaD/L, regPConst = (2*lambdaP)/((NTrain+NTest)*NSegments); 
		
		for(int i = 0; i < NTrain+NTest; i++) 
		{
			for(int j = 0; j < NSegments; j++) 
			{
				for(int l = 0; l < L; l++) 
				{
					e_ijl = S[i][j][l] - Reconstruct(i, j, l); 
					
					for(int k = 0; k < K; k++) 
					{
						D[i][j][k] = D[i][j][k] - eta*(-2*beta*e_ijl*P[k][l] + regDConst*D[i][j][k]); 
						P[k][l] = P[k][l] - eta*(-2*beta*e_ijl*D[i][j][k] + regPConst*P[k][l]); 
					}					
				}
			}
		}
	}
	
	public void LearnLA()
	{
		LearnLA(false);
	}
	
	public void LearnLA(boolean updateOnlyW)
	{
		if( beta == 1) return;
		
		double e_i = 0, F_ik = 0;
		double regWConst = (2*lambdaW)/NTrain;
		
		for(int i = 0; i < NTrain; i++)
		{
			e_i = Y.get(i) - Sigmoid.Calculate(Predict(i));
			
			for(int k = 0; k < K; k++)
			{
				F_ik = 0;
			
				for(int j = 0; j < NSegments; j++)
				{
					D[i][j][k] -= eta*((1-beta)*-2*e_i*W[k] + lambdaD*D[i][j][k]);
					
					F_ik += D[i][j][k];
				}	
					
				W[k] -= eta*((1-beta)*-2*e_i*F_ik + regWConst*W[k]);								
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
			LearnLA();
			
			lossHistory.add(LR+LATrain);
			
			if( lossHistory.size() > 10 )
			{
				//if( LR+LATrain > lossHistory.get( lossHistory.size() - 9  )  )
					//break;
			}
		}
		
		return GetMCRTestSet(); 
	}

}
