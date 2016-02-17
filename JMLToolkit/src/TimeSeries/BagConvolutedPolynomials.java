package TimeSeries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.swing.text.Utilities;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import Regression.TimeSeriesPolynomialApproximation;
import TimeSeries.BagOfPatterns.RepresentationType;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;
import Classification.NearestNeighbour;
import Classification.WekaClassifierInterface;
import DataStructures.Coordinate;
import DataStructures.DataSet;
import DataStructures.Matrix;


public class BagConvolutedPolynomials 
{
	public boolean isSupervised;
	
	// number of training and testing instances
	public int NTrain, NTest;
	// length of a time-series
	public int M;
	// length of a segment
	public int L;
	// number of latent patterns
	public int K;
	// degree of the polynomial
	public int degree;
	
	// local segments
	int NSegments;
	double S[][][];
	
	// latent patterns
	double P[][];
	
	// polynomial coefficients
	double beta[][];
	
	// degrees of membership
	double D[][][];
	
	// the residual errors
	double E[][][];
	
	// classification weights
	double W[];
	
	// the classification errors
	double Phi[];
	
	// the histogram of patterns F_i, sum of D_ij
	double F[][];
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y;
		
	// the number of iterations
	public int maxIter;
	
	// the regularization parameters
	public double lambdaD, lambdaBeta, lambdaW, alpha;
	
	// the delta increment between segments of the sliding window
	public int deltaT;
	
	public double svmC = 1.0, svmDegree= 1.0;
	
	//List<int[]> seriesIndexes;
	
	Random rand = new Random();
	
	
	
	// constructor
	public BagConvolutedPolynomials()
	{
		isSupervised = false;
		deltaT = 1;
		
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		Logging.println("NTrain="+NTrain + ", NTest="+NTest + ", M_i="+M, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L="+L + ", degree="+degree, LogLevel.DEBUGGING_LOG); 
		Logging.println("maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lambdaD="+ lambdaD + ", lambdaF="+ lambdaBeta + ", lamdaW="+lambdaW + ", alpha="+alpha, LogLevel.DEBUGGING_LOG);
		
		if( deltaT < 1) deltaT = 1;
		else if(deltaT > L ) deltaT = L;
		
		
		NSegments = (M-L)/deltaT; 
		
		Logging.println("deltaT="+ deltaT + ", NSegments="+ NSegments, LogLevel.DEBUGGING_LOG);
		
		SegmentTimeSeriesDataset();		
		InitializeCoefficientsRandomly();
		InitializeHardMembershipsToClosestPattern();
	
		
		E = new double[NTrain+NTest][NSegments][L];
		Phi = new double[NTrain];
		F = new double[NTrain][K];
		
		W = new double[K];
		for(int k = 0; k < K; k++) W[k] = 2*rand.nextDouble()-1;
		
		// initialize the errors tensor E_i
		InitializeErrors();
		
		// make the labels binary +1, -1
		if( isSupervised)
		{
			// set the labels to be binary -1 and 1
			for(int i = 0; i < NTrain+NTest; i++)
			{
				if(Y.get(i)!=1) Y.set(i, 0, -1.0);
			}
			
			InitializeWeights();
		}
		
		
		double initialError = MeasureRecontructionLoss();
		double mcr = ClassifyNearestNeighbor();
		double accuracyLoss = 0;
		
		if(isSupervised)
			accuracyLoss=MeasureAccuracyLoss();
		
		Logging.println("InitRecErr="+initialError+", InitAccErr="+ accuracyLoss + 
					", InitMCR=" +mcr, LogLevel.DEBUGGING_LOG);
		
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// initialize the patterns from random segments
	public void InitializeCoefficientsRandomly()
	{
		TimeSeriesPolynomialApproximation tspa = new TimeSeriesPolynomialApproximation(L, degree);
		// initialize the patterns to some random segments 
		beta = new double[K][degree+1];
		
		for(int k = 0; k < K; k++)
		{
			int i = rand.nextInt(NTrain + NTest);
			int j = rand.nextInt(NSegments);
			
			beta[k] = tspa.FitPolynomialToSubSeries(S[i][j]);
		}
		P = new double[K][L];
		
	}
	
	// initialize the degree of membership to the closest pattern
	public void InitializeHardMembershipsToClosestPattern()
	{
		ComputeP();
		
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
		for(int iter = 0; iter < maxIter; iter++)
		{
			UpdateWeightsCoordDesc();
			
			double accuracyLoss = MeasureAccuracyLoss();
			
			Logging.println("InitW: It=" + iter + ", LA="+accuracyLoss, LogLevel.DEBUGGING_LOG);
		}
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
				{
					//System.out.println(i + ", "+ ", "+ j + ", " + ((j*deltaT) + l) ) ;
					S[i][j][l] = T.get(i, (j*deltaT) + l); 
				}
				
				// normalize the segment 
				double [] normalizedSegment = StatisticalUtilities.Normalize(S[i][j]);
				for(int l = 0; l < L; l++)
				{
					S[i][j][l] = normalizedSegment[l];
				}
				
			}
		}
		
		Logging.println("Partion to Normalized Segments Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// apply one unsupervised iteration over the patterns and memberships
 	public void ApplyOneUnsupervisedCoordDescIteration()
	{
		UpdateMembershipsUnsupervisedCoordDesc();
		UpdateCoefficientsCoordDesc();
	}
	
 	// update all the D_ijk cells in a coordinate descent fashion
 	public void UpdateMembershipsUnsupervisedCoordDesc()
 	{
 		ComputeP();
 		
 		double z = 0, nominator = 0, denominator = 0;
		
		// update all D_ijk
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int k = 0; k < K; k++)
				{
					// compute optimal D_ijk denoted as z
					z = 0; nominator = 0; denominator = 0;
					for(int l = 0; l < L; l++)
					{
						nominator += (E[i][j][l] + D[i][j][k]*P[k][l])*P[k][l];
						denominator += P[k][l]*P[k][l];
					}
					denominator += lambdaD;
					z = nominator/denominator;
					
					//update the errors
					for(int l = 0; l < L; l++)
						E[i][j][l] = E[i][j][l] - (z - D[i][j][k])*P[k][l]; 
					
					// update the membership
					D[i][j][k] = z;
				}
			}
		}
		
 	}
 	
 	// update all the P_kl cells in a coordinate descent fashion
 	public void UpdateCoefficientsCoordDesc()
 	{
 		double z = 0, nominator = 0, denominator = 0, cte = 0;
 		
 		for(int k = 0; k < K; k++)
 		{
 			for(int d = 0; d < degree+1; d++)
 			{
 				z=0; nominator=0; denominator=0;
 				
 				for(int i = 0; i < NTrain+NTest; i++)
 				{
 					for(int j = 0; j < NSegments; j++)
 					{
 						for(int l = 0; l < L; l++)
 						{
 							cte = D[i][j][k]*StatisticalUtilities.PowerInt(l, d);
 							nominator += (E[i][j][l] + cte*beta[k][d])*cte;
 							denominator += cte*cte;
 						}
 					}
 				}
 				denominator += lambdaBeta;
 				
 				z = nominator / denominator;
 				
 				for(int i = 0; i < NTrain+NTest; i++)
 					for(int j = 0; j < NSegments; j++)
 						for(int l = 0; l < L; l++)
 						{
 							cte = D[i][j][k]*StatisticalUtilities.PowerInt(l, d);
 							E[i][j][l] = E[i][j][l] - (z - beta[k][d])*cte;
 						}
 					
 				beta[k][d] = z;
 			}
 		}
 		
		
				
				
 	}
 	
 	// compute frequencies
 	public void ComputeFrequencies()
 	{
 		for(int i = 0; i < NTrain; i++)
 			for(int k = 0; k < K; k++)
 			{
				F[i][k] = 0;
				for(int j = 0; j < NSegments; j++)
					F[i][k] = F[i][k] + D[i][j][k];
 			}
 	}
 	
 	// update the 
 	public void UpdateWeightsCoordDesc()
 	{
 		double nominator = 0, denominator = 0, z = 0; 		
 		
 		// compute the frequencies F
 		ComputeFrequencies();
 		
 		for(int k = 0; k < K; k++)
 		{
 			nominator = 0;
 			denominator = 0;
 			
 			for(int i = 0; i < NTrain; i++)
 			{
 				nominator += (Phi[i]+ W[k]*F[i][k])*F[i][k];
 				denominator += F[i][k]*F[i][k];
 			}
 			denominator += lambdaW;
 			
 			z = nominator/denominator;
 			
 			for(int i = 0; i < NTrain; i++)
 				Phi[i] = Phi[i] - (z - W[k])*F[i][k];
 			
 			W[k] = z;
 		}
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
	
	// compute the patterns P from coefficients beta
	public void ComputeP()
	{
		for(int k = 0; k < K; k++)
			for(int l = 0; l < L; l++)
			{ 
				P[k][l] = 0;
				for(int d = 0; d < degree+1; d++) 
					P[k][l] += beta[k][d] * StatisticalUtilities.PowerInt(l, d);
			}
		
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
		double reconstructionLoss = 0;
		
		// iterate through all the time series
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int l = 0; l < L; l++)
				{
					reconstructionLoss += E[i][j][l]*E[i][j][l]; 
				}
			}
		}
		
		return reconstructionLoss;
	}
	
	// initalize the error matrix
	public void InitializeErrors()
	{
		ComputeP();
		
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
					E[i][j][l]  = S[i][j][l] - Reconstruct(i,j,l);
		
		// initialize the label residuals
		if(isSupervised)
		{
			for(int i = 0; i < NTrain; i++)
				Phi[i] = Y.get(i) - Predict(i);
		}
	}
	
	

	public double ClassifyNearestNeighbor()
	{
		Matrix F = new Matrix(NTrain+NTest, K);
		
		// count the frequencies and store in a new representation
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int k = 0; k < K; k++)
			{
				double F_ik = 0;
				
				for(int j = 0; j < NSegments; j++)
					F_ik += D[i][j][k];
				
				F.set(i, k, F_ik);
			}
		}
		
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y, 0, NTrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(F, Y, NTrain, NTrain+NTest); 
	    
	    NearestNeighbour nn = new NearestNeighbour("euclidean");
	    
		return nn.Classify(trainSetHist, testSetHist);
	}
	
	public double MeasureAccuracyLoss()
	{
		double accuracyLoss = 0;
		double mcrTrain = 1, mcrTest = 1;
		
		double Y_hat_i = 0;
		double incorrectClassifications = 0;
		
		for(int i = 0; i < NTrain; i++)
		{
			Y_hat_i = Predict(i);
			
			accuracyLoss += Phi[i]*Phi[i]; 
			
			if( (Y.get(i) == 1 && Y_hat_i <= 0) 
					|| (Y.get(i) == -1 && Y_hat_i > 0)  ) 
				incorrectClassifications += 1.0;
		}
		
		mcrTrain = incorrectClassifications/NTrain;
		
		incorrectClassifications = 0;
		for(int i = NTrain; i < NTrain+NTest; i++)
		{
			Y_hat_i = Predict(i);
			
			if( (Y.get(i) == 1 && Y_hat_i <= 0) 
					|| (Y.get(i) == -1 && Y_hat_i > 0)  ) 
				incorrectClassifications += 1.0;
		}
		mcrTest = incorrectClassifications/NTest;
		
		Logging.println("MCR=["+mcrTrain + ", "+mcrTest + "]", LogLevel.DEBUGGING_LOG);
		
		
		return accuracyLoss;
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
			double reconstructionLoss = MeasureRecontructionLoss(); 
			
			double accuracyLoss = 0;
			if( isSupervised )	accuracyLoss = MeasureAccuracyLoss();
			
			double mcr = ClassifyNearestNeighbor();
			//double mcr = ClassifySVM();
			
			Logging.println("It=" + iter + ", LR="+reconstructionLoss + ", LA="+accuracyLoss + ", MCR="+mcr, LogLevel.DEBUGGING_LOG);
			
			// fix the reconstruction error of every cell
			// for both reconstruction and accuracy losses
			
			if(!isSupervised)
				ApplyOneUnsupervisedCoordDescIteration();
			else 
				ApplyOneSupervisedCoordDescIteration();
			
			
			if(iter > 5)
			{
				if(reconstructionLoss+accuracyLoss > lossHistory.get(lossHistory.size()-4))
					break;
			}
			
			lossHistory.add(reconstructionLoss+accuracyLoss);
		}
		
		return ClassifyNearestNeighbor();
	}

	private void ApplyOneSupervisedCoordDescIteration() 
	{
		UpdateMembershipsUnsupervisedCoordDesc();
		UpdateCoefficientsCoordDesc();
		UpdateWeightsCoordDesc();
	}
}
