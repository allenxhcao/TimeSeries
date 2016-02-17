package TimeSeries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.StatisticalUtilities;
import Classification.NearestNeighbour;
import DataStructures.Coordinate;
import DataStructures.DataSet;
import DataStructures.Matrix;


public class LocalConvolutions 
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
	
	// the learning rate eta
	public double eta;
	
	// the number of iterations
	public int maxIter;
	
	// the regularization parameters
	public double lambdaP, lambdaD, lambdaW, alpha;
	
	// the delta increment between segments of the sliding window
	public int deltaT;
	
	List<int[]> seriesIndexes;
	
	Random rand = new Random();
	
	// constructor
	public LocalConvolutions()
	{
		isSupervised = false;
		
		
	}
	
	// initialize the data structures
	public void Initialize()
	{
		Logging.println("NTrain="+NTrain + ", NTest="+NTest + ", M_i="+M, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L="+L, LogLevel.DEBUGGING_LOG);
		Logging.println("eta="+ eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lambdaD="+ lambdaD + ", lambdaF="+ lambdaP + ", lamdaW="+lambdaW + ", " +  ", alpha="+alpha, LogLevel.DEBUGGING_LOG);
		
		// set the delta increment of the sliding window to a fraction of the segment size
		deltaT = (int)L/10; 
		
		//deltaT = 1;
		
		if( deltaT < 1) deltaT = 1;
		else if(deltaT > L ) deltaT = L;
		
		NSegments = (M-L)/deltaT; 
		 
		Logging.println("deltaT="+ deltaT + ", NSegments="+ NSegments, LogLevel.DEBUGGING_LOG);
		
		// partition the dataset into segments
		SegmentTimeSeriesDataset();
				
		// initialize the patterns to some random segments
		P = new double[K][L];
		for(int k= 0; k < K; k++)
		{
			int i = rand.nextInt(NTrain + NTest);
			int j = rand.nextInt(NSegments);
			
			for(int l = 0; l < L; l++)
				P[k][l]= S[i][j][l]; 
		}
		
		// initialize the degree of membership to 1 for the closest initial pattern
		// and 0 for all the others
		D = new double[NTrain+NTest][NSegments][K];
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
			{
				// compute the distance between the i,j-th segment and the k-th pattern
				for(int k = 0; k < K; k++)
				{					
					double dist = 0;
					for(int l = 0; l < L; l++)
					{
						double err = S[i][j][l] - P[k][l];
						dist += err*err;
					}
					
					D[i][j][k] = 1.0/(dist+1.0);
				}	
			}
		
		// initialize W between (-1,1)
		W = new double[K];
		for(int k = 0; k < K; k++)
			W[k] = 2*rand.nextDouble()-1;
		
		// set the labels to be binary -1 and 1
		for(int i = 0; i < NTrain+NTest; i++)
		{
			if(Y.get(i)!=1) Y.set(i, 0, -1.0);
		}
		
		
		// scale the regularizations
		lambdaP = 2*lambdaP/( (NTrain+NTest)*NSegments );
		lambdaW = 2*lambdaW/NTrain;
		
		
		// initialize the indices for iteration
		seriesIndexes = new ArrayList<int [] >();
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++) 
					seriesIndexes.add(new int[]{i, j});
		
		Collections.shuffle( seriesIndexes );
	
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
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
	
	// update the reconstruction loss by stochastically updating
	// the partial error of every point of a segment
	public void AppyOneIteration()
	{
		double E_ijl = 0, D_ijk = 0, P_kl = 0;
		
		int numIndexes = seriesIndexes.size();
		
		for(int indexCount = 0; indexCount < numIndexes; indexCount++)
		{
			int [] index = seriesIndexes.get(indexCount);
			int i = index[0], j = index[1];
			
			for(int l = 0; l < L; l++)
			{
				E_ijl = S[i][j][l] - Reconstruct(i,j,l);
				
				for(int k = 0; k < K; k++)
				{
					D_ijk = D[i][j][k];
					P_kl = P[k][l];
					
					D[i][j][k] -= eta * (-2*E_ijl*P_kl + (lambdaD/L)*D_ijk);
					P[k][l] -= eta * (-2*E_ijl*D_ijk + lambdaP*P_kl);
				}
			}
			
			if(i < NTrain)
				UpdateAccuracyLoss(i);
		}
	}
	
	// update accuracy Loss
	public void UpdateAccuracyLoss()
	{
		for(int i = 0; i < NTrain; i++)
		{
			UpdateAccuracyLoss(i);
		}
	}
	
	public void UpdateAccuracyLoss(int i)
	{
		double Y_hat_i = 0, Y_Y_hat_i = 0, grad_W_k = 0, grad_D_ijk = 0;
		double F_i[] = new double[K];
		
		// compute F_i, vartheta_i and Y_Y_hat_i
		Y_hat_i = 0;
		for(int k = 0; k < K; k++)
		{
			F_i[k] = 0;				
			for(int j = 0; j < NSegments; j++)
				F_i[k] += D[i][j][k];
			
			Y_hat_i += F_i[k] * W[k];
		}			
		Y_Y_hat_i = Y.get(i)*Y_hat_i;
		
		// update all the weights and degrees of memberships
		for(int k = 0; k < K; k++)
		{
			// update the weights
			if(Y_Y_hat_i <= 0) grad_W_k = -Y.get(i)*F_i[k];
			else if( 0 < Y_Y_hat_i && Y_Y_hat_i < 1 ) grad_W_k = Y.get(i)*F_i[k] - 1;
			else grad_W_k = 0;
			
			W[k] -= eta*(alpha*grad_W_k + lambdaW*W[k] );
			
			// update the degrees of membership
			for(int j = 0; j < NSegments; j++)
			{
				if(Y_Y_hat_i <= 0) grad_D_ijk = -Y.get(i)*W[k];
				else if( 0 < Y_Y_hat_i && Y_Y_hat_i < 1 ) grad_D_ijk = Y.get(i)*W[k] - 1;
				else grad_D_ijk = 0; 
					
				D[i][j][k] -= eta*(alpha*grad_D_ijk + lambdaD*D[i][j][k] );
			}
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
		double E_ijl = 0;
		double reconstructionLoss = 0;
		
		// iterate through all the time series
		int numIndexes = seriesIndexes.size();
		for(int indexCount = 0; indexCount < numIndexes; indexCount++)
		{
			int [] index = seriesIndexes.get(indexCount);
			int i = index[0], j = index[1];
			
			for(int l = 0; l < L; l++)
			{
				// compute the error in reconstructing point l of the j-th segment
				// of the i-th time seriees
				E_ijl = S[i][j][l] - Reconstruct(i,j,l);
				reconstructionLoss += E_ijl*E_ijl; 
			}
		}
		
		return reconstructionLoss;
	}
	
	// optimize the objective function
	public double Learn()
	{
		// initialize the data structures
		Initialize();
		
		// apply the stochastic gradient descent in a series of iterations
		for(int iter = 0; iter < maxIter; iter++)
		{

			// measure the loss
			double reconstructionLoss = MeasureRecontructionLoss(); 
			double accuracyLoss = MeasureAccuracyLoss();
			double mcrNN = ClassifyNearestNeighbor();
			Logging.println("It=" + iter + ", LR="+reconstructionLoss + ", LA="+accuracyLoss + ",MCRNN="+mcrNN, LogLevel.DEBUGGING_LOG);
			
			// fix the reconstruction error of every cell
			// for both reconstruction and accuracy losses
			AppyOneIteration();
			
		}
		
		return ClassifyNearestNeighbor();
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
		
		double Y_Y_hat_i = 0;
		double incorrectClassifications = 0;
		
		for(int i = 0; i < NTrain; i++)
		{
			Y_Y_hat_i = Y.get(i)*Predict(i);
			
			if(Y_Y_hat_i <= 0) incorrectClassifications += 1.0;
			
			if(Y_Y_hat_i <= 0) accuracyLoss += 0.5 - Y_Y_hat_i;
			else if( 0 < Y_Y_hat_i && Y_Y_hat_i <= 1 ) accuracyLoss += 0.5*(1- Y_Y_hat_i)*(1- Y_Y_hat_i);
			else accuracyLoss += 0; 
		}
		
		mcrTrain = incorrectClassifications/NTrain;
		
		incorrectClassifications = 0;
		for(int i = NTrain; i < NTrain+NTest; i++)
		{
			Y_Y_hat_i = Y.get(i)*Predict(i);
			
			if(Y_Y_hat_i <= 0) incorrectClassifications += 1.0;
		}
		mcrTest = incorrectClassifications/NTest;
		
		Logging.println("MCR=["+mcrTrain + ", "+mcrTest + "]", LogLevel.DEBUGGING_LOG);
		
		
		
		return accuracyLoss;
	}
	
	
}
