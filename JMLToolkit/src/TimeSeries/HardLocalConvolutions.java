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


public class HardLocalConvolutions 
{
	// number of training and testing instances
	public int N, NTrain, NTest;
	public int Q;
	
	// number of segments
	int M;
	// length of a segment
	public int L;
	// number of latent patterns
	public int K;
	
	double S[][][];
	
	// latent patterns
	double P[][];
	
	// degrees of membership
	int D[][];
	
	// the offset among two segment starts
	public int deltaT;
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y;
	
	// a random number generator
	Random rand = new Random();
	
	// constructor
	public HardLocalConvolutions()
	{
		
		
	}
	
	// initialize the data structures
	public void Initialize()
	{
		N = NTrain+NTest;
		Logging.println("NTrain="+NTrain + ", NTest="+NTest +  ", N="+N +", Q="+Q, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L="+L, LogLevel.DEBUGGING_LOG);
		
		// set the delta increment of the sliding window to a fraction of the segment size
		deltaT = 1; 
				
		if( deltaT < 1) deltaT = 1;
		else if(deltaT > L ) deltaT = L;
		
		M = (Q-L)/deltaT; 
		 
		Logging.println("deltaT="+ deltaT + ", M_i="+ M, LogLevel.DEBUGGING_LOG);
		
		// partition the dataset into segments
		SegmentTimeSeriesDataset();
				
		// initialize the patterns to some random segments
		P = new double[K][L];
		for(int k= 0; k < K; k++)
		{
			for(int l = 0; l < L; l++)
				P[k][l] = 0.0;
			
			int i = rand.nextInt(N);
			int j = rand.nextInt(M);
			
			for(int l = 0; l < L; l++)
				P[k][l] = S[i][j][l];
			
		}
		
		// initialize the degree of membership to 1 for the closest initial pattern
		// and 0 for all the others
		D = new int[N][M];
		AssignMemberships();

		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// partition the time series into segments
	public void SegmentTimeSeriesDataset()
	{
		S = new double[N][M][L]; 
		
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < M; j++)
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
	
	// measure the reconstruction loss
	public double MeasureLoss()
	{
		double reconstructionLoss = 0;
		
		for(int i = 0; i < N; i++)
			for(int j = 0; j < M; j++)
				reconstructionLoss += StatisticalUtilities.SumOfSquares(S[i][j], P[D[i][j]]); 
		
		return reconstructionLoss;
	}
	
	// optimize the objective function
	public double Learn()
	{
		// initialize the data structures
		Initialize();
		double prevLoss = Double.MAX_VALUE;
		
		// apply the stochastic gradient descent in a series of iterations
		for(int iter=0; ; iter++)
		{
			// measure the loss
			
			// Center patterns
			CenterPatterns();
			// Assign memberships
			AssignMemberships();
			
			
			double loss = MeasureLoss(); 
			
			if(loss < prevLoss)
			{
				prevLoss = loss;
				
				double mcr = ClassifyNearestNeighbor();
				Logging.println("It=" + iter + ", L="+loss + ", MCR="+mcr, LogLevel.DEBUGGING_LOG);
			}
			else
			{
				break;
			}
		}
		
		return ClassifyNearestNeighbor();
	}
	
	// center the patterns according to memberships
	private void CenterPatterns() 
	{
		// count how many segments does each pattern have
		// set the frequencies and the pattern values to zero
		double Ptemp[][] = new double[K][L];
		
		double frequencies[] = new double[K];
		for(int k=0; k < K; k++)
		{
			for(int l=0; l < L; l++)
				Ptemp[k][l] = 0;
			
			frequencies[k] = 0.0;
		}
		
		// go to every segment and add the content to the cluster it is assigned to
		// later we will average the contents
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < M; j++)
			{
				for(int l = 0; l < L; l++)
					Ptemp[ D[i][j] ][l] = Ptemp[ D[i][j] ][l] + S[i][j][l];
				
				frequencies[D[i][j]] = frequencies[D[i][j]] + 1.0;
			}
		}
		
		// average the content
		for(int k=0; k < K; k++)
		{
			if(frequencies[k] > 0.0)
			{ 
				for(int l=0; l < L; l++)
					P[k][l] = Ptemp[k][l] / frequencies[k];
			}
		}
		
		
	}

	// assign memberships to the closest pattern
	private void AssignMemberships() 
	{
		// iterate through all the segments and assign the membership degrees to the 
		// closest pattern
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < M; j++)
			{
				int closestK = 0;
				double closestDistance = Double.MAX_VALUE;
				
				for( int k = 0; k < K; k++ )
				{
					double dist = StatisticalUtilities.SumOfSquares(S[i][j], P[k]);
					
					if(dist < closestDistance)
					{
						closestDistance = dist;
						closestK = k;
					}
				}
				
				D[i][j] = closestK;
			}
		}
		
	}

	public double ClassifyNearestNeighbor()
	{
		// create a histogram of patterns
		Matrix H = new Matrix(NTrain+NTest, K);
		// set all frequencies to zero
		H.SetUniqueValue(0.0);
		
		// count the frequencies and store in a new representation
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < M; j++)
			{
				// numerosity reduction
				//if( j == 0 || ( D_i[i][j] != D_i[i][j-1]  ))
					H.set(i, D[i][j], H.get(i,D[i][j])+1 );
			}
		}
		
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(H, Y, 0, NTrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(H, Y, NTrain, N); 
	    
	    NearestNeighbour nn = new NearestNeighbour("euclidean");
	    
		return nn.Classify(trainSetHist, testSetHist);
	}
	
	
}
