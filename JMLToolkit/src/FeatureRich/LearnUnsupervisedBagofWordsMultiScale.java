package FeatureRich;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;

import Clustering.KMeans;
import Utilities.Logging;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;
import Utilities.Logging.LogLevel;

public class LearnUnsupervisedBagofWordsMultiScale 
{
	// the predictor segments
	public double [][][][] S;
	// the binary targets 
	public double [][] Y;
	
	// the learned codebook
	public double [][][] P;
	
	// the final frequencies
	public double [][][] F;
	
	// number of training and testing instances
	public int ITrain, ITest;

	// length of shapelet
	public int [] L;
	// scales of the patterns
	public int R;
	// number of latent patterns
	public int [] K;
	// number of classes
	public int C; 
	
	// learning rate
	public double eta;
	// number of epochs for the gradient descent
	public int maxEpochs;
	// regularization
	public double lambdaW, lambdaP;
	
	double [][][] W;
	double [][][] GradHistW;
	
	double [] biasW;
	double [] GradHistBiasW;
	
	Random rand = new Random();
	
	
	public double testMCR = Double.MAX_VALUE;
	
	// learn the unsupervised codebook by the specified method
	// return the accuracy of the learned codebook, in a bag-of-words way
	public double LearnUnsupervisedCodebook()
	{
		ComputeKMeansCodebook();
		
		Logging.println("K-Means centroid computation finished.", LogLevel.DEBUGGING_LOG); 
		
		ComputeFrequencies();
		
		return LearnClassifier(); 
	}
	
	
	// initialize the shapelets from the centroids of the segments
	public void ComputeKMeansCodebook() 
	{		 
		P = new double[R][][];
		
		// store all the instances indexes for the scales
		List<Integer> rIdxs = new ArrayList<Integer>();
		for(int r = 0; r < R; r++) 
				rIdxs.add(r);
		
		Parallel_1x0.ForEach(rIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer r)
		    {
				int numSegments = 0;
				for(int i= 0; i < ITrain; i++) 
					numSegments += S[i][r].length;
				
				double [][] segmentsR = new double[numSegments][L[r]];   
				
				int idxFirstInstanceSegment = 0; 
				
				for(int i=0; i<ITrain; i++)   
				{ 
					for(int j= 0; j < S[i][r].length; j++)  
						for(int l = 0; l < L[r]; l++)  
							segmentsR[idxFirstInstanceSegment + j][l] = S[i][r][j][l]; 
					
					idxFirstInstanceSegment += S[i][r].length;  
				}
				
				KMeans kmeans = new KMeans();
				P[r] = kmeans.InitializeKMeansPP(segmentsR, K[r], 100);   
								
				if( P[r] == null ) 
					System.out.println("P not set");
			}
		});
		
	}
	
	// compute the frequencies of the i-th instances
	public void ComputeFrequencies()
	{
		// initialize frequencies matrix
		F = new double[ITrain+ITest][R][];
	
		// compute all the distances
		for(int i = 0; i < ITrain+ITest; i++)
		{
			double D_i[][][] = new double[R][][];
			
			for(int r=0; r<R; r++)
			{
				// initialize distances
				D_i[r] = new double[S[i][r].length][K[r]];
				
				for(int j=0; j < S[i][r].length; j++)
				{
					// first set all distances to 0 
					for(int k = 0; k < K[r]; k++)  
						D_i[r][j][k] = 0;  
					 
					// then set the closest segment to 1
					int minPatternIndex = -1; 
					double minDist = Double.MAX_VALUE; 
					
					for(int k = 0; k < K[r]; k++) 
					{ 
						double dist =  StatisticalUtilities.SumOfSquares(S[i][r][j], P[r][k]); 
						if(dist < minDist) 
						{ 
							minDist = dist; 
							minPatternIndex = k; 
						} 
					} 
					
					D_i[r][j][minPatternIndex] = 1; 
					
				}
			}
			
			// compute the frequencies
			

			
			for(int r=0; r<R; r++)
			{
				F[i][r] = new double[K[r]];
				
				for(int k=0; k<K[r]; k++) 
				{
					F[i][r][k] = 0;
					for(int j=0; j < S[i][r].length; j++)  
						F[i][r][k] += D_i[r][j][k];
					
					F[i][r][k] = F[i][r][k] / (double) S[i][r].length;  
				}
			}
		}	
		
	}
	
	// learn a classifier using the segments, patterns and binary targets
	public double LearnClassifier()
	{
		
		W = new double[C][R][];
		GradHistW = new double[C][R][];
		
		biasW = new double[C];
		GradHistBiasW = new double[C];
		
		// set gradient histories to 0
		for(int c=0; c<C; c++)
		{
			GradHistBiasW[c] = 0.0;
			
			for(int r=0; r<R; r++)
			{
				W[c][r] = new double[K[r]];
				GradHistW[c][r] = new double[K[r]];
				
				for(int k=0; k<K[r]; k++)
				{
					W[c][r][k] = 2*rand.nextDouble()-1;
					GradHistW[c][r][k] = 0.0;
				}
			}
		}
		
		 
		
		Logging.println("UnsupervisedBoW:LearnClassifier: eta=" + eta + ", epochs=" + maxEpochs 
				+ ", C=" + C + ", L=" + L + ", R=" + R + ", lambdaW=" + lambdaW);     
		Logging.println("K:"); 
		Logging.println(K);  
		
		// store all the instances indexes for the scales
		List<Integer> iIdxs = new ArrayList<Integer>();
		for(int i = 0; i < ITrain; i++) 
				iIdxs.add(i); 
		
		for(int iter=0; iter<maxEpochs; iter++)
		{
			Parallel_1x0.ForEach(iIdxs, new ForEachTask_1x0<Integer>() 
			{
				public void iteration(Integer i)
			    {
					double eps = 0.000001; 
					double Y_hat_ic = 0; 
					// the gradients
					double dL_dYic = 0, dYic_dWcrk = 0, dReg_dWcrk = 0, dOic_dWcrk = 0, dOic_dWc0 = 0;
					
					
					for(int c = 0; c < C; c++)
					{
						// compute the estimated target variable
						Y_hat_ic = biasW[c]; 
						for(int r=0; r<R; r++)
							for(int k = 0; k < K[r]; k++)
								Y_hat_ic += F[i][r][k]*W[c][r][k];
						
						
						// compute the derivative of the 
						double z = Y[i][c]*Y_hat_ic;					
						if( z <= 0 )
							dL_dYic = -Y[i][c];
						else if(z > 0 && z < 1 )
							dL_dYic = (z-1)*Y[i][c];
						else
							dL_dYic = 0;
					
						
						// compute the derivative of the objective wrt bias term of weights
						dOic_dWc0 = dL_dYic;
						// update the gradient history of the bias
						GradHistBiasW[c] += dOic_dWc0*dOic_dWc0;
						// update the bias term
						biasW[c] -= (eta/(eps+Math.sqrt(GradHistBiasW[c])))*dOic_dWc0; 
						
						// update all the patterns and weights
						for(int r=0; r<R; r++)
							for(int k = 0; k < K[r]; k++)
							{
								dYic_dWcrk = F[i][r][k];
								dReg_dWcrk = (2.0/(double)ITrain)*lambdaW*W[c][r][k];
								
								// compute the partial derivative of the objective with respect to 
								// the decomposed objective function 
								dOic_dWcrk = dL_dYic*dYic_dWcrk + dReg_dWcrk; 
								
								// update the history of weights' gradients 
								GradHistW[c][r][k] += dOic_dWcrk*dOic_dWcrk; 
								// update the weight
								W[c][r][k] -= (eta/(eps+Math.sqrt(GradHistW[c][r][k]))) * dOic_dWcrk; 										
							}		
					}
			    }
			});
		}
		
		
		// compute the classification performance
		ClassificationPerformanceTensor cp = new ClassificationPerformanceTensor();
		cp.ITrain = ITrain;
		cp.ITest = ITest;  
		cp.K = K;
		cp.R = R;
		cp.C = C;
		
		cp.ComputeClassificationAccuracy(F, Y, W, biasW); 
		
		Logging.println("UnsupervisedCodebookLearning: lossTrain=" + cp.trainLoss + ", lossTest=" + cp.testLoss 
							+ ", MCRTrain="+ cp.trainMCR + ", MCRTest=" + cp.testMCR);
		
		
		testMCR = cp.testMCR;
		
		return cp.testMCR;
	}
}
