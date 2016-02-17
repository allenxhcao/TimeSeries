package FeatureRich;

import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;

import Clustering.KMeans;
import Utilities.Logging;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;

public class LearnUnsupervisedBagofWords 
{
	// the predictor segments
	public double [][][] S;
	// the binary targets
	public double [][] Y;
	
	// the learned codebook
	public double [][] P;
	
	// the final frequencies
	public double [][] F;
	
	// number of training and testing instances
	public int ITrain, ITest;

	// length of shapelet
	public int L;
	// number of latent patterns
	public int K;
	// number of classes
	public int C; 
	
	// learning rate
	public double eta;
	// number of epochs for the gradient descent
	public int maxEpochs;
	// regularization
	public double lambdaW;
	
	double [][] W;
	double [][] GradHistW;
	
	double [] biasW;
	double [] GradHistBiasW;
	
	
	// learn the unsupervised codebook by the specified method
	// return the accuracy of the learned codebook, in a bag-of-words way
	public double LearnUnsupervisedCodebook()
	{
		ComputeKMeansCodebook();
		
		ComputeFrequencies();
		
		return LearnClassifier(); 
	}
	
	
	// initialize the shapelets from the centroids of the segments
	public void ComputeKMeansCodebook() 
	{		 
		P = null;
		int numSegments = 0;
		for(int i= 0; i < ITrain; i++) 
			numSegments += S[i].length;
		
		double [][] segmentsR = new double[numSegments][L]; 
		
		int idxFirstInstanceSegment = 0; 
		
		for(int i= 0; i < ITrain; i++)  
		{ 
			for(int j= 0; j < S[i].length; j++) 
			{ 
				for(int l = 0; l < L; l++) 
					segmentsR[idxFirstInstanceSegment + j][l] = S[i][j][l]; 
			} 
			
			idxFirstInstanceSegment += S[i].length; 
		}
		
		KMeans kmeans = new KMeans();
		P = kmeans.InitializeKMeansPP(segmentsR, K, 100);  
						
		if( P == null )
			System.out.println("P not set");
		
	}
	
	// compute the frequencies of the i-th instances
	public void ComputeFrequencies()
	{
		// initialize frequencies matrix
		F = new double[ITrain+ITest][K];
		
		// compute all the frequencies
		for(int i = 0; i < ITrain+ITest; i++)
		{
			double D_i[][] = new double[S[i].length][K];
			
			for(int j=0; j < S[i].length; j++)
			{
				// first set all distances to 0
				for(int k = 0; k < K; k++)
					D_i[j][k] = 0;
				
				// then set the closest segment to 1
				int minPatternIndex = -1;
				double minDist = Double.MAX_VALUE;
				
				for(int k = 0; k < K; k++)
				{
					double dist =  StatisticalUtilities.SumOfSquares(S[i][j], P[k]);
					if(dist < minDist)
					{
						minDist = dist;
						minPatternIndex = k;
					}
				}
				
				D_i[j][minPatternIndex] = 1;
				
			}
			
			// compute the frequencies
			
			for(int k = 0; k < K; k++)
			{
				F[i][k] = 0;
				for(int j=0; j < S[i].length; j++)
				{
					F[i][k] += D_i[j][k];
				}
				
				F[i][k] = F[i][k] / (double) S[i].length; 
			}
		}	
	}
	
	// learn a classifier using the segments, patterns and binary targets
	public double LearnClassifier()
	{
		
		W = new double[C][K];
		GradHistW = new double[C][K];
		
		biasW = new double[C];
		GradHistBiasW = new double[C];
		
		// set gradient histories to 0
		for(int c=0; c<C; c++)
		{
			GradHistBiasW[c] = 0.0;			
			for(int k=0; k<K; k++)
				GradHistW[c][k] = 0.0; 
		}
		
		double eps = 0.000001;  
		
		Logging.println("UnsupervisedBoW:LearnClassifier: eta=" + eta + ", epochs=" + maxEpochs 
				+ ", C=" + C + ", K=" + K + ", lambdaW=" + lambdaW);  
		
		for(int iter=0; iter<maxEpochs; iter++)
		{
			for(int i = 0; i < ITrain; i++) 
		    {
				double Y_hat_ic = 0; 
				// the gradients
				double dL_dYic = 0, dYic_dWck = 0, dReg_dWck = 0, dOic_dWck = 0, dOic_dWc0 = 0;
				
				
				for(int c = 0; c < C; c++)
				{
					// compute the estimated target variable
					Y_hat_ic = biasW[c];
					for(int k = 0; k < K; k++)
						Y_hat_ic += F[i][k]*W[c][k];
					
					
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
					for(int k = 0; k < K; k++)
					{
						dYic_dWck = F[i][k];
						dReg_dWck = (2.0/(double)ITrain)*lambdaW*W[c][k];
						
						// compute the partial derivative of the objective with respect to 
						// the decomposed objective function 
						dOic_dWck = dL_dYic*dYic_dWck + dReg_dWck; 
						
						// update the history of weights' gradients 
						GradHistW[c][k] += dOic_dWck*dOic_dWck; 
						// update the weight
						W[c][k] -= (eta/(eps+Math.sqrt(GradHistW[c][k]))) * dOic_dWck; 										
					}		
				}
		    }		
		}
		
		// compute the classification performance
		ClassificationPerformance cp = new ClassificationPerformance();
		cp.ITrain = ITrain;
		cp.ITest = ITest;
		cp.K = K;
		cp.C = C;
		
		cp.ComputeClassificationAccuracy(F, Y, W, biasW); 
		
		Logging.println("UnsupervisedCodebookLearning: lossTrain=" + cp.trainLoss + ", lossTest=" + cp.testLoss 
							+ ", MCRTrain="+ cp.trainMCR + ", MCRTest=" + cp.testMCR);
		
		return 0;
	}
}
