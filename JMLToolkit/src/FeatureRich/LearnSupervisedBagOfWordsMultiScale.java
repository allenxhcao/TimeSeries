package FeatureRich;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageTranscoder;

import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;

import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;
import Clustering.KMeans;
import DataStructures.DataSet;
import DataStructures.Matrix;
import FeatureRich.SupervisedCodebookLearning.LossTypes;
import FeatureRich.SupervisedCodebookLearning.SimilarityTypes;


public class LearnSupervisedBagOfWordsMultiScale 
{	 
	// number of training and testing instances
	public int ITrain, ITest;
	// the scales of learned patterns
	public int R;
	// length of shapelet
	public int [] L; 
	// number of latent patterns
	public int [] K; 
	
	// number of classes
	public int C; 
	
	// patterns
	double P[][][];	
	// classification weights
	double W[][][]; 
	double biasW[]; 
	
	// accumulate the gradients
	double GradHistP[][][];
	double GradHistW[][][]; 
	double GradHistBiasW[];
	
	// the softmax parameter
	public double gamma;	
	// time series data and the label 
	public Matrix labels;
	
	// the segments of the feature-rich data
	public double S[][][][];
	// the binary labels
	public double Y[][];
	
	public boolean avoidConvertingLabelsToBinary;
	public boolean initializePatternsRandomly;
	
	
	// the frequencies
	public double [][][] F;
	
	// the number of iterations
	public int maxIter;
	// the learning rate
	public double eta; 
	
	// the regularization parameters
	public double lambdaW, lambdaP;
	// the list of the nominal labels
	public List<Double> nominalLabels;	
	
	Random rand = new Random();
	
	List<Integer> instanceIdxs;
	List<Integer> rIdxs;
	
	// an instance of the unsupervised bag of words is needed to initialize
	LearnUnsupervisedBagofWordsMultiScale lubow;
	ClassificationPerformanceTensor cp;
	
	// globally minimum train loss and test mcr
	public double minTrainLoss = Double.MAX_VALUE;
	public double minTestMCR = Double.MAX_VALUE;
	public double minUnsupervisedTestMCR = Double.MAX_VALUE;
	
	// constructor
	public LearnSupervisedBagOfWordsMultiScale()
	{
		avoidConvertingLabelsToBinary = false;
		initializePatternsRandomly = false;
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		// set the labels to be binary 0 and 1, needed for the logistic loss
		if(!avoidConvertingLabelsToBinary)
			CreateOneVsAllTargets();		
		
	
		
		Logging.println("ITrain="+ITrain + ", ITest="+ITest  + ", Classes="+C, LogLevel.DEBUGGING_LOG);
		Logging.println("L=", LogLevel.DEBUGGING_LOG);  
		Logging.println(L, LogLevel.DEBUGGING_LOG); 
		Logging.println("K=", LogLevel.DEBUGGING_LOG); 
		Logging.println(K, LogLevel.DEBUGGING_LOG); 
		Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG); 
		Logging.println("lambdaW="+lambdaW + ", gamma="+ gamma, LogLevel.DEBUGGING_LOG); 
		
		
		
		// initialize the history of the patterns' gradients 
		GradHistP = new double[R][][];
		for(int r=0; r<R; r++)
		{
			GradHistP[r] = new double[K[r]][L[r]]; 
			
			for(int k=0; k<K[r]; k++)   
				for(int l=0; l<L[r]; l++)  
					GradHistP[r][k][l] = 0;
		}
		
		if(!initializePatternsRandomly)
		{
			
			lubow = new LearnUnsupervisedBagofWordsMultiScale();
			lubow.ITrain = ITrain;
			lubow.ITest = ITest;
			lubow.K = K;
			lubow.L = L;
			lubow.C = C;
			lubow.R = R; 
			lubow.eta = eta;
			lubow.maxEpochs = maxIter; 
			lubow.lambdaW = lambdaW;	 
			lubow.S = S;
			
			lubow.Y = Y;
			lubow.LearnUnsupervisedCodebook(); 
			
			P = lubow.P;
		}
		else // otherwise initialize patterns randomly
		{
			P = new double[R][][];
			
			for(int r=0; r<R; r++)
			{
				P[r] = new double[K[r]][L[r]]; 
				
				for(int k=0; k<K[r]; k++)   
					for(int l=0; l<L[r]; l++)  
						P[r][k][l] = 2*rand.nextDouble()-1; 
			}
		}
		
		
		// initialize the weights and the gradient history for classification weights
		GradHistBiasW = new double[C];
		biasW = new double[C];
		GradHistW = new double[C][R][]; 
		W = new double[C][R][];
		
		for(int c=0; c < C; c++)
		{
			biasW[c] = 2*rand.nextDouble()-1;
			GradHistBiasW[c] = 0;
			
			for(int r=0; r<R; r++) 
			{
				W[c][r] = new double[K[r]];
				GradHistW[c][r] = new double[K[r]];
				
				for(int k=0; k<K[r]; k++)
				{
					W[c][r][k] = 2*rand.nextDouble()-1;
					GradHistW[c][r][k] = 0;
				}
			}
		}
		
		
		
		// initialize the classification performance
		cp = new ClassificationPerformanceTensor();
		cp.ITrain=ITrain;
		cp.ITest=ITest;
		cp.C = C;
		cp.K = K;
		cp.R = R;
		
		// store all the instances indexes for
		instanceIdxs = new ArrayList<Integer>();
		for(int i = 0; i < ITrain; i++)
				instanceIdxs.add(i);
		
		// shuffle the order for a better convergence
		Collections.shuffle(instanceIdxs);
		
		// initialize the frequencies matrix
		F = new double[ITrain+ITest][R][];
		for(int i = 0; i < ITrain+ITest; i++) 
			for(int r=0; r<R; r++)
				F[i][r] = new double[K[r]];
		
		
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// create one-cs-all targets
	public void CreateOneVsAllTargets() 
	{
		C = nominalLabels.size(); 
		
		Y = new double[ITrain+ITest][C];
		
		// initialize the extended representation  
        for(int i = 0; i < ITrain+ITest; i++) 
        {
        	// firts set everything to zero
            for(int c = 0; c < C; c++)   
               	Y[i][c] = -1.0; 
            
            // then set the real label index to 1
            int indexLabel = nominalLabels.indexOf( labels.get(i, 0) );  
        	Y[i][indexLabel] = 1.0;  
        }  

	}  
	
	// compute the distances of the i-th instance segments 
	// to all the patterns
	public double[][][] ComputeDistances(int i)
	{ 
		// compute the similarity distances
		double[][][] D_i = new double[R][][];
		double err=0;  
		
		for(int r=0; r<R; r++)  
		{
			D_i[r] = new double[S[i][r].length][K[r]];
			
			for(int j=0; j < S[i][r].length; j++)   
			{
				for(int k=0; k < K[r]; k++)  
				{
					D_i[r][j][k]=0;   
					
					for(int l=0; l < L[r]; l++)  
					{
						err = P[r][k][l] - S[i][r][j][l];  
						D_i[r][j][k] += err*err;   
					} 
					
					D_i[r][j][k] = Math.exp( ( gamma/ (double)L[r] ) * D_i[r][j][k] );     
				}
			} 
		}
			
		return D_i;
	}
	
	// compute the frequencies of the i-th instances
	public void ComputeFrequencies(int i, double [][][] D_i)
	{
		// compute the frequencies 
		for(int r=0; r<R; r++) 
			for(int k=0; k<K[r]; k++)  
			{ 
				F[i][r][k] = 0;
				
				for(int j=0; j < S[i][r].length; j++)  
				{ 
					F[i][r][k] += D_i[r][j][k]; 
				} 
				 
				// normalize by the number of segments
				F[i][r][k] = F[i][r][k] / (double)S[i][r].length; 
			}
		
		
		
	}
	
	// compute the frequencies of the test instances
	public void ComputeInstancesFrequencies()
	{
		for(int i = 0; i < ITrain+ITest; i++)
			ComputeFrequencies(i, ComputeDistances(i));		
	}
	
	// compute the estimated target variable
	public double EstimateTarget(int i, int c)
	{
		double y_hat_ic = biasW[c];
		
		for(int r=0; r<R; r++)
			for(int k=0; k<K[r]; k++) 
				y_hat_ic += F[i][r][k]*W[c][r][k];
		
		return y_hat_ic;
	}
	
	
	public void LearnFSGD()
	{ 
		// reshufle the instance indices
		Collections.shuffle(instanceIdxs);
		
		// parallel implementation of the learning, one thread per instance
		// up to as much threads as JVM allows
		Parallel_1x0.ForEach(instanceIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer i)
		    {
				double Y_hat_ic = 0; 
				// the gradients
				double dL_dYic = 0, dYic_dWcrk = 0, dYic_dFirk = 0, dFik_dPrkl = 0, 
						dReg_dWrck = 0, dReg_dPrklTmp = 0, dReg_dPrkl = 0, dOic_dWcrk = 0, 
						dOic_dWc0 = 0, dOic_dPrkl = 0, dFik_dPrklTemp = 0; 
				
				double eps = 0.000000001; 
				
				// compute the distances of all the segments of the i-th instance
				// to all the patterns
				double [][][] D_i = ComputeDistances(i); 
				// compute the frequencies of the i-th instance to the patterns
				ComputeFrequencies(i, D_i);  
				
				// iterate through the classes
				for(int c = 0; c < C; c++) 
				{ 
					// compute the estimated target variable 
					Y_hat_ic = EstimateTarget(i, c); 
					
					// compute the partial derivative of the loss wrt estimated target  
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
					
					for(int r = 0; r < R; r++) 
					{ 
						dFik_dPrklTemp = (2.0*gamma)/(double) L[r];
						
						// update all the patterns and weights	
						for(int k = 0; k < K[r]; k++) 
						{ 
							dYic_dWcrk = F[i][r][k]; 
							dReg_dWrck = (2.0/(double)ITrain)*lambdaW*W[c][r][k]; 
							
							// compute the partial derivative of the objective with respect to 
							// the decomposed objective function 
							dOic_dWcrk = dL_dYic*dYic_dWcrk + dReg_dWrck; 
							
							// update the history of weights' gradients 
							GradHistW[c][r][k] += dOic_dWcrk*dOic_dWcrk; 
							// update the weight
							W[c][r][k] -= (eta/(eps+Math.sqrt(GradHistW[c][r][k]))) * dOic_dWcrk; 
							
							// the partial derivative of the estimated target wrt the frequency
							dYic_dFirk = W[c][r][k];  
							
							// compute the first part of the partial derivative of 
							// the regularization wrt pattern
							dReg_dPrklTmp = (2.0/(double)(ITrain*C))*lambdaP; 
							 
							// update every point of the pattern
							for(int l=0; l<L[r]; l++)
							{
								// compute the partial derivative of the frequency wrt the pattern
								dFik_dPrkl = 0; 
								for(int j=0; j< S[i][r].length; j++)
									dFik_dPrkl += D_i[r][j][k]*dFik_dPrklTemp*(P[r][k][l]-S[i][r][j][l]); 
								
								// divide frequency vs. pattern derivative by the number of segments
								// the normalization is also used when computing the frequencies from the distances 
								dFik_dPrkl = dFik_dPrkl / (double) S[i][r].length; 
								
								// compute the partial derivative of the regularization with 
								// respect to the pattern
								dReg_dPrkl = dReg_dPrklTmp*P[r][k][l]; 
								
								// compute the derivative of the objective wrt the pattern
								dOic_dPrkl = dL_dYic*dYic_dFirk*dFik_dPrkl + dReg_dPrkl;  
								
								// update the history of the patterns' gradients
								GradHistP[r][k][l] += dOic_dPrkl*dOic_dPrkl;  
								// update the pattern
								P[r][k][l] -= (eta/(eps+Math.sqrt(GradHistP[r][k][l]))) * dOic_dPrkl; 
							}						
						}
					}
				}			 	
		    }		
		});		
	}
	
	// optimize the objective function
	
	public double Learn()
	{
		int numRandomRestarts = 1;
		
		for(int run=0; run < numRandomRestarts; run++)
		{
			// initialize the data structures
			Initialize();
			
			List<Double> lossHistory = new ArrayList<Double>();
			lossHistory.add(Double.MIN_VALUE);
			
			int logDisplayFrequency = maxIter/3;
			
			// apply the stochastic gradient descent in a series of iterations
			for(int iter = 0; iter <= maxIter; iter++) 
			{ 
				// learn the latent matrices
				
				// measure the loss
				if( iter % logDisplayFrequency == 0)
				{
					// compute the predictors, i.e. the freqiencies 
					ComputeInstancesFrequencies();
					cp.ComputeClassificationAccuracy(F, Y, W, biasW);  
					
					lossHistory.add(cp.trainLoss);
					
					Logging.println("It=" + iter + ", gamma= "+gamma+", lossTrain=" + cp.trainLoss + ", lossTest="+ cp.testLoss  +
									", MCRTrain=" +cp.trainMCR + ", MCRTest=" + cp.testMCR, LogLevel.DEBUGGING_LOG);   
					
					// if divergence is detected start from the beggining 
					// at a lower learning rate
					if( Double.isNaN(cp.trainLoss) || cp.trainMCR == 1.0 )
					{
						iter = 0;
						eta /= 3;
						lossHistory.clear();
						
						Initialize();
						Logging.println("Divergence detected. Restarting at eta=" + eta, LogLevel.DEBUGGING_LOG);
					}
					
					if( lossHistory.size() > 100 ) 
						if( cp.trainLoss > lossHistory.get( lossHistory.size() - 2  )  )
							break;
				}
				
				// iterate a SGD step
				LearnFSGD(); 
			}
			
			if(cp.trainLoss < minTrainLoss)
			{
				minTrainLoss = cp.trainLoss; 
				minTestMCR = cp.testMCR;
				minUnsupervisedTestMCR = lubow.testMCR;
			}
			
		}
		
		return minTestMCR; 
	}
}
