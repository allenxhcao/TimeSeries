package FeatureRich;

import java.util.Random;

import javax.swing.text.Utilities;

import org.apache.commons.collections.functors.SwitchClosure;

import Clustering.KMeans;
import DataStructures.DataSet;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;

public class SupervisedCodebookLearning 
{
	// number of instances
	public int NTrain, NTest;
	// number of sub-instance patches
	public int K;
	// dimensionality of a patch representation
	public int D;
	// number of classes
	public int C;
	
	// the feature-rich data
	public double X[][][];
	// the labels
	public int L[];
	// the binary targets
	public double Y[][];
	
	// the number of codebook patterns
	public int Q;
	// the codebook patterns
	public double P[][];
	// the classification weights
	public double W[][];
	
	// regularization parameters
	public double lambdaW, lambdaF;
	
	public double gamma;
	
	
	// the learning rate
	public double eta;
	// the maximum number of iterations
	public int maxIters;
	
	// random number generator
	Random rand = new Random();
	double epsilon = 0.1;
	
	public enum LossTypes{LOGISTIC, HINGE, LEASTSQUARES};
	public LossTypes lossType;
	
	public enum SimilarityTypes{EUCLIDEAN, GAUSSIAN, SOFTMAX};
	public SimilarityTypes similarityType;
	
	// precomputed euclidean distances
	double eucDists[][][];
	
	// constructors
	public SupervisedCodebookLearning()
	{
		lambdaW = 0.001;
		eta=0.001;
		maxIters=5000;
	}
	
	// initialize from a dataset
	public void Initialize(DataSet dsTrain, DataSet dsTest)
	{
		
		// count the number of patches overlapping
		int patchIncrementOffset = D/5 + 1;
		
		//K = dsTrain.numFeatures-D+1;
		K = 0;
		for(int i=0; i< dsTrain.numFeatures - D + 1; i+= patchIncrementOffset)
			K++;
 
		String lossTypeStr = ""; 
		switch (lossType) 
		{
	    	case LOGISTIC:  lossTypeStr = "LOGISTIC"; break;
	    	case LEASTSQUARES:  lossTypeStr = "LEASTSQUARES"; break;
	    	case HINGE:  lossTypeStr = "HINGE"; break;
		}
		
		// print the data size and parameters
		System.out.println("NTrain="+NTrain + ", NTest="+NTest + ", K="+K  + 
				", D="+D+ ", Q="+Q+ ", C="+C + ", lossType="+ lossTypeStr);
		
		
		// initialize vectors
		X = new double[NTrain+NTest][K][D]; 
		L = new int[NTrain+NTest]; 
		
		
		// initialize segments
		for(int n=0; n < NTrain+NTest; n++) 
			for(int k=0; k< K; k++) 
				for(int t=0; t<D; t++)
				{
					if( n < NTrain )
						X[n][k][t] = dsTrain.instances.get(n).features.get(k*patchIncrementOffset + t).value;
					else
						X[n][k][t] = dsTest.instances.get(n-NTrain).features.get(k*patchIncrementOffset + t).value;
				}

		
		// initialize labels
		for(int n=0; n < NTrain+NTest; n++)
		{
			if( n < NTrain )
				L[n] = dsTrain.nominalLabels.indexOf( dsTrain.instances.get(n).target );
			else
				L[n] = dsTest.nominalLabels.indexOf( dsTest.instances.get(n-NTrain).target );

			//System.out.println( ds.instances.get(n).target + " " + L[n] );
		}
		
		// create the one-vs-all targets
		CreateOneVsAllTargets();
				
		// initialize patterns from the k-means
		InitializePatternsKMeans();
		
		// initialize the weights and then initially learn them from the k-means patterns' frequencies
		W = new double[C][Q];		
		for(int c=0; c<C; c++)
			for(int q=0; q<Q; q++)
				W[c][q] = 2*epsilon*rand.nextDouble()-epsilon;	
		
		// initialize the matrix of the euclidean distances
		eucDists = new double[NTrain+NTest][K][Q];
		for(int n=0; n<NTrain+NTest; n++) 
			for(int k=0; k<K; k++)
				for(int q=0; q<Q; q++)
					eucDists[n][k][q] = 0.0;
	}
	
	// precompute the euclidean distances
	public void PrecomputeEuclideanDistances()
	{
		for(int n=0; n<NTrain+NTest; n++) 
			for(int k=0; k<K; k++)
				for(int q=0; q<Q; q++)
					eucDists[n][k][q] = EuclideanDistance(n, k, q); 
	}
	
	// initialize the patterns from random segments
	public void InitializePatternsKMeans()
	{		
		P = new double[Q][D];
		
		Logging.println("Initialize patterns ... ", LogLevel.DEBUGGING_LOG);
		
		double [][] allPatches = new double[NTrain*K][D]; 
		
		for(int n= 0; n < NTrain; n++) 
			for(int k= 0; k < K; k++) 
				for(int d = 0; d < D; d++)
					allPatches[n*K + k][d] = X[n][k][d];

		// normalize segments
		for(int n= 0; n < NTrain; n++) 
			for(int k= 0; k < K; k++) 
				allPatches[n*K + k] = StatisticalUtilities.Normalize(allPatches[n*K + k]);
		
		
		KMeans kmeans = new KMeans();
		P = kmeans.InitializeKMeansPP(allPatches, Q, 10);  
		
		if( P == null)
		{
			System.out.println("P not set");
			System.exit(1);
		}
		else if( P.length != Q )
		{
			System.err.println("Mismatch between the found number of centroids and requested ones. " + 
					Q + " to " + P.length);
			System.exit(1); 
		}
	}

	
	// create one-cs-all targets
	public void CreateOneVsAllTargets() 
	{		
		Y = new double[NTrain+NTest][C];
		
		// initialize the extended representation  
        for(int n = 0; n < NTrain+NTest; n++) 
        {
        	// first set everything to the negative class
            for(int c = 0; c < C; c++)  
            {
            	if( lossType == LossTypes.LOGISTIC )
            		Y[n][c] = 0.0;
            	else if(lossType == LossTypes.LEASTSQUARES)
            		Y[n][c] = -1.0;
            	else if(lossType == LossTypes.HINGE)
            		Y[n][c] = -1.0;
            }
            
            // then set the index of the label to 1 
            Y[n][ L[n] ] = 1.0;
        } 
        
	} 
	
	// predict the target value of the c-th class for the n-th instance
	public double Predict(int n, int c)
	{
		double Y_hat_nc = 0;

		for(int q = 0; q < Q; q++)
			Y_hat_nc += W[c][q] * Frequency(n, q);
		
		return Y_hat_nc;
	}
	
	// the frequency of the q-th pattern over the segments of the n-th instance
	public double Frequency(int n, int q)
	{
		double F_nq = 0;
		
		for(int k=0; k < K; k++) 
		{
			if( similarityType == SimilarityTypes.EUCLIDEAN )
				//F_nq += EuclideanDistance(n, k, q);
				F_nq += eucDists[n][k][q];
			else if( similarityType == SimilarityTypes.GAUSSIAN)
				//F_nq += Math.exp( gamma * EuclideanDistance(n, k, q) );
				F_nq += Math.exp( gamma * eucDists[n][k][q]);
		}
		
		return F_nq / ( (double) (K*D) );   
	} 
		
	
	// compute the euclidean distance between the k-th patch of the n-th instance 
	// and the q-th pattern
	public double EuclideanDistance(int n, int k, int q)
	{
		double err = 0, dist = 0;
		
		for (int d=0; d<D; d++) 
		{ 
			//System.out.println(n + ", q=" + q + ", k=" + k + ", d=" + d); 
			
			err = X[n][k][d] - P[q][d]; 
			dist += err*err; 
		}
		
		return dist;
	}
	
	// compute the accuracy loss of instance i according to the 
	// different types of losses
	public double Loss(int n, int c)
	{
		double loss = 0;
		
		double hat_Y_nc = Predict(n, c);
		
		if( lossType == LossTypes.LOGISTIC )
		{
			double sig_y_nc = Sigmoid.Calculate( hat_Y_nc );
			loss = -Y[n][c]*Math.log( sig_y_nc ) - (1-Y[n][c])*Math.log(1-sig_y_nc);
		}
		else if( lossType == LossTypes.LEASTSQUARES)
		{
			double error = Y[n][c] - hat_Y_nc;
			loss = error*error;
		}
		else if( lossType == LossTypes.HINGE)
		{
			double z = Y[n][c]*hat_Y_nc;
			
			if( z <= 0 )
				loss = 0.5 - z;
			else if(z > 0 && z < 1 )
			{
				double error = 1 - z;
				loss = 0.5*error*error;
			}
			else
				loss = 0;
		}
		
		return loss;
	}
	
	// derivative of the loss with respect to Y_nc 
	public double DerivativeLossY(int n, int c)
	{
		double dL_ncY_nc = 0;
		
		double hat_Y_nc = Predict(n, c);
		
		if( lossType == LossTypes.LOGISTIC )
			dL_ncY_nc = -( Y[n][c] - Sigmoid.Calculate( hat_Y_nc ));
		else if( lossType == LossTypes.LEASTSQUARES)
			dL_ncY_nc = -2*(Y[n][c] - hat_Y_nc);
		else if( lossType == LossTypes.HINGE)
		{
			double z = Y[n][c]*hat_Y_nc;
			
			if( z <= 0 )
				dL_ncY_nc = -Y[n][c];
			else if(z > 0 && z < 1 )
				dL_ncY_nc = (z-1)*Y[n][c];
			else
				dL_ncY_nc = 0;
		}
	
		
		return dL_ncY_nc;
	}
	
	// compute the accuracy loss of the train set
	public double AccuracyLossTrainSet()
	{
		double accuracyLoss = 0;
		
		for(int n = 0; n < NTrain; n++)
			for(int c = 0; c < C; c++)
				accuracyLoss += Loss(n, c);
		
		return accuracyLoss / ((double) NTrain*C);
	}
	// compute the accuracy loss of the train set
	public double AccuracyLossTestSet()
	{
		double accuracyLoss = 0;
		
		for(int n = NTrain; n < NTrain+NTest; n++)  
			for(int c = 0; c < C; c++) 
				accuracyLoss += Loss(n, c); 
		
		return accuracyLoss  / ((double) NTest*C);   
	}
	
	// the MCR calculations
	// compute the MCR on the test set
	private double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int n = 0; n < NTrain; n++)
		{
			double max_Y_nc = Double.MIN_VALUE;
			int label_n = -1; 
			
			for(int c = 0; c < C; c++)
			{
				double hat_Y_nc = Predict(n, c);
				
				// the logistic loss requires to take to sigmoid of the 
				// estimated Y
				if( lossType == LossTypes.LOGISTIC )
					hat_Y_nc = Sigmoid.Calculate( hat_Y_nc ); 
				
				if(hat_Y_nc > max_Y_nc)
				{
					max_Y_nc = hat_Y_nc; 
					label_n = (int)Math.ceil(c);
				}
			}
			
			if( L[n] != label_n ) 
				numErrors++;
		}
		
		return (double)numErrors/(double)NTrain;
	}
		private double GetMCRTestSet() 
		{
			int numErrors = 0;
			
			for(int n = NTrain; n < NTrain+NTest; n++)
			{
				double max_Y_nc = Double.MIN_VALUE;
				int label_n = -1; 
				
				for(int c = 0; c < C; c++)
				{
					double hat_Y_nc = Predict(n, c);
					
					// the logistic loss requires to take to sigmoid of the 
					// estimated Y
					if( lossType == LossTypes.LOGISTIC )
						hat_Y_nc = Sigmoid.Calculate( hat_Y_nc );  
					
					if(hat_Y_nc > max_Y_nc)
					{
						max_Y_nc = hat_Y_nc; 
						label_n = (int)Math.ceil(c);
					}
				}
				
				if( L[n] != label_n ) 
					numErrors++;
			}
			
			return (double)numErrors/(double)NTest;
		}
		
	// the learning algorithm
	public void Learn()
	{
		double regConstW = (2*lambdaW)/(NTrain*C);
		
		for(int iter=0; iter<maxIters; iter++)
		{
			// precompute the euclidean distances
			PrecomputeEuclideanDistances();
			
			// minimize the error across the training instances
			for(int n=0; n<NTrain; n++)
			{
				for(int c=0; c<C; c++)
				{
					double dL_ncY_nc = DerivativeLossY(n, c);
					
					// iterate through all the latent patterns
					for(int q=0; q<Q; q++)
					{
						W[c][q] = W[c][q] - eta*(dL_ncY_nc*Frequency(n, q) + regConstW*W[c][q]);
					
						for(int d=0; d<D; d++)
							P[q][d] = P[q][d] - eta*dL_ncY_nc*W[c][q]*ComputeGradientFwrtP(n, q, d);  
					}
				}
			}
			
			if(iter % 50 == 0)
			{
				double lossTrain = AccuracyLossTrainSet();
				double lossTest = AccuracyLossTestSet();
				double mcrTrain = GetMCRTrainSet();
				double mcrTest = GetMCRTestSet();
				
				System.out.println("Iter="+iter+ ", lossTrain="+lossTrain + ", lossTest="+lossTest + 
						", mcrTrain="+mcrTrain + ", mcrTest="+mcrTest ); 
				
			}
		}
	}
	
	// compute the gradient ( d F_nq / d P_qd )
	public double ComputeGradientFwrtP(int n, int q, int d)
	{
		double dFnqPQd = 0;
		
		if( similarityType == SimilarityTypes.EUCLIDEAN)
		{
			for(int k = 0; k< K; k++) 
				dFnqPQd += (X[n][k][d]-P[q][d]);
			
			 dFnqPQd = (-2.0 / ( (double)(K*D)) ) * dFnqPQd;
		}
		else if( similarityType == SimilarityTypes.GAUSSIAN )
		{
			// precompute the euclidean distances
			for(int k = 0; k< K; k++) 
				dFnqPQd += (X[n][k][d]-P[q][d])* Math.exp( gamma * eucDists[n][k][q]); //EuclideanDistance(n, k, q) );
			
			dFnqPQd = ( (-2.0*gamma) / ( (double)(K*D)) ) * dFnqPQd;			
		}
		
		return dFnqPQd;
	}
	
	
}
