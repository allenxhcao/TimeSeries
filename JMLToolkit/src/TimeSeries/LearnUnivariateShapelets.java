package TimeSeries;

import info.monitorenter.gui.chart.ITrace2D;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.logging.Logger.Level;
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
import DataStructures.Pair;
import FeatureRich.SupervisedCodebookLearning.LossTypes;


public class LearnUnivariateShapelets 
{	 
	// number of training and testing instances
	public int ITrain, ITest;
	// length of a time-series 
	public int Q;
	// length of shapelet
	public int L[];
	// number of latent patterns
	public int K;
	// number of classes
	public int C;
	// shapelets
	double Shapelets[][];
	// classification weights
	double W[][];
	double biasW[];
	
	// the softmax parameter
	public double alpha;
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y, Y_b;
		
	// the number of iterations
	public int maxIter;
	// the learning rate
	public double eta; 
	
	// the regularization parameters
	public double lambdaW, lambdaS;
	
	public List<Double> nominalLabels;
	

	public enum LossTypes{HINGE,LOGISTIC, LEASTSQUARES};
	public LossTypes lossType;
	
	
	
	// structures for storing the precomputed terms
	// the precomputed distances
	double D[][][];
	// the precomputed e^D
	double E[][][]; 
	// the precomputed minimum distances
	double M[][];
	// the precomputed sum of phi-s 
	double Psi[][];
	// the precomputed derivative of the loss with respect to y_hat for point i,c
	double phi[][];
	
	List<Integer> idxPairs;

	Random rand = new Random();
	
	// constructor
	public LearnUnivariateShapelets()
	{
		lossType = LossTypes.LOGISTIC;
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		// avoid K=0 
		if(K == 0) 
			K = 1;
		
		Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lambdaW="+lambdaW + ", alpha="+ alpha, LogLevel.DEBUGGING_LOG);
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		CreateOneVsAllTargets();
		
		Random rand = new Random();
		
		L = new int[K];
		for(int k=0; k < K; k++)
			L[k] = rand.nextInt(Q);
		
		
		Logging.println("Classes="+C, LogLevel.DEBUGGING_LOG);
		
		// initialize shapelets
		InitializeShapeletsRandomly();
		
		// initialize weights
		W = new double[C][K];
		biasW = new double[C];
		
		
		// initialize the terms for pre-computation
		D = new double[ITrain+ITest][K][];
		E = new double[ITrain+ITest][K][];
		
		for(int i = 0; i < ITrain+ITest; i++)
				for(int k = 0; k < K; k++)
				{
					D[i][k] = new double[Q-L[k]+1];
					E[i][k] = new double[Q-L[k]+1];
				}
		
		M = new double[ITrain+ITest][K];
		Psi = new double[ITrain+ITest][K];		 
		phi = new double[ITrain+ITest][C];
		
		
		// store all the index and class indexes
		idxPairs = new ArrayList<Integer>();
		for(int i = 0; i < ITrain; i++)
		{
				idxPairs.add(i);
				
				PreCompute(i); 
		}
		
	
		// shuffle the order
		Collections.shuffle(idxPairs);
		
				
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// create one-cs-all targets
	public void CreateOneVsAllTargets() 
	{
		C = nominalLabels.size(); 
		
		Y_b = new Matrix(ITrain+ITest, C);
		
		// initialize the extended representation  
        for(int i = 0; i < ITrain+ITest; i++) 
        {
        	// firts set everything to -1
            for(int c = 0; c < C; c++)  
            	if( lossType == LossTypes.HINGE || lossType == LossTypes.LEASTSQUARES )
            		Y_b.set(i, c, -1);
            	else if ( lossType == LossTypes.LOGISTIC )
            		Y_b.set(i, c, 0);
            
            // then set the real label index to 1
            int indexLabel = nominalLabels.indexOf( Y.get(i,0) ); 
            Y_b.set(i, indexLabel, 1.0); 
        }

	} 
	
	
	public void InitializeShapeletsRandomly()
	{		
		Shapelets = new double[K][];
		
		for(int k=0; k < K; k++)
		{
			Shapelets[k] = new double[L[k]];
			
			int i = rand.nextInt(ITrain);
			int j = rand.nextInt(Q-L[k]+1); 
			
			for(int l=0; l < L[k]; l++)
			{
				Shapelets[k][l] = T.get(i, j+l);  
			}
		}
	}
	
	
	// predict the label value vartheta_i
	public double Predict(int i, int c)
	{
		double Y_hat_ic = biasW[c];

		for(int k = 0; k < K; k++)
			Y_hat_ic += M[i][k] * W[c][k];
		
		return Y_hat_ic;
	}
	
	// precompute terms
	public void PreCompute(int i)
	{
		// precompute terms
		for(int k = 0; k < K; k++)
		{
			for(int j = 0; j < Q-L[k]+1; j++)
			{
				// precompute D_i
				D[i][k][j] = 0;
				double err = 0;
				
				for(int l = 0; l < L[k]; l++)
				{
					err = T.get(i, j+l)- Shapelets[k][l];
					D[i][k][j] += err*err; 
				}
				
				D[i][k][j] /= (double)L[k]; 
				
				// precompute E
				E[i][k][j] = Math.exp(alpha * D[i][k][j]); 
			}
			
			// precompute Psi_i 
			Psi[i][k] = 0; 
			for(int j = 0; j < Q-L[k]+1; j++) 
				Psi[i][k] +=  Math.exp( alpha * D[i][k][j] );
			
			// precompute M_i 
			M[i][k] = (1.0/alpha) * Math.log( Psi[i][k] ); 
		}
	
		
		
		for(int c = 0; c < C; c++)
			phi[i][c] = DerivativeLossY(i, c);
		
	}
	
	// compute the MCR on the test set
	private double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
			
			double max_Y_hat_ic = Double.MIN_VALUE;
			int label_i = -1; 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = Predict(i, c); 
				

				if(lossType == LossTypes.LOGISTIC)
					Y_hat_ic = Sigmoid.Calculate(Y_hat_ic);
				
				if(Y_hat_ic > max_Y_hat_ic)
				{
					max_Y_hat_ic = Y_hat_ic; 
					label_i = (int)Math.ceil(c);
				}
			}
			
			if( nominalLabels.indexOf(Y.get(i)) != label_i ) 
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
			PreCompute(i);
			
			double max_Y_hat_ic = Double.MIN_VALUE;
			int label_i = -1; 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = Predict(i, c);
				

				if(lossType == LossTypes.LOGISTIC)
					Y_hat_ic = Sigmoid.Calculate(Y_hat_ic);
				
				if(Y_hat_ic > max_Y_hat_ic)
				{
					max_Y_hat_ic = Y_hat_ic; 
					label_i = (int)Math.ceil(c);
				}
			}
			
			if( nominalLabels.indexOf(Y.get(i)) != label_i ) 
				numErrors++;
		}
		
		return (double)numErrors/(double)ITest; 
	}
	
	// compute the accuracy loss of instance i according to the 
		// smooth hinge loss 
		public double AccuracyLoss(int i, int c)
		{
			double loss = 0;
			
			if( lossType == LossTypes.HINGE)
			{
				double z = Y_b.get(i, c)*Predict(i, c); 
				
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
			else if(lossType == LossTypes.LOGISTIC)
			{
				double sig_y_ic = Sigmoid.Calculate(Predict(i,c));
				loss = - Y_b.get(i, c)*Math.log( sig_y_ic ) - (1-Y_b.get(i, c))*Math.log(1-sig_y_ic); 
			}
			else if( lossType == LossTypes.LEASTSQUARES )
			{
				double error = Y_b.get(i, c) - Predict(i, c); 
				loss = error*error;
			}
			
			return loss;
		}
	
	// derivative of the loss with respect to Y_nc 
	public double DerivativeLossY(int i, int c)
	{
		double dLdY = 0;
		
		if( lossType == LossTypes.HINGE)
		{
			double z = Y_b.get(i, c)*Predict(i, c);
			
			if( z <= 0 )
				dLdY = -Y_b.get(i, c);
			else if(z > 0 && z < 1 )
				dLdY = (z-1)*Y_b.get(i, c);
			else
				dLdY = 0;
		}
		else if(lossType == LossTypes.LOGISTIC)
		{
			dLdY = -(Y_b.get(i, c) - Sigmoid.Calculate(Predict(i,c)) ); 
		} 
		else if( lossType == LossTypes.LEASTSQUARES )
		{
			dLdY = -2*(Y_b.get(i, c) - Predict(i, c));
		}
		
		return dLdY;
	}
	
	// compute the accuracy loss of the train set
	public double AccuracyLossTrainSet()
	{
		double accuracyLoss = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
		
			for(int c = 0; c < C; c++)
				accuracyLoss += AccuracyLoss(i, c);
		}
		
		return accuracyLoss;
	}
	// compute the accuracy loss of the train set
	public double AccuracyLossTestSet()
	{
		double accuracyLoss = 0;
		
		for(int i = ITrain; i < ITrain+ITest; i++) 
		{
			PreCompute(i);
			
			for(int c = 0; c < C; c++) 
				accuracyLoss += AccuracyLoss(i, c); 
		}
		return accuracyLoss;
	}
	
	public void LearnFOnlyW()
	{
		
		// run in parallel
		Parallel_1x0.ForEach(idxPairs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer i)
		    {
				double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
				
		        PreCompute(i);
				
				for(int c = 0; c < C; c++)
				{
					for(int k = 0; k < K; k++)
					{
						W[c][k] -= eta*(phi[i][c]*M[i][k] + regWConst*W[c][k]);
						
						for(int l = 0; l < L[k]; l++)
						{
							double gradM_ik_S_kln =  (2.0) / (L[k] * Psi[i][k]) ;
							
							double sum_j = 0;
							for(int j = 0; j < Q-L[k]+1; j++)
								sum_j = E[i][k][j]*(Shapelets[k][l] - T.get(i, j+l));
							
							gradM_ik_S_kln *= sum_j;  
							
							Shapelets[k][l] -= eta*(phi[i][c]*W[c][k]*gradM_ik_S_kln);
						}				
					}
	
					biasW[c] -= eta*phi[i][c];  
				}			 	
		    }
		});
		
		
		
		
	}
	

	public void LearnF()
	{
		
		// run in parallel
		Parallel_1x0.ForEach(idxPairs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer i)
		    {
				double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
				
		        for(int c = 0; c < C; c++)
				{
		        	phi[i][c] = DerivativeLossY(i, c); 
		        			
					for(int k = 0; k < K; k++)
					{
						W[c][k] -= eta*(phi[i][c]*M[i][k] + regWConst*W[c][k]);
									
					}
	
					biasW[c] -= eta*phi[i][c];  
				}			 	
		    }
		});
	}
		

	

	// optimize the objective function
	public double Learn()
	{
		// initialize the data structures
		Initialize();
		for(int iter = 0; iter < 1000; iter++)
			LearnFOnlyW();
		
		List<Double> lossHistory = new ArrayList<Double>();
		lossHistory.add(Double.MIN_VALUE);
		
		// apply the stochastic gradient descent in a series of iterations
		for(int iter = 0; iter < maxIter; iter++)
		{
			// learn the shapelets and the weights
			LearnF();
			
			// measure the loss
			if( iter % 500 == 0)
			{
				double mcrTrain = GetMCRTrainSet();
				double mcrTest = GetMCRTestSet();
				
				double lossTrain = AccuracyLossTrainSet();
				double lossTest = AccuracyLossTestSet();
				
				lossHistory.add(lossTrain);
				
				Logging.println("It=" + iter + ", lossTrain="+ lossTrain + ", lossTest="+ lossTest  +
								", MCRTrain=" +mcrTrain + ", MCRTest=" +mcrTest , LogLevel.DEBUGGING_LOG);
				
				if( Double.isNaN(lossTrain) )
				{
					iter = 0;
					eta /= 3;
					
					Initialize();
					
					Logging.println("Divergence detected. Restarting at eta=" + eta, LogLevel.DEBUGGING_LOG);
				}
				else
				{
					//eta *= 0.9;
				}
				
			}
			
						
						

						
		}
		
		// print shapelets for debugging purposes
		//PrintShapelets();
		
		//Logging.print(M_i, System.out, LogLevel.DEBUGGING_LOG); 
		
		return GetMCRTestSet(); 
	}
	
	
//
//	public void PrintShapeletsWeights()
//	{
//		for(int r = 0; r < R; r++)
//		{
//			for(int k = 0; k < K; k++)
//			{
//				System.out.print("Shapelets("+r+","+k+")= [ ");
//				
//				for(int l = 0; l < L[r]; l++)
//				{
//					System.out.print(Shapelets[r][k][l] + " ");
//				}
//				
//				System.out.println("]");
//			}
//		}
//		
//		for(int c = 0; c < C; c++)
//		{
//			for(int r = 0; r < R; r++)
//			{
//				System.out.print("W("+c+","+r+")= [ ");
//				
//				for(int k = 0; k < K; k++)
//					System.out.print(W[c][r][k] + " ");
//				
//				System.out.print(biasW[c] + " ");
//				
//				System.out.println("]");
//			}
//		}
//	}
//	
//	
//	public void PrintProjectedData()
//	{
//		int r = 0, c = 0;
//		
//		System.out.print("Data= [ ");
//		
//		for(int i = 0; i < ITrain; i++)
//		{
//			PreCompute(i); 
//			
//			System.out.print(Y_b.get(i, c) + " "); 
//			
//			for(int k = 0; k < K; k++)
//			{
//				System.out.print(M[i][r][k] + " ");
//			}
//			
//			System.out.println(";");
//		}
//		
//		System.out.println("];");
//	}
//	
	

}
