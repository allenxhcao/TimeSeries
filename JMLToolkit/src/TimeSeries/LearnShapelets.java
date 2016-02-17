package TimeSeries;

import info.monitorenter.gui.chart.ITrace2D;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

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


public class LearnShapelets 
{	 
	// number of training and testing instances
	public int ITrain, ITest;
	// length of a time-series 
	public int Q;
	// length of shapelet
	public int L;
	// number of latent patterns
	public int K;
	// number of segments
	public int J;
	// shapelets
	double Shapelets[][];
	// classification weights
	double W[];
	double biasW;
	
	// the softmax parameter
	public double alpha;
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y;
		
	// the number of iterations
	public int maxIter;
	// the learning rate
	public double eta; 
	
	// the regularization parameters
	public double lambdaW;
	public double lambdaS;
	
	
	// structures for storing the precomputed terms
	double D_i[][];
	double E_i[][]; 
	double M_i[];
	double Psi_i[];
	double sigY;

	Random rand = new Random();
	
	// constructor
	public LearnShapelets()
	{
		
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		// avoid K=0 
		if(K == 0) 
			K = 1;
		
		Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L="+L, LogLevel.DEBUGGING_LOG);
		Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lamdaW="+lambdaW + ", alpha="+ alpha, LogLevel.DEBUGGING_LOG);
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		for(int i = 0; i < ITrain+ITest; i++)
			if(Y.get(i) != 1.0) 
				Y.set(i, 0, 0.0); 
		
		// set the number of segments
		J = Q - L;
		
		// initialize shapelets
		InitializeShapeletsProbabilityDistance();
		//InitializeShapeletsRandomly();
		
		// initialize the terms for pre-computation
		D_i = new double[K][J];
		E_i = new double[K][J];
		M_i = new double[K];
		Psi_i = new double[K];
		
		// initialize the weights
		W = new double[K];
		for(int k = 0; k < K; k ++)
			W[k] = 2*rand.nextDouble()-1;
		biasW = 2*rand.nextDouble()-1;
		// learn the weights
		LearnLOnlyW();
		
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	public void InitializeShapeletsRandomly()
	{
		Shapelets = new double[K][L];
		for(int k=0; k < K; k++)
			for(int l=0; l < K; l++)
				Shapelets[k][l] = 4*rand.nextDouble() - 2;
	}
	
	// initialize the patterns from random segments
	public void InitializeShapeletsProbabilityDistance()
	{
		double [][] segments = new double[ITrain*J][L];
		for(int i= 0; i < ITrain; i++) 
			for(int j= 0; j < J; j++) 
				for(int l = 0; l < L; l++)
					segments[i*J + j][l] = T.get(i, j+l);
		
		// normalize segments
		for(int i= 0; i < ITrain; i++) 
			for(int j= 0; j < J; j++) 
				for(int l = 0; l < L; l++)
					segments[i*J + j] = StatisticalUtilities.Normalize(segments[i*J + j]);
		
		KMeans kmeans = new KMeans();
		Shapelets = kmeans.InitializeKMeansPP(segments, K);
	}
	
	// predict the label value vartheta_i
	public double Predict_i()
	{
		double Y_hat_i = biasW;

		for(int k = 0; k < K; k++)
			Y_hat_i += M_i[k] * W[k];
		
		return Y_hat_i;
	}
	
	// precompute terms
	public void PreCompute(int i)
	{
		// precompute terms
		for(int k = 0; k < K; k++)
		{
			for(int j = 0; j < J; j++)
			{
				// precompute D_i
				D_i[k][j] = 0;
				double err = 0;
				
				for(int l = 0; l < L; l++)
				{
					err = T.get(i, j+l)- Shapelets[k][l];
					D_i[k][j] += err*err; 
				}
				
				D_i[k][j] /= (double)L; 
				
				// precompute E_i
				E_i[k][j] = Math.exp(alpha * D_i[k][j]);
			}
			
			// precompute Psi_i 
			Psi_i[k] = 0; 
			for(int j = 0; j < J; j++)
				Psi_i[k] +=  Math.exp( alpha * D_i[k][j] );
			
			// precompute M_i 
			M_i[k] = 0;
			
			for(int j = 0; j < J; j++)
				M_i[k] += D_i[k][j]* E_i[k][j];
			
			M_i[k] /= Psi_i[k];
		}
		
		sigY = Sigmoid.Calculate( Predict_i() ); 
	}
	
	// compute the MCR on the test set
	private double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
			double label_i = Sigmoid.Calculate(Predict_i());
			
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
			PreCompute(i);
			double label_i = Sigmoid.Calculate(Predict_i()); 
			
			if( (Y.get(i) == 1 && label_i < 0.5) || (Y.get(i) == 0 && label_i >= 0.5) )
						numErrors++;
		}
		
		return (double)numErrors/(double)ITest;
	}
	
	// compute the accuracy loss of instance i according to the 
	// smooth hinge loss 
	public double AccuracyLoss(int i)
	{
		PreCompute(i);
		
		double Y_hat_i = Predict_i();
		double sig_y_i = Sigmoid.Calculate(Y_hat_i);
		
		return -Y.get(i)*Math.log( sig_y_i ) - (1-Y.get(i))*Math.log(1-sig_y_i); 
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
	
	public void LearnF()
	{
		double regWConst = ((double)2.0*lambdaW) / ((double) ITrain);
		double regSConst = ((double)2.0*lambdaW) / ((double) ITrain);
		double phi_ikj = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
			
			// learn shapelets
			for(int k = 0; k < K; k++)
			{
				
				W[k] -= eta*(-(Y.get(i) - sigY)*M_i[k] + regWConst*W[k]);
				
				for(int j = 0; j < J; j++)
				{
					phi_ikj = ( (2.0*E_i[k][j]) / (L*Psi_i[k]) ) 
								* (1 + alpha*(D_i[k][j] - M_i[k])); 
					
					for(int l = 0; l < L; l++)
					{
						Shapelets[k][l] -= eta*(-(Y.get(i) - sigY)
								*phi_ikj*(Shapelets[k][l] - T.get(i, j+l))*W[k] 
										+ regSConst*Shapelets[k][l]);    
					}
				}				
			}
			
			biasW -= eta*(-(Y.get(i) - sigY));
		}
	}
		
	
	public void LearnLOnlyW()
	{
		double regWConst = ((double)2.0*lambdaW) / ((double) ITrain);
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i); 
			
			// learn shapelets
			for(int k = 0; k < K; k++)
					W[k] -= eta*(-(Y.get(i) - sigY)*M_i[k] + regWConst*W[k]);
				
			biasW -= eta*(-(Y.get(i) - sigY));
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
			// learn the latent matrices
			LearnF();

			// measure the loss
			if( iter % 10 == 0)
			{
				double mcrTrain = GetMCRTrainSet();
				double mcrTest = GetMCRTestSet();
				double lossTrain = AccuracyLossTrainSet();
				double lossTest = AccuracyLossTestSet();
				
				lossHistory.add(lossTrain);
				
				Logging.println("It=" + iter + ", lossTrain="+ lossTrain + ", lossTest="+ lossTest  +
								", MCRTrain=" +mcrTrain + ", MCRTest=" +mcrTest, LogLevel.DEBUGGING_LOG);
				
				if( lossHistory.size() > 50 )
				{
					if( lossTrain > lossHistory.get( lossHistory.size() - 49  )  )
						break;
				}
			}

			
			
		}
		
		// print shapelets for debugging purposes
		//PrintShapelets();
		
		//Logging.print(M_i, System.out, LogLevel.DEBUGGING_LOG); 
		
		return GetMCRTestSet(); 
	}
	
	public void PrintShapelets()
	{
		System.out.print("Shapelets=[ "); 
		
		for(int k = 0; k < K; k++)
		{
			
			for(int l = 0; l < L; l++)
			{
				System.out.print(Shapelets[k][l] + " ");
			}
			
			if( k != K-1)
				System.out.println(";");
			else 
				System.out.println("]");
		}
		
		System.out.print("W=[ "); 
		
		for(int k = 0; k < K; k++)
		{
			System.out.print(W[k] + " ");
			
			if( k == K-1)
				System.out.println("]");
		}
	}
	

}
