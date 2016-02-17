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


public class LearnSupervisedBagOfWords 
{	 
	// number of training and testing instances
	public int ITrain, ITest;
	// length of a time-series 
	public int Q;
	// length of shapelet
	public int L;
	// number of latent patterns
	public int K;
	// number of classes
	public int C; 
	
	// patterns
	double P[][];	
	// classification weights
	double W[][];
	double biasW[]; 
	
	// accumulate the gradients
	double GradHistP[][];
	double GradHistW[][];
	double GradHistBiasW[];
	
	// the softmax parameter
	public double gamma;	
	// time series data and the label 
	public Matrix labels;
	
	// the segments of the feature-rich data
	public double S[][][];
	// the ninary labels
	public double Y[][];
	
	// the frequencies
	public double [][] F;
	
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
	LearnUnsupervisedBagofWords lubow;
	ClassificationPerformance cp;
	
	// globally minimum train loss and test mcr
	public double minTrainLoss = Double.MAX_VALUE;
	public double minTestMCR = Double.MAX_VALUE;
	
	// constructor
	public LearnSupervisedBagOfWords()
	{
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		// set the labels to be binary 0 and 1, needed for the logistic loss
		CreateOneVsAllTargets();		
		
		// set the total number of shapelets per scale as a rule of thumb 
		// to the logarithm of the total segments
		int totalSegments = 0;
		for(int i = 0; i < ITrain; i++)
				totalSegments+=S[i].length;
		
		// avoid K=0 
		if( K <= 0)	K = (int) Math.sqrt(totalSegments);   
		
		Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q + ", Classes="+C, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L="+ L, LogLevel.DEBUGGING_LOG);
		Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lambdaW="+lambdaW + ", gamma="+ gamma, LogLevel.DEBUGGING_LOG);
		Logging.println("totalSegments="+totalSegments, LogLevel.DEBUGGING_LOG);
		
		
		// initialize the shapelets (complete initialization during the clustering)
		P = new double[K][L];
		
		lubow = new LearnUnsupervisedBagofWords();
		lubow.ITrain = ITrain;
		lubow.ITest = ITest;
		lubow.K = K;
		lubow.L = L;
		lubow.C = C;
		lubow.eta = eta;
		lubow.maxEpochs = maxIter; 
		lubow.lambdaW = lambdaW;	 
		lubow.S = S;
		lubow.Y = Y;
		lubow.LearnUnsupervisedCodebook();
		
		P = lubow.P;
		/*
		W = lubow.W;
		biasW= lubow.biasW;
		GradHistW= lubow.GradHistW;
		GradHistBiasW= lubow.GradHistBiasW;
		*/
		
		// initialize the weights and the gradient history for classification weights
		GradHistBiasW = new double[C];
		biasW = new double[C];
		GradHistW = new double[C][K]; 
		W = new double[C][K];
		
		for(int c=0; c < C; c++)
		{
			biasW[c] = 2*rand.nextDouble()-1;
			GradHistBiasW[c] = 0;
			
			for(int k=0; k<K; k++)
			{
				W[c][k] = 2*rand.nextDouble()-1;
				GradHistW[c][k] = 0;
			}
		}
		
		GradHistP = new double[K][L];
		for(int k = 0; k < K; k++) 
			for(int l = 0; l < L; l++)
				GradHistP[k][l] = 0;
		
		// initialize the classification performance
		cp = new ClassificationPerformance();
		cp.C = C;
		cp.ITrain=ITrain;
		cp.ITest=ITest;
		cp.K = K;
		
		// store all the instances indexes for
		instanceIdxs = new ArrayList<Integer>();
		for(int i = 0; i < ITrain; i++)
				instanceIdxs.add(i);
		
		// shuffle the order for a better convergence
		Collections.shuffle(instanceIdxs);
		
		// initialize the frequencies matrix
		F = new double[ITrain+ITest][K];
		
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
	public double[][] ComputeDistances(int i)
	{ 
		// compute the similarity distances
		double[][] D_i = new double[S[i].length][K];
		
		for(int j = 0; j < S[i].length; j++)
			for(int k = 0; k < K; k++)
			{
				D_i[j][k]=0;
				double err=0; 
				
				for(int l=0; l < L; l++)
				{
					err = P[k][l] - S[i][j][l];
					D_i[j][k] += err*err; 
				}
				
				D_i[j][k] = Math.exp((gamma/(double)L)*D_i[j][k]);
			}
			
		return D_i;
	}
	
	// compute the frequencies of the i-th instances
	public void ComputeFrequencies(int i, double [][] D_i)
	{
		// compute the frequencies
		for(int k = 0; k < K; k++)
		{
			F[i][k] = 0;
			for(int j=0; j < S[i].length; j++)
			{
				F[i][k] += D_i[j][k];
			}
			
			// normalize by the number of segments
			F[i][k] = F[i][k] / (double)S[i].length;
		}
		
		
		
	}
	
	// compute the frequencies of the test instances
	public void ComputeInstancesFrequencies()
	{
		for(int i = 0; i < ITrain+ITest; i++)
		{
			double [][] D_i = ComputeDistances(i);
			ComputeFrequencies(i, D_i);
		}		
	}
	
	// compute the estimated target variable
	public double EstimateTarget(int i, int c)
	{
		double y_hat_ic = biasW[c];
		
		for(int k = 0; k < K; k++)
			y_hat_ic += F[i][k]*W[c][k];
		
		return y_hat_ic;
	}
	
	
	
	public void LearnFSGD()
	{ 
		// parallel implementation of the learning, one thread per instance
		// up to as much threads as JVM allows
		Parallel_1x0.ForEach(instanceIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer i)
		    {
				double Y_hat_ic = 0; 
				// the gradients
				double dL_dYic = 0, dYic_dWck = 0, dYic_dFik = 0, dFik_dPkl = 0, 
						dReg_dWck = 0, dReg_dPklTmp = 0, dReg_dPkl = 0, dOic_dWck = 0, 
						dOic_dWc0 = 0, dOic_dPkl = 0, dFik_dPklTemp = (2.0*gamma)/(double)L;
				
				double eps = 0.000000001;
				
				// compute the distances of all the segments of the i-th instance
				// to all the patterns
				double [][] D_i = ComputeDistances(i);
				// compute the frequencies of the i-th instance to the patterns
				ComputeFrequencies(i, D_i);  
				
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
						
						// the partial derivative of the estimated target wrt the frequency
						dYic_dFik = W[c][k];  
						
						// compute the first part of the partial derivative of 
						// the regularization wrt pattern
						dReg_dPklTmp = (2.0/(double)(ITrain*C))*lambdaP; 
						 
						// update every point of the pattern
						for(int l=0; l<L; l++) 
						{
							// compute the partial derivative of the frequency wrt the pattern
							dFik_dPkl = 0; 
							for(int j=0; j< S[i].length; j++)
								dFik_dPkl += D_i[j][k]*dFik_dPklTemp*(P[k][l]-S[i][j][l]);   
							
							// divide frequency vs. pattern derivative by the number of segments
							// the normalization is also used when computing the frequencies from the distances 
							dFik_dPkl = dFik_dPkl / (double) S[i].length; 
							
														// compute the partial derivative of the regularization with 
							// respect to the pattern
							dReg_dPkl = dReg_dPklTmp*P[k][l];
							
							// compute the derivative of the objective wrt the pattern
							dOic_dPkl = dL_dYic*dYic_dFik*dFik_dPkl + dReg_dPkl;  
							
							// update the history of the patterns' gradients
							GradHistP[k][l] += dOic_dPkl*dOic_dPkl; 
							// update the pattern
							P[k][l] -= (eta/(eps+Math.sqrt(GradHistP[k][l]))) * dOic_dPkl; 
						}						
					}		
				}			 	
		    }		
		});		
	}
	
	// optimize the objective function
	
	public double Learn()
	{
		int numRandomRestarts = 5;
		
		for(int run=0; run < numRandomRestarts; run++)
		{
			// initialize the data structures
			Initialize();
			
			List<Double> lossHistory = new ArrayList<Double>();
			lossHistory.add(Double.MIN_VALUE);
			
			int logDisplayFrequency = maxIter/10;
			
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
			}
			
		}
		
		return minTestMCR; 
	}
	
		

	
	// the main execution of the program
	public static void main(String [] args)
	{
		// in case ones wants to run it from an IDE like eclipse 
		// then the command line parameters can be set as
		if (args.length == 0) {
			//String dir = "E:\\Data\\classification\\timeseries\\",
			String dir = "/run/media/josif/E/Data/classification/timeseries/",
			ds = "MoteStrain";   

			String sp = File.separator;
		
			args = new String[] {  
				"trainSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
						+ ds + "_TRAIN",  
				"testSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
						+ ds + "_TEST",  
 				"lambdaW=0.0001", 
 				"lambdaP=0.0001", 
				"maxEpochs=1000", 
				"gamma=-10", 
				"delta=0.2",
				//"K=0.4", 
				"L=0.2",
				"eta=0.1" 
				}; 
		}			

		// values of hyperparameters
		double eta = -1, lambdaW = -1, lambdaP = -1, gamma = -1, L = -1, K = -1, delta = 0.5;
		int maxEpochs = -1, R = -1;
		String trainSetPath = "", testSetPath = "";
		
		// read and parse parameters
		for (String arg : args) {
			String[] argTokens = arg.split("=");
			
			if (argTokens[0].compareTo("eta") == 0) 
				eta = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("lambdaW") == 0)
				lambdaW = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("lambdaP") == 0)
				lambdaP = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("gamma") == 0)
				gamma = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("delta") == 0)
				delta = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("maxEpochs") == 0)
				maxEpochs = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("R") == 0)
				R = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("L") == 0)
				L = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("K") == 0)
				K = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("trainSet") == 0)
				trainSetPath = argTokens[1];
			else if (argTokens[0].compareTo("testSet") == 0)
				testSetPath = argTokens[1];
		}
		
		
		// set predefined parameters if none set
		if(R < 0) R = 4;
		if(L < 0) L = 0.15;
		if(eta < 0) eta = 0.01;
		if(gamma > 0) gamma = -1; 
		if(maxEpochs < 0) maxEpochs = 10000;
		
		long startTime = System.currentTimeMillis();
		 
		// load dataset
		DataSet trainSet = new DataSet();
		trainSet.LoadDataSetFile(new File(trainSetPath));
		DataSet testSet = new DataSet();
		testSet.LoadDataSetFile(new File(testSetPath));

		// normalize the data instance
		trainSet.NormalizeDatasetInstances();
		testSet.NormalizeDatasetInstances();
		
		// predictor variables T
		Matrix T = new Matrix();
        T.LoadDatasetFeatures(trainSet, false);
        T.LoadDatasetFeatures(testSet, true);
        // outcome variable O
        Matrix O = new Matrix();
        O.LoadDatasetLabels(trainSet, false);
        O.LoadDatasetLabels(testSet, true);

        LearnSupervisedBagOfWords lsBoW = new LearnSupervisedBagOfWords();   
        // initialize the sizes of data structures
        lsBoW.ITrain = trainSet.GetNumInstances();  
        lsBoW.ITest = testSet.GetNumInstances();
        lsBoW.Q = T.getDimColumns();
        
        lsBoW.labels = O;
        // set the learn rate and the number of iterations
        lsBoW.maxIter = maxEpochs;
        // set te number of patterns
        lsBoW.L = (int)(L*T.getDimColumns());
        lsBoW.K = (int)(K*T.getDimColumns());
        
        // create the representation
        
        int deltaPoints = (int) ( lsBoW.L * delta) + 1; 
        
        //System.out.println(deltaPoints);        
        
        // convert the time series dataset into partitioned segments
        TimeSeriesTransformations tst = new TimeSeriesTransformations(); 
        
        lsBoW.S = tst.NormalizationTransformations(T.cells, lsBoW.L, deltaPoints); 
        //lsBoW.S = tst.DerivativeTransformations(T.cells, lsBoW.L, deltaPoints); 
        //lsBoW.S = tst.NormalizedOriginalAndDerivativeTransformations(T.cells, lsBoW.L, deltaPoints);
        
        
        
        // set the regularization parameter
        lsBoW.lambdaW = lambdaW;
        lsBoW.lambdaP = lambdaP;
        
        lsBoW.eta = eta;  
        lsBoW.gamma = gamma;
        
        
        trainSet.ReadNominalTargets(); 
        lsBoW.nominalLabels =  new ArrayList<Double>(trainSet.nominalLabels); 
        
        // learn the model
        lsBoW.Learn();  
        
        // learn the local convolutions
        long endTime = System.currentTimeMillis(); 
        
		System.out.println( 
				String.valueOf(lsBoW.minTestMCR)  + " " + String.valueOf(lsBoW.minTrainLoss) + " " 
				+ "L=" + L	+ " " 
				+ "K=" + lsBoW.K + " "
				+ "lW=" + lambdaW + " "
				+ "gamma=" + gamma + " " 
				+ "eta=" + eta + " " 
				+ "maxEpochs="+ maxEpochs + " " 
				+ "time="+ (endTime-startTime) 
				); 
	}

}
