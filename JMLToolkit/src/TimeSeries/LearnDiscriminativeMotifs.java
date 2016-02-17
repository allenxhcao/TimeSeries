package TimeSeries;

import info.monitorenter.gui.chart.ITrace2D;

import java.io.File;
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


public class LearnDiscriminativeMotifs 
{	 
	// number of training and testing instances
	public int ITrain, ITest;
	// length of a time-series 
	public int Q;
	// length of shapelet
	public int L[];
	public int L_min;
	// number of latent patterns
	public int K;
	// scales of the shapelet length
	public int R; 
	// number of classes
	public int C;
	// number of segments
	public int J[];
	// shapelets
	double Motifs[][][];
	// classification weights
	double W[][][];
	double biasW[];
	
	
	// the softmax parameter
	public double gamma;
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y, Y_b;
		
	// the number of iterations
	public int maxIter;
	// the learning rate
	public double eta; 
	
	public int kMeansIter;
	
	// the regularization parameters
	public double lambdaW;
	
	public List<Double> nominalLabels;
	
	// structures for storing the precomputed terms
	double E_i[][][]; 
	double F_i[][];
	double phi[]; 

	Random rand = new Random();
	
	// constructor
	public LearnDiscriminativeMotifs()
	{
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		// avoid K=0 
		if(K == 0) 
			K = 1;
		
		Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L_min="+ L_min + ", R="+R, LogLevel.DEBUGGING_LOG);
		Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lamdaW="+lambdaW + ", gamma="+ gamma, LogLevel.DEBUGGING_LOG);
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		CreateOneVsAllTargets();
		
		Logging.println("Classes="+C, LogLevel.DEBUGGING_LOG);
		
		// initialize shapelets
		InitializeMotifsKMeans();		
		
		// initialize the terms for pre-computation
		E_i = new double[R][K][];
		for(int r = 0; r < R; r++)
			for(int k = 0; k < K; k++)
				E_i[r][k] = new double[J[r]];
		
		F_i = new double[R][K];
		phi = new double[C];
		
		for(int i = 0; i < ITrain+ITest; i++)
			PreCompute(i); 
		
		LearnOOnlyW();
		
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
        	// firts set everything to zero
            for(int c = 0; c < C; c++)  
                    Y_b.set(i, c, 0);
            
            // then set the real label index to 1
            int indexLabel = nominalLabels.indexOf( Y.get(i, 0) ); 
            Y_b.set(i, indexLabel, 1.0); 
        } 

	} 
	
	// initialize the patterns from random segments
	public void InitializeMotifsKMeans()
	{		
		Motifs = new double[R][][];
		J = new int[R]; 
		L = new int[R];
		
		for(int r=0; r < R; r++)
		{
			L[r] = (r+1)*L_min;
			J[r] = Q - L[r];
			
			Logging.println("Initialize Shapelets: r="+r+", J[r]="+J[r]+", L[r]="+L[r], LogLevel.DEBUGGING_LOG);
			
			double [][] segmentsR = new double[(ITrain + ITest)*J[r]][L[r]];
			
			for(int i= 0; i < ITrain; i++) 
				for(int j= 0; j < J[r]; j++) 
					for(int l = 0; l < L[r]; l++)
						segmentsR[i*J[r] + j][l] = T.get(i, j+l);

			// normalize segments
			for(int i= 0; i < ITrain; i++) 
				for(int j= 0; j < J[r]; j++) 
					for(int l = 0; l < L[r]; l++)
						segmentsR[i*J[r] + j] = StatisticalUtilities.Normalize(segmentsR[i*J[r] + j]);
			
			
			KMeans kmeans = new KMeans();
			Motifs[r] = kmeans.InitializeKMeansPP(segmentsR, K, 100); 
			
			if( Motifs[r] == null)
				System.out.println("P not set");
		}
	}
	
	
	
	// predict the label value phi
	public double Predict(int c)
	{
		double Y_hat_ic = biasW[c];

		for(int r = 0; r < R; r++)
			for(int k = 0; k < K; k++)
			Y_hat_ic += F_i[r][k] * W[c][r][k];
		
		return Y_hat_ic;
	}
	
	
	// precompute terms
	public void PreCompute(int i)
	{
		// precompute terms
		for(int r = 0; r < R; r++)
		{
			for(int k = 0; k < K; k++)
			{
				for(int j = 0; j < J[r]; j++)
					E_i[r][k][j] = Math.exp(-gamma * GetSegmentDist(r, i, k, j)); 
				
				// precompute F_i 
				F_i[r][k] = 0;				
				for(int j = 0; j < J[r]; j++)
					F_i[r][k] += E_i[r][k][j];
				
				F_i[r][k] /= (double) J[r];
			}
		}
		
		for(int c = 0; c < C; c++)
			phi[c] = -(Y_b.get(i,c) - Sigmoid.Calculate( Predict(c) )); 
	}
	
	// the distance between the k-th motif at scale r and the j-th segment of the i-th series
	public double GetSegmentDist(int r, int i, int k, int j)
	{
		double err = 0, D_rikj = 0;
		
		for(int l = 0; l < L[r]; l++)
		{
			err = T.get(i, j+l)- Motifs[r][k][l];
			D_rikj += err*err; 
		}
		
		D_rikj /= (double)L[r];  
		
		return D_rikj;
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
				double Y_hat_ic = Sigmoid.Calculate( Predict(c) );
				
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
				double Y_hat_ic = Sigmoid.Calculate( Predict(c) );
				
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
		double Y_hat_ic = Predict(c);
		double sig_y_ic = Sigmoid.Calculate(Y_hat_ic);
		
		return -Y_b.get(i,c)*Math.log( sig_y_ic ) - (1-Y_b.get(i, c))*Math.log(1-sig_y_ic); 
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
	
	public void LearnO()
	{
		double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
		double temp = 0;
		double grad_rkl = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
			
			for(int c = 0; c < C; c++)
			{
			    for(int r = 0; r < R; r++)
				{
					temp = (2*gamma) / L[r];
					
					for(int k = 0; k < K; k++)
					{
						W[c][r][k] -= eta*(phi[c]*F_i[r][k] + regWConst*W[c][r][k]); 
						
						for(int l = 0; l < L[r]; l++)
						{
							grad_rkl = 0;
							
							for(int j = 0; j < J[r]; j++) 
								grad_rkl += E_i[r][k][j]*( Motifs[r][k][l] - T.get(i, j+l));
							
							grad_rkl *= phi[c]*W[c][r][k]*temp; 
							
							Motifs[r][k][l] -= eta*grad_rkl; 
						}									
					}
				}

				biasW[c] -= eta*phi[c]; 
			}			 	
		}
	}
	
	
	public void LearnOOnlyW()
	{
		W = new double[C][R][K];
		biasW = new double[C];
		
		for(int i = 0; i < ITrain; i++)
		{
			for(int c = 0; c < C; c++)
			{
				for(int r = 0; r < R; r++)
					for(int k = 0; k < K; k++)
						W[c][r][k] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON; 
					
				biasW[c] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON; 				
			}
		}
		
		
		double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
		
		for(int epochs = 0; epochs < maxIter/10; epochs++)
		{
			for(int i = 0; i < ITrain; i++)
			{
				PreCompute(i);
				
				for(int c = 0; c < C; c++)
				{
					for(int r = 0; r < R; r++)
					{
						for(int k = 0; k < K; k++)
						{
							W[c][r][k] -= eta*(phi[c]*F_i[r][k] + regWConst*W[c][r][k]);
						}
					}
					
					biasW[c] -= eta*phi[c]; 				
				}
			}
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
			LearnO();
			
			//LearnOWithoutPreComputation();  

			// measure the loss
			if( iter % 2 == 0) 
			{ 
				double mcrTrain = GetMCRTrainSet(); 
				double mcrTest = GetMCRTestSet(); 
				
				double lossTrain = AccuracyLossTrainSet(); 
				double lossTest = AccuracyLossTestSet(); 
				
				lossHistory.add(lossTrain); 
				
				Logging.println("It=" + iter + ", gamma= "+gamma+", lossTrain="+ lossTrain + ", lossTest="+ lossTest  +
								", MCRTrain=" +mcrTrain + ", MCRTest=" +mcrTest, LogLevel.DEBUGGING_LOG);
				
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
				
				//PrintShapeletsWeights();
				//PrintProjectedData();
				
				if( lossHistory.size() > 50 )
				{
					if( lossTrain > lossHistory.get( lossHistory.size() - 49  )  )
						break;
				}
			}
		}
		
		// print shapelets and the data for debugging purposes
		//PrintShapeletsAndWeights();
		//PrintProjectedData();
		
		
		return GetMCRTestSet(); 
	}
	

	// the main execution of the program
	public static void main(String [] args)
	{
		// in case ones wants to run it from an IDE like eclipse 
		// then the command line parameters can be set as
		if (args.length == 0) {
			String dir = "/mnt/E/Data/classification/timeseries/",
			ds = "MoteStrain";

			String sp = File.separator; 
		
			args = new String[] {  
				"trainSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
						+ ds + "_TRAIN",  
				"testSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
						+ ds + "_TEST",  
 				"lambdaW=1",       
				"maxEpochs=10000",    
				"gamma=-1",   
				"K=0.3",
				"L=0.2", 
				"R=3",
				"eta=0.01"
				};
		}			

		// values of hyperparameters
		double eta = -1, lambdaW = -1, gamma = -1, L = -1, K = -1;
		int maxEpochs = -1, R = -1;
		String trainSetPath = "", testSetPath = "";
		
		// read and parse parameters
		for (String arg : args) {
			String[] argTokens = arg.split("=");
			
			if (argTokens[0].compareTo("eta") == 0) 
				eta = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("lambdaW") == 0)
				lambdaW = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("gamma") == 0)
				gamma = Integer.parseInt(argTokens[1]); 
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
		if(gamma > 0) gamma = -30;
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

        LearnDiscriminativeMotifs lsg = new LearnDiscriminativeMotifs();   
        // initialize the sizes of data structures
        lsg.ITrain = trainSet.GetNumInstances();  
        lsg.ITest = testSet.GetNumInstances();
        lsg.Q = T.getDimColumns();
        // set the time series and labels
        lsg.T = T;
        lsg.Y = O;
        // set the learn rate and the number of iterations
        lsg.maxIter = maxEpochs;
        // set te number of patterns
        lsg.K = (int)(K*T.getDimColumns());
        lsg.L_min = (int)(L*T.getDimColumns());
        lsg.R = R;
        // set the regularization parameter
        lsg.lambdaW = lambdaW;  
        lsg.eta = eta;  
        lsg.gamma = gamma; 
        trainSet.ReadNominalTargets();
        lsg.nominalLabels =  new ArrayList<Double>(trainSet.nominalLabels);
        
        // learn the model
        lsg.Learn(); 
        
        // learn the local convolutions

        long endTime = System.currentTimeMillis(); 
        
		System.out.println( 
				String.valueOf(lsg.GetMCRTestSet())  + " " + String.valueOf(lsg.GetMCRTrainSet()) + " "
				+ "L=" + L	+ " " 
				+ "R=" + R	+ " " 
				+ "lW=" + lambdaW + " "
				+ "gamma=" + gamma + " " 
				+ "eta=" + eta + " " 
				+ "maxEpochs="+ maxEpochs + " " 
				+ "time="+ (endTime-startTime) 
				); 
	}
		
}
