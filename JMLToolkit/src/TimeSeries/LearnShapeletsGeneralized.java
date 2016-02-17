package TimeSeries;

import info.monitorenter.gui.chart.ITrace2D;

import java.io.File;
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


public class LearnShapeletsGeneralized 
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
	double Shapelets[][][];
	// classification weights
	double W[][][];
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
	
	public int kMeansIter;
	
	// the regularization parameters
	public double lambdaW;
	
	public List<Double> nominalLabels;
	
	
	// structures for storing the precomputed terms
	double D[][][][];
	double E[][][][]; 
	double M[][][];
	double Psi[][][]; 
	double sigY[][]; 

	Random rand = new Random();
	
	List<Integer> instanceIdxs;
	List<Integer> rIdxs;
	
	// constructor
	public LearnShapeletsGeneralized()
	{
		kMeansIter = 100;
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		// avoid K=0 
		if(K == 0) 
			K = 1;
		
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		CreateOneVsAllTargets();
		
		
		// initialize the shapelets (complete initialization during the clustering)
		Shapelets = new double[R][][];
		// initialize the number of shapelets and the length of the shapelets 
		J = new int[R]; 
		L = new int[R];
		// set the lengths of shapelets and the number of segments
		// at each scale r
		int totalSegments = 0;
		for(int r = 0; r < R; r++)
		{
			L[r] = (r+1)*L_min;
			J[r] = Q - L[r];
			
			totalSegments += ITrain*J[r]; 
		}
		
		// set the total number of shapelets per scale as a rule of thumb
		// to the logarithm of the total segments
		if( K < 0)
			K = (int) Math.log(totalSegments);
		
		Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q + ", Classes="+C, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L_min="+ L_min + ", R="+R, LogLevel.DEBUGGING_LOG);
		Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lamdaW="+lambdaW + ", alpha="+ alpha, LogLevel.DEBUGGING_LOG);
		Logging.println("totalSegments="+totalSegments + ", K="+ K, LogLevel.DEBUGGING_LOG);
		
		// initialize an array of the sizes
		rIdxs = new ArrayList<Integer>();
		for(int r = 0; r < R; r++)
				rIdxs.add(r);
		
		// initialize shapelets
		InitializeShapeletsKMeans();
		
		
		// initialize the terms for pre-computation
		D = new double[ITrain+ITest][R][K][];
		E = new double[ITrain+ITest][R][K][];
		
		for(int i=0; i <ITrain+ITest; i++)
			for(int r = 0; r < R; r++)
				for(int k = 0; k < K; k++)
				{
					D[i][r][k] = new double[J[r]];
					E[i][r][k] = new double[J[r]];
				}
		
		// initialize the placeholders for the precomputed values
		M = new double[ITrain+ITest][R][K];
		Psi = new double[ITrain+ITest][R][K];
		sigY = new double[ITrain+ITest][C];
		
		// initialize the weights
		
		W = new double[C][R][K];
		biasW = new double[C];
		
		for(int c = 0; c < C; c++)
		{
			for(int r = 0; r < R; r++)
				for(int k = 0; k < K; k++)
					W[c][r][k] = 2*rand.nextDouble()-1; 
			
			biasW[c] = 2*rand.nextDouble()-1; 				
		}
	
		// precompute the M, Psi, sigY, used later for setting initial W
		for(int i=0; i < ITrain+ITest; i++)
			PreCompute(i); 
		
		// initialize W by learning the model on the centroid data
		LearnFOnlyW();
		
		// store all the instances indexes for
		instanceIdxs = new ArrayList<Integer>();
		for(int i = 0; i < ITrain; i++)
				instanceIdxs.add(i);
		// shuffle the order for a better convergence
		Collections.shuffle(instanceIdxs);
		
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
	
	// initialize the shapelets from the centroids of the segments
	public void InitializeShapeletsKMeans()
	{		
		// a multi-threaded parallel implementation of the clustering
		// on thread for each scale r, i.e. for each set of K shapelets at
		// length L_min*(r+1)
		Parallel_1x0.ForEach(rIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer r)
		    {
				//Logging.println("Initialize Shapelets: r="+r+", J[r]="+J[r]+", L[r]="+L[r], LogLevel.DEBUGGING_LOG);
				
				double [][] segmentsR = new double[(ITrain)*J[r]][L[r]];
				
				for(int i= 0; i < (ITrain); i++) 
					for(int j= 0; j < J[r]; j++) 
						for(int l = 0; l < L[r]; l++)
							segmentsR[i*J[r] + j][l] = T.get(i, j+l);
	
				// normalize segments
				for(int i= 0; i < (ITrain); i++) 
					for(int j= 0; j < J[r]; j++) 
						for(int l = 0; l < L[r]; l++)
							segmentsR[i*J[r] + j] = StatisticalUtilities.Normalize(segmentsR[i*J[r] + j]);
				
				
				KMeans kmeans = new KMeans();
				Shapelets[r] = kmeans.InitializeKMeansPP(segmentsR, K, 100); 
				
				if( Shapelets[r] == null)
					System.out.println("P not set");
			}
		});
	}
	
	// predict the label value vartheta_i
	public double Predict_i(int i, int c)
	{
		double Y_hat_ic = biasW[c];

		for(int r = 0; r < R; r++)
			for(int k = 0; k < K; k++)
			Y_hat_ic += M[i][r][k] * W[c][r][k];
		
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
				{
					// precompute D
					D[i][r][k][j] = 0;
					double err = 0;
					
					for(int l = 0; l < L[r]; l++)
					{
						err = T.get(i, j+l)- Shapelets[r][k][l];
						D[i][r][k][j] += err*err; 
					}
					
					D[i][r][k][j] /= (double)L[r]; 
					
					// precompute E
					E[i][r][k][j] = Math.exp(alpha * D[i][r][k][j]);
				}
				
				// precompute Psi 
				Psi[i][r][k] = 0; 
				for(int j = 0; j < J[r]; j++) 
					Psi[i][r][k] +=  Math.exp( alpha * D[i][r][k][j] );
				
				// precompute M 
				M[i][r][k] = 0;
				
				for(int j = 0; j < J[r]; j++)
					M[i][r][k] += D[i][r][k][j]* E[i][r][k][j];
				
				M[i][r][k] /= Psi[i][r][k];
			}
		}
		
		for(int c = 0; c < C; c++)
			sigY[i][c] = Sigmoid.Calculate( Predict_i(i, c) ); 
	}
	
	// compute the MCR on the test set
	public double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
			
			double max_Y_hat_ic = Double.MIN_VALUE;
			int label_i = -1; 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = Sigmoid.Calculate( Predict_i(i, c) );
				
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
				double Y_hat_ic = Sigmoid.Calculate( Predict_i(i, c) );
				
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
		double Y_hat_ic = Predict_i(i, c);
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
	
	public void LearnF()
	{ 
		// parallel implementation of the learning, one thread per instance
		// up to as much threads as JVM allows
		Parallel_1x0.ForEach(instanceIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer i)
		    {
				double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
				double regSConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
				
				double tmp2 = 0, tmp1 = 0, dLdY = 0, dMdS=0; 
				
				for(int c = 0; c < C; c++)
				{
					PreCompute(i);
					
					dLdY = -(Y_b.get(i, c) - sigY[i][c]);
					
					for(int r = 0; r < R; r++)
					{
						for(int k = 0; k < K; k++)
						{
							W[c][r][k] -= eta*(dLdY*M[i][r][k] + regWConst*W[c][r][k]); 
							
							tmp1 = ( 2.0 / ( (double) L[r] * Psi[i][r][k]) );
							
							for(int l = 0; l < L[r]; l++) 
							{
								tmp2=0;
								for(int j = 0; j < J[r]; j++)
									tmp2 += E[i][r][k][j]*(1 + alpha*(D[i][r][k][j] - M[i][r][k]))*(Shapelets[r][k][l] - T.get(i, j+l));
								
								dMdS = tmp1*tmp2;
								
								Shapelets[r][k][l] -= eta*(dLdY*W[c][r][k]*dMdS ); 
								
							}				
						}
					}
		
					biasW[c] -= eta*dLdY; 
				}			 	
		    }
		
		});
		
	}
		
	
	public void LearnFOnlyW()
	{
		double regWConst = ((double)2.0*lambdaW) / ((double) ITrain);
		
		for(int epochs = 0; epochs < maxIter; epochs++)
		{
			for(int i = 0; i < ITrain; i++)
			{
				for(int c = 0; c < C; c++)
				{
					sigY[i][c] = Sigmoid.Calculate(Predict_i(i, c));
					
					for(int r = 0; r < R; r++)
					{
						for(int k = 0; k < K; k++)
						{
							W[c][r][k] -= eta*(-(Y_b.get(i, c) - sigY[i][c])*M[i][r][k] + regWConst*W[c][r][k]); 
						}
					}
					
					biasW[c] -= eta*(-(Y_b.get(i, c) - sigY[i][c]));  				
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
		for(int iter = 0; iter <= maxIter; iter++)
		{
			// learn the latent matrices
			LearnF(); 
			
			// measure the loss
			if( iter % 500 == 0)
			{
				double mcrTrain = GetMCRTrainSet();
				double mcrTest = GetMCRTestSet(); 
				
				double lossTrain = AccuracyLossTrainSet();
				double lossTest = AccuracyLossTestSet();
				
				lossHistory.add(lossTrain);
				
				Logging.println("It=" + iter + ", alpha= "+alpha+", lossTrain="+ lossTrain + ", lossTest="+ lossTest  +
								", MCRTrain=" +mcrTrain + ", MCRTest=" +mcrTest //+ ", SVM=" + mcrSVMTest
								, LogLevel.DEBUGGING_LOG);
				
				// if divergence is detected start from the beggining 
				// at a lower learning rate
				if( Double.isNaN(lossTrain) || mcrTrain == 1.0 )
				{
					iter = 0;
					
					eta /= 3;
					
					lossHistory.clear();
					
					Initialize();
					
					Logging.println("Divergence detected. Restarting at eta=" + eta, LogLevel.DEBUGGING_LOG);
				}
				
				if( lossHistory.size() > 500 ) 
					if( lossTrain > lossHistory.get( lossHistory.size() - 2  )  )
						break;
			}
		}
		
		return GetMCRTestSet(); 
	}
	
		
	public void PrintShapeletsAndWeights()
	{
		for(int r = 0; r < R; r++)
		{
			for(int k = 0; k < K; k++)
			{
				System.out.print("Shapelets("+r+","+k+")= [ ");
				
				for(int l = 0; l < L[r]; l++)
				{
					System.out.print(Shapelets[r][k][l] + " ");
				}
				
				System.out.println("]");
			}
		}
		
		for(int c = 0; c < C; c++)
		{
			for(int r = 0; r < R; r++)
			{
				System.out.print("W("+c+","+r+")= [ ");
				
				for(int k = 0; k < K; k++)
					System.out.print(W[c][r][k] + " ");
				
				System.out.print(biasW[c] + " ");
				
				System.out.println("]");
			}
		}
	}
	
	
	public void PrintProjectedData()
	{
		int r = 0, c = 0;
		
		System.out.print("Data= [ ");
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i); 
			
			System.out.print(Y_b.get(i, c) + " "); 
			
			for(int k = 0; k < K; k++)
			{
				System.out.print(M[r][k] + " ");
			}
			
			System.out.println(";");
		}
		
		System.out.println("];");
	}
	
	// the main execution of the program
	public static void main(String [] args)
	{
		// in case ones wants to run it from an IDE like eclipse 
		// then the command line parameters can be set as
		if (args.length == 0) {
			String dir = "E:\\Data\\classification\\timeseries\\",
			ds = "MoteStrain";

			String sp = File.separator; 
		
			args = new String[] {  
				"trainSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
						+ ds + "_TRAIN",  
				"testSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
						+ ds + "_TEST",  
 				"lambdaW=1",       
				"maxEpochs=10000",    
				"alpha=-100",   
				"K=0.3",
				"L=0.2", 
				"R=3",
				"eta=0.01"
				};
		}			

		// values of hyperparameters
		double eta = -1, lambdaW = -1, alpha = -1, L = -1, K = -1;
		int maxEpochs = -1, R = -1;
		String trainSetPath = "", testSetPath = "";
		
		// read and parse parameters
		for (String arg : args) {
			String[] argTokens = arg.split("=");
			
			if (argTokens[0].compareTo("eta") == 0) 
				eta = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("lambdaW") == 0)
				lambdaW = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("alpha") == 0)
				alpha = Integer.parseInt(argTokens[1]); 
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
		if(alpha > 0) alpha = -30;
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

        LearnShapeletsGeneralized lsg = new LearnShapeletsGeneralized();   
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
        lsg.alpha = alpha; 
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
				+ "alpha=" + alpha + " " 
				+ "eta=" + eta + " " 
				+ "maxEpochs="+ maxEpochs + " " 
				+ "time="+ (endTime-startTime) 
				); 
	}

}
