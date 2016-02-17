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


public class LearnShapeletsGeneralizedHinge 
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
	
	// the regularization parameters
	public double lambdaW, lambdaS;
	
	public List<Double> nominalLabels;
	
	
	// structures for storing the precomputed terms
	double D_i[][][];
	double E_i[][][]; 
	double M_i[][];
	double Psi_i[][];
	double sigY_i[]; 

	Random rand = new Random();
	
	// constructor
	public LearnShapeletsGeneralizedHinge()
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
		Logging.println("lamdaW="+lambdaW + ", alpha="+ alpha, LogLevel.DEBUGGING_LOG);
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		CreateOneVsAllTargets();
		
		Logging.println("Classes="+C, LogLevel.DEBUGGING_LOG);
		
		// initialize shapelets
		InitializeShapeletsProbabilityDistance();
		//InitializeShapeletsRandomly();
		
		// initialize the terms for pre-computation
		D_i = new double[R][K][];
		E_i = new double[R][K][];
		for(int r = 0; r < R; r++)
			for(int k = 0; k < K; k++)
			{
				D_i[r][k] = new double[J[r]];
				E_i[r][k] = new double[J[r]];
			}
		M_i = new double[R][K];
		Psi_i = new double[R][K];
		sigY_i = new double[C];
		
		// initialize the weights
		// learn the weights
		LearnFOnlyW();
		
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
	public void InitializeShapeletsProbabilityDistance()
	{		
		Shapelets = new double[R][][];
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
			Shapelets[r] = kmeans.InitializeKMeansPP(segmentsR, K, 30); 
			
			if( Shapelets[r] == null)
				System.out.println("P not set");
		}
	}
	
	
	// initialize the patterns from random segments
		public void InitializeShapeletsRandomly()
		{		
			Shapelets = new double[R][][];
			J = new int[R]; 
			L = new int[R];
			
			for(int r=0; r < R; r++)
			{
				L[r] = (r+1)*L_min;
				J[r] = Q - L[r];
				
				Shapelets[r] = new double[K][L[r]]; 
				for(int k=0; k < K; k++)
					for(int l=0; l < L[r]; l++)
						Shapelets[r][k][l] = -1.0 + (2.0*(double)(k+1))/(double)K;
			}
		}
	
	// predict the label value Y_hat_i
	public double Predict_i(int c)
	{
		double Y_hat_ic = biasW[c];

		for(int r = 0; r < R; r++)
			for(int k = 0; k < K; k++)
			Y_hat_ic += M_i[r][k] * W[c][r][k];
		
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
					// precompute D_i
					D_i[r][k][j] = 0;
					double err = 0;
					
					for(int l = 0; l < L[r]; l++)
					{
						err = T.get(i, j+l)- Shapelets[r][k][l];
						D_i[r][k][j] += err*err; 
					}
					
					D_i[r][k][j] /= (double)L[r]; 
					
					// precompute E_i
					E_i[r][k][j] = Math.exp(alpha * D_i[r][k][j]);
				}
				
				// precompute Psi_i 
				Psi_i[r][k] = 0; 
				for(int j = 0; j < J[r]; j++) 
					Psi_i[r][k] +=  Math.exp( alpha * D_i[r][k][j] );
				
				// precompute M_i 
				M_i[r][k] = 0;
				
				for(int j = 0; j < J[r]; j++)
					M_i[r][k] += D_i[r][k][j]* E_i[r][k][j];
				
				M_i[r][k] /= Psi_i[r][k];
			}
		}
		
		for(int c = 0; c < C; c++)
			sigY_i[c] = Sigmoid.Calculate( Predict_i(c) ); 
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
				double Y_hat_ic = Sigmoid.Calculate( Predict_i(c) );
				
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
				double Y_hat_ic = Sigmoid.Calculate( Predict_i(c) );
				
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
		double Y_hat_ic = Predict_i(c);
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
		double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
		double regSConst = ((double)2.0*lambdaS) / ((double) ITrain*C);
		
		double phi_rikj = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
			
			for(int c = 0; c < C; c++)
			{
				for(int r = 0; r < R; r++)
				{
					for(int k = 0; k < K; k++)
					{
						W[c][r][k] -= eta*(-(Y_b.get(i, c) - sigY_i[c])*M_i[r][k] + regWConst*W[c][r][k]); 
						
						for(int j = 0; j < J[r]; j++)
						{
							phi_rikj = ( (2.0*E_i[r][k][j]) / (L[r] * Psi_i[r][k]) ) 
										* (1 + alpha*(D_i[r][k][j] - M_i[r][k])); 
							
							for(int l = 0; l < L[r]; l++)
							{
								Shapelets[r][k][l] -= eta*(-(Y_b.get(i,c) - sigY_i[c])
										*phi_rikj*(Shapelets[r][k][l] - T.get(i, j+l))*W[c][r][k] 
												+ regSConst*Shapelets[r][k][l] );    
							}
						}				
					}
				}

				biasW[c] -= eta*(-(Y_b.get(i, c) - sigY_i[c])); 
			}			 	
		}
		
		
	}
		
	
	public void LearnFOnlyW()
	{
		W = new double[C][R][K];
		biasW = new double[C];
		
		for(int i = 0; i < ITrain; i++)
		{
			for(int c = 0; c < C; c++)
			{
				for(int r = 0; r < R; r++)
				{
					for(int k = 0; k < K; k++)
					{
						W[c][r][k] = 2*rand.nextDouble()-1; 
					}
				}
				biasW[c] -= 2*rand.nextDouble()-1; 				
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
							W[c][r][k] -= eta*(-(Y_b.get(i, c) - sigY_i[c])*M_i[r][k] + regWConst*W[c][r][k]); 
						}
					}
					
					biasW[c] -= eta*(-(Y_b.get(i, c) - sigY_i[c]));  				
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
			LearnF();

			// measure the loss
			if( iter % 100 == 0)
			{
				double mcrTrain = GetMCRTrainSet();
				double mcrTest = GetMCRTestSet();
				double mcrSVMTest = ClassifySVM();
				
				double lossTrain = AccuracyLossTrainSet();
				double lossTest = AccuracyLossTestSet();
				
				lossHistory.add(lossTrain);
				
				Logging.println("It=" + iter + ", lossTrain="+ lossTrain + ", lossTest="+ lossTest  +
								", MCRTrain=" +mcrTrain + ", MCRTest=" +mcrTest + ", SVM=" + mcrSVMTest, LogLevel.DEBUGGING_LOG);
				
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
		
		// print shapelets for debugging purposes
		//PrintShapelets();
		
		//Logging.print(M_i, System.out, LogLevel.DEBUGGING_LOG); 
		
		return GetMCRTestSet(); 
	}
	
	

	public double ClassifySVM() 
	{
		Matrix F = new Matrix(ITrain+ITest, K*R);
		
		// count the frequencies and store in a new representation
		for(int i = 0; i < ITrain+ITest; i++)
		{
			PreCompute(i); 
			
			for(int r = 0; r < R; r++)
			{
				for(int k = 0; k < K; k++)
				{
					F.set(i, r*K + k,  M_i[r][k] );
				}
			}
		}
		
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y, 0, ITrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(F, Y, ITrain, ITrain+ITest); 
	    
	    Instances trainWeka = trainSetHist.ToWekaInstances();
		Instances testWeka = testSetHist.ToWekaInstances();

		SMO svm = WekaClassifierInterface.getPolySvmClassifier(1.0/lambdaW, 2);
		//SMO svm = WekaClassifierInterface.getRbfSvmClassifier(1, 1);
		
		Evaluation eval = null;
		
		try
		{
			svm.buildClassifier(trainWeka);
			eval = new Evaluation(trainWeka);
			eval.evaluateModel(svm, testWeka);
		}
		catch(Exception exc) 
		{
			Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
		}
		
	    
		return eval.errorRate();
	}
	
	public void PrintShapeletsWeights()
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
				System.out.print(M_i[r][k] + " ");
			}
			
			System.out.println(";");
		}
		
		System.out.println("];");
	}
	
	

}
