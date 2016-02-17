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


public class LearnDiscriminativeMotifsInverseEuclidean 
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
	
	double scale = 0.05;
	int offsets[];
	
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
	
	Random rand = new Random();
	
	// constructor
	public LearnDiscriminativeMotifsInverseEuclidean()
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
		offsets = new int[R];
		
		for(int r=0; r < R; r++)
		{
			L[r] = (r+1)*L_min;
			
			offsets[r] = (int) (L[r]*scale);
			if(offsets[r] <= 0) offsets[r] = 1;
			
			J[r] = (Q - L[r])/offsets[r];
			
			Logging.println("Initialize Shapelets: r="+r+", J[r]="+J[r]+", L[r]="+L[r] + ", offset[r]="+offsets[r], LogLevel.DEBUGGING_LOG);
			
			double [][] segmentsR = new double[(ITrain + ITest)*J[r]][L[r]];
			
			for(int i= 0; i < ITrain; i++) 
				for(int j= 0; j < J[r]; j++) 
					for(int l = 0; l < L[r]; l++)
						segmentsR[i*J[r] + j][l] = T.get(i, j*offsets[r]+l);

			// normalize segments
			for(int i= 0; i < ITrain; i++) 
				for(int j= 0; j < J[r]; j++) 
					for(int l = 0; l < L[r]; l++)
						segmentsR[i*J[r] + j] = StatisticalUtilities.Normalize(segmentsR[i*J[r] + j]);
			
			
			KMeans kmeans = new KMeans();
			Motifs[r] = kmeans.InitializeKMeansPP(segmentsR, K, kMeansIter); 
			
			if( Motifs[r] == null)
				System.out.println("P not set");
		}
	}
	
	public double Predict(int i, int c)
	{
		double Y_hat_ic = biasW[c];

		for(int r = 0; r < R; r++)
			for(int k = 0; k < K; k++)
			Y_hat_ic +=  GetFrequency(r,i,k) * W[c][r][k];
		
		return Y_hat_ic;
	} 
	
	// get the frequency of motif k at scale r inside the i-th series
	public double GetFrequency(int r, int i, int k)
	{
		double F_irk = 0;
		
		for(int j = 0; j < J[r]; j++)
			F_irk += 1.0 / (1.0 + gamma*GetSegmentDist(r, i, k, j)); 
		
		
		return F_irk; 
	}
	
	
	// the distance between the k-th motif at scale r and the j-th segment of the i-th series
	public double GetSegmentDist(int r, int i, int k, int j)
	{
		double err = 0, D_rikj = 0;
		
		for(int l = 0; l < L[r]; l++)
		{
			err = T.get(i, j*offsets[r]+l)- Motifs[r][k][l];
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
			double max_Y_hat_ic = Double.MIN_VALUE;
			int label_i = -1; 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = Sigmoid.Calculate( Predict(i, c) );
				
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
			double max_Y_hat_ic = Double.MIN_VALUE;
			int label_i = -1; 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = Sigmoid.Calculate( Predict(i, c) );
				
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
		double Y_hat_ic = Predict(i, c);
		double sig_y_ic = Sigmoid.Calculate(Y_hat_ic);
		
		return -Y_b.get(i,c)*Math.log( sig_y_ic ) - (1-Y_b.get(i, c))*Math.log(1-sig_y_ic); 
	}
	
	// compute the accuracy loss of the train set
	public double AccuracyLossTrainSet()
	{
		double accuracyLoss = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
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
			for(int c = 0; c < C; c++) 
				accuracyLoss += AccuracyLoss(i, c); 
		}
		return accuracyLoss;
	}
	
	public void LearnO()
	{
		LearnOOnlyW();
		
		double regWConst = (double)2.0*lambdaW;
		double temp = 0;
		
		// first update the weights
		for(int c = 0; c < C; c++)
		{
		    for(int r = 0; r < R; r++)
			{
				for(int k = 0; k < K; k++)
				{
					W[c][r][k] -= eta*regWConst*W[c][r][k];					
					for(int i = 0; i < ITrain; i++)
						W[c][r][k] += eta*(Y_b.get(i,c) - Predict(i,c))*GetFrequency(r, i, k);   
				}
			}
		    
		    for(int i = 0; i < ITrain; i++)
		    	biasW[c] += eta*(Y_b.get(i,c) - Predict(i,c));
		}
		
		// then update the
		double gradOverSegments = 0;
		
		for(int r = 0; r < R; r++)
		{
			temp = (2*gamma) / L[r];
			
			for(int k = 0; k < K; k++)
			{
				for(int l = 0; l < L[r]; l++)
				{
					for(int i = 0; i < ITrain; i++) 
					{
						for(int c = 0; c < C; c++) 
						{
							gradOverSegments = 0;
							for(int j = 0; j < J[r]; j++)
								gradOverSegments += (Motifs[r][k][l] - T.get(i, j+l)) / 
														StatisticalUtilities.Power(1.0 + gamma*GetSegmentDist(r, i, k, j), 2); 
								
							Motifs[r][k][l] -= eta*( Y_b.get(i, c) - Predict(i, c) )*W[c][r][k]*temp*gradOverSegments;
						}
						
					}
					
					
				}									
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
				{
					for(int k = 0; k < K; k++)
					{
						W[c][r][k] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON; 
					}
				}
				biasW[c] -= 2*rand.nextDouble()*2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON; 				
			}
		}
		
		
		double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
		
		for(int epochs = 0; epochs < maxIter/10; epochs++)
		{
			for(int c = 0; c < C; c++)
			{
			    for(int r = 0; r < R; r++)
				{
					for(int k = 0; k < K; k++)
					{
						W[c][r][k] -= eta*regWConst*W[c][r][k];					
						for(int i = 0; i < ITrain; i++)
							W[c][r][k] -= eta*(Predict(i,c) - Y_b.get(i,c))*GetFrequency(r, i, k);   
					}
				}
			    
			    for(int i = 0; i < ITrain; i++)
			    	biasW[c] -= eta*(Predict(i,c) - Y_b.get(i,c));
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
			if( iter % 1 == 0)
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
	
	
	
	// optimize the objective function
	public double LearnSearchInitialMotifs()
	{
		// initialize the data structures
		Initialize();
		
		List<Double> lossHistory = new ArrayList<Double>();
		lossHistory.add(Double.MIN_VALUE);
		
		for(double mtf1Val=-1.0; mtf1Val <= 1.0; mtf1Val += 0.1)
			for(double mt2Val=-1.0; mt2Val <= 1.0; mt2Val += 0.1)
			{
				
				int idxMiddle = L_min/2;
				for(int idx = 0; idx < idxMiddle; idx++)
					Motifs[0][0][idx] = mtf1Val;
					
				for(int idx = idxMiddle; idx < L_min; idx++)
					Motifs[0][0][idx] = mt2Val; 
				
				// apply the stochastic gradient descent in a series of iterations
				for(int iter = 0; iter < maxIter; iter++)
					LearnO();
				
				double lossTrain = AccuracyLossTrainSet();
				double mcrTrain = GetMCRTrainSet();
				System.out.println( mtf1Val+", " + mt2Val + ", " + lossTrain + ", " + mcrTrain);
				
			}
		
		
		
		//Logging.print(M_i, System.out, LogLevel.DEBUGGING_LOG); 
		
		return GetMCRTestSet(); 
	}
	

	public double ClassifySVM() 
	{
		Matrix F = new Matrix(ITrain+ITest, K*R);
		
		// count the frequencies and store in a new representation
		for(int i = 0; i < ITrain+ITest; i++)
		{
			for(int r = 0; r < R; r++)
			{
				for(int k = 0; k < K; k++)
				{
					F.set(i, r*K + k,  GetFrequency(r, i, k) ); 
				}
			}
		}
		
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y, 0, ITrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(F, Y, ITrain, ITrain+ITest); 
	    
	    Instances trainWeka = trainSetHist.ToWekaInstances();
		Instances testWeka = testSetHist.ToWekaInstances();

		SMO svm = WekaClassifierInterface.getPolySvmClassifier(1.0/lambdaW, 3);
		
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
	
	public void PrintShapeletsAndWeights()
	{
		for(int r = 0; r < R; r++)
		{
			for(int k = 0; k < K; k++)
			{
				System.out.print("Motifs("+r+","+k+")= [ ");
				
				for(int l = 0; l < L[r]; l++)
				{
					System.out.print(Motifs[r][k][l] + " ");
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
			System.out.print(Y_b.get(i, c) + " "); 
			
			for(int k = 0; k < K; k++)
			{
				System.out.print(GetFrequency(r, i, k) + " ");
			}
			
			System.out.println(";");
		}
		
		System.out.println("];");
	}
	
	

}
