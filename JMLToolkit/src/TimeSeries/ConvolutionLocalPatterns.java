package TimeSeries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.swing.text.Utilities;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import TimeSeries.BagOfPatterns.RepresentationType;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;
import Classification.NearestNeighbour;
import Classification.VariableSubsetSelection;
import Classification.WekaClassifierInterface;
import Clustering.KMeans;
import DataStructures.Coordinate;
import DataStructures.DataSet;
import DataStructures.Matrix;


public class ConvolutionLocalPatterns 
{
	// number of training and testing instances
	public int NTrain, NTest;
	// length of a time-series
	public int M;
	// length of a segment
	public int L;
	// number of latent patterns
	public int K;
	
	// local segments
	int NSegments;
	double S[][][];
	
	// latent patterns
	double P[][];
	
	// degrees of membership
	double D[][][];
	
	// the residual errors
	double E[][][];
	
	double svmC, svmDegree;
	
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y;
		
	// the number of iterations
	public int maxIter;
	
	// the regularization parameters
	public double lambdaD, lambdaP, lambdaW, alpha;
	
	// the delta increment between segments of the sliding window
	public int deltaT;
	
	//List<int[]> seriesIndexes;
	
	Random rand = new Random();
	
	public String dataset, fold;
	
	// constructor
	public ConvolutionLocalPatterns()
	{
		deltaT = 1;
		
		svmC = 1.0;
		svmDegree = 3.0;
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		//Logging.println("dataset="+ dataset + ", fold="+ fold, LogLevel.DEBUGGING_LOG);
		//Logging.println("NTrain="+NTrain + ", NTest="+NTest + ", M_i="+M_i, LogLevel.DEBUGGING_LOG);
		
		//Logging.println("maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		//Logging.println("lambdaD="+ lambdaD + ", lambdaF="+ lambdaF + ", lamdaW="+lambdaW + ", alpha="+alpha, LogLevel.DEBUGGING_LOG);
		
		if( deltaT < 1) deltaT = 1;
		else if(deltaT > L ) deltaT = L;
		
		// if the stepsize is equal to the sliding window size,
		// i.e. no overlap between sliding windows
		if( deltaT == L )
			NSegments = M/L;
		else 
			NSegments = (M-L)/deltaT;
		
		// enforce at least one segment is set
		if(L <= M && NSegments <= 0) NSegments = 1;
		//K = NSegments; 
		
		Logging.println("K="+K + ", L="+L, LogLevel.DEBUGGING_LOG);
		Logging.println("deltaT="+ deltaT + ", NSegments="+ NSegments, LogLevel.DEBUGGING_LOG);
		
		//lambdaD *= L;
		//lambdaF *= (NTrain+NTest)*M_i;
		
		
		SegmentTimeSeriesDataset();
		InitializePatternsProbabilityDistance();
		InitializeHardMembershipsToClosestPattern();
	
		E = new double[NTrain+NTest][NSegments][L];
		InitializeErrors();
		
		
		double initialError = MeasureRecontructionLoss();
		//double mcr = ClassifyNearestNeighbor();//ClassifySVM();
		double mcr = ClassifySVM();
		
		Logging.println("InitRecErr="+initialError+", InitMCR=" +mcr, LogLevel.DEBUGGING_LOG);
	}
	
	// initialize the patterns from random segments
	public void InitializePatternsProbabilityDistance()
	{
		double [][] segments = new double[(NTrain + NTest)*NSegments][L];
		for(int i= 0; i < NTrain + NTest; i++) 
			for(int j= 0; j < NSegments; j++) 
				for(int l = 0; l < L; l++)
					segments[i*NSegments + j][l] = S[i][j][l];
		
		KMeans kmeans = new KMeans();
		P = kmeans.InitializeKMeansPP(segments, K);
		
		if( P == null)
			System.out.println("P not set");
	}
	
	public void InitializePatterns()
	{
		// initialize the patterns to some random segments
		P = new double[K][L];
		for(int k= 0; k < K; k++) 
		{
			for(int l = 0; l < L; l++)
			{
				P[k][l]=0;
				
				for(int i = 0; i < NTrain+NTest; i++)
					P[k][l] += S[i][k][l];
				
				P[k][l] /= NTrain+NTest;
			}
			
		}
	}
	
	private void InitializePatternsFromSax() 
	{
		BagOfPatterns bop = new BagOfPatterns();
        
		int initL = 0, initInnDim = 0, initAlphSize = 0;
		
		if(dataset.compareTo("ECG2")==0){
			if(fold.compareTo("1")==0)	   { initL=100; initInnDim=4; initAlphSize=4; }
			else if(fold.compareTo("2")==0){ initL=100; initInnDim=4; initAlphSize=6; }
			else if(fold.compareTo("3")==0){ initL=100; initInnDim=4; initAlphSize=4; }
			else if(fold.compareTo("4")==0){ initL=100; initInnDim=4; initAlphSize=4; }
			else if(fold.compareTo("5")==0){ initL=102; initInnDim=6; initAlphSize=6; }
		}else if(dataset.compareTo("ratbp")==0){
			//if(fold.compareTo("1")==0)	   { initL=105; initInnDim=7; initAlphSize=4; }
			if(fold.compareTo("1")==0)	   { initL=45; initInnDim=3; initAlphSize=4; }
			else if(fold.compareTo("2")==0){ initL=102; initInnDim=6; initAlphSize=8; }
			else if(fold.compareTo("3")==0){ initL=105; initInnDim=7; initAlphSize=4; }
			else if(fold.compareTo("4")==0){ initL=105; initInnDim=7; initAlphSize=6; }
			else if(fold.compareTo("5")==0){ initL=100; initInnDim=4; initAlphSize=8; }
		}else if(dataset.compareTo("gaitpd")==0){
			if(fold.compareTo("1")==0)	   { initL=105; initInnDim=7; initAlphSize=6; }
			else if(fold.compareTo("2")==0){ initL=105; initInnDim=7; initAlphSize=6; }
			else if(fold.compareTo("3")==0){ initL=105; initInnDim=7; initAlphSize=6; }
			else if(fold.compareTo("4")==0){ initL=102; initInnDim=6; initAlphSize=8; }
			else if(fold.compareTo("5")==0){ initL=105; initInnDim=7; initAlphSize=8; }
		}
		if(dataset.compareTo("nesfdb")==0){
			if(fold.compareTo("1")==0)	   { initL=100; initInnDim=4; initAlphSize=4; }
			else if(fold.compareTo("2")==0){ initL=100; initInnDim=4; initAlphSize=4; }
			else if(fold.compareTo("3")==0){ initL=100; initInnDim=4; initAlphSize=6; }
			else if(fold.compareTo("4")==0){ initL=300; initInnDim=4; initAlphSize=4; }
			else if(fold.compareTo("5")==0){ initL=204; initInnDim=3; initAlphSize=4; }
		}
        
		bop.slidingWindowSize = initL; 
        bop.representationType = RepresentationType.SAX; 
        bop.innerDimension = initInnDim; 
        bop.alphabetSize = initAlphSize; 
        
        Logging.println("Selected parameters: initL="+initL+", initInnDim="+initInnDim 
        					+ ", initAlphSize="+initAlphSize, LogLevel.DEBUGGING_LOG);
        
		Matrix H = bop.CreateWordFrequenciesMatrix( T );
		
        L = initL;
        K = bop.dictionary.size();
        NSegments = (M-L)/deltaT; 
		Logging.println("After SAX Initialization, L="+L+", K="+K + ", NSegments="+NSegments, LogLevel.DEBUGGING_LOG); 
        
        P = new double[K][L];
        
        for(int saxWordIdx = 0; saxWordIdx < K; saxWordIdx++)
        {
        	String saxWord = bop.dictionary.get(saxWordIdx);
        	
        	double [] series = bop.sr.RestoreSeriesFromSax(saxWord, L, initAlphSize);
        	
        	for(int l = 0; l < L; l++)
        		P[saxWordIdx][l] = series[l];
        }
        
	}

	public void InitializeMembershipsRandomly()
	{
		D = new double[NTrain+NTest][NSegments][K];
		
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int k = 0; k < K; k++) 
					D[i][j][k] = 1.0 / (double) K; 	
	}
	
	// initialize the degree of membership to the closest pattern
	public void InitializeHardMembershipsToClosestPattern()
	{
		// initialize the degree of membership to 1 for the closest initial pattern
		// and 0 for all the others
		D = new double[NTrain+NTest][NSegments][K];
				
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
			{
				// compute the distance between the i,j-th segment and the k-th pattern
				double minDist = Double.MAX_VALUE;
				int closestPattern = 0;
				
				for(int k = 0; k < K; k++)
				{			
					
					double dist = 0;
					for(int l = 0; l < L; l++)
					{
						double err = S[i][j][l] - P[k][l];
						dist += err*err;
					}
					
					//D_i[i][j][k] = 1.0 /(1.0+ dist);
					
					
					if(dist < minDist)
					{
						minDist = dist;
						closestPattern = k;
					}
					
				}	
				
				
				for(int k = 0; k < K; k++)
					if( k == closestPattern)
						D[i][j][k] = 1.0;
					else
						D[i][j][k] = 0.0;
				
			}
	}
	

	
	// partition the time series into segments
	public void SegmentTimeSeriesDataset()
	{
		S = new double[NTrain+NTest][NSegments][L]; 
		
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int l = 0; l < L; l++)
				{
					S[i][j][l] = T.get(i, (j*deltaT) + l); 
				}
				
				// normalize the segment 
				S[i][j] = StatisticalUtilities.Normalize(S[i][j]);
			}
		}
		
		//Logging.println("Partion to Normalized Segments Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	
 	// update all the D_ijk cells in a coordinate descent fashion
 	public void UpdateMembershipsCoordDesc()
 	{
 		double z = 0, nominator = 0, denominator = 0, Q = 0, diff_P_w_k = 0;
 		double epsilon = 0.0000001;
 		
 		int i=0, j=0, k=0, w = 0;
		
		// update all D_ijk
		for(int iIdx = 0; iIdx < NTrain+NTest; iIdx++)
		{
			for(int jIdx = 0; jIdx < NSegments; jIdx++)
			{
				for(int kIdx = 0; kIdx < K; kIdx++)
				{
					i = rand.nextInt(NTrain+NTest);
					j = rand.nextInt(NSegments);
					
					// pick two indexes k&w which are not zero
					do{ 
						k = rand.nextInt(K); 
						w = rand.nextInt(K);
						Q = D[i][j][k] + D[i][j][w];
					}
					while( w == k || Q == 0);
					
					
					// compute optimal D_ijk denoted as z
					z = 0; 
					nominator = epsilon; 
					denominator = epsilon;
					
					for(int l = 0; l < L; l++)
					{
						diff_P_w_k = P[w][l] - P[k][l];
						
						nominator += -(E[i][j][l] - D[i][j][k]*diff_P_w_k)* diff_P_w_k;  
						denominator += diff_P_w_k*diff_P_w_k; 
						
						//nominator += E_i[i][j][l]*diff_P_w_k;
						//denominator += diff_P_w_k*diff_P_w_k; 
					}
					
					//z = D_i[i][j][k] - nominator/denominator; 
					
					z = nominator/denominator;
					
					z = Math.max(0, Math.min(z,Q)); 
					
					//update the errors 
					for(int l = 0; l < L; l++) 
					{ 
						E[i][j][l] = E[i][j][l] - (z - D[i][j][k])*P[k][l]; 
						E[i][j][l] = E[i][j][l] - (Q - z - D[i][j][w])*P[w][l]; 
					} 
					
					// update the membership
					D[i][j][k] = z; 
					D[i][j][w] = Q - z; 
					
				}
			}
		}
		
 	}
 	
 	// update all the P_kl cells in a coordinate descent fashion
 	public void UpdatePatternsCoordDesc()
 	{
 		
 		double z = 0, nominator = 0, denominator = 0;
 		
 		int k = 0, l = 0;
 		
		// update all P_kl
		for(int kIdx = 0; kIdx < K; kIdx++)
		{
			for(int lIdx = 0; lIdx < L; lIdx++)
			{
				k = rand.nextInt(K); 
				l = rand.nextInt(L); 
				
				UpdatePatternsCoordDesc(k, l);
				 
			}
		}
 	}
 	
 	public void UpdatePatternsCoordDesc(int k, int l)
 	{
 		double z = 0, nominator = 0, denominator = 0;
				
		// compute the optimal value of P_kl denoted as z
		z= 0; nominator=0; denominator=0;
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				nominator += (E[i][j][l] + D[i][j][k]*P[k][l])*D[i][j][k];
				denominator += D[i][j][k]*D[i][j][k];
			}
		}
		
		z = (nominator)/( lambdaP + denominator); 
		
		// update the errors
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
				E[i][j][l] = E[i][j][l] - (z - P[k][l])*D[i][j][k];
		
		P[k][l] = z;
		
 	}
 	
 	// reconstruct the point l of the j-th segment of the i-th time series
	public double Reconstruct(int i, int j, int l)
	{
		double S_ijl = 0;
		
		// apply a convolution of k-many patterns and their degrees of membership
		// use the sum of the l-th points from each pattern P
		for(int k = 0; k < K; k++)
			S_ijl += D[i][j][k] * P[k][l];
		
		return S_ijl;
	}
	
	// measure the reconstruction loss
	public double MeasureRecontructionLoss()
	{
		double reconstructionLoss = 0;
		
		// iterate through all the time series
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int l = 0; l < L; l++)
				{
					reconstructionLoss += E[i][j][l]*E[i][j][l]; 
				}
			}
		}
		
		return reconstructionLoss;
	}
	
	// initalize the error matrix
	public void InitializeErrors()
	{
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
					E[i][j][l]  = S[i][j][l] - Reconstruct(i,j,l);
	}
	
	

	public double ClassifyNearestNeighbor()
	{
		Matrix F = new Matrix(NTrain+NTest, K);
		
		// count the frequencies and store in a new representation
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int k = 0; k < K; k++)
			{
				double F_ik = 0;
				
				for(int j = 0; j < NSegments; j++)
					F_ik += D[i][j][k];
				
				F.set(i, k, F_ik);
			}
		}
		
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y, 0, NTrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(F, Y, NTrain, NTrain+NTest); 
	    
	    NearestNeighbour nn = new NearestNeighbour("euclidean");
	    
		return nn.Classify(trainSetHist, testSetHist);
	}
	

	private double ClassifySVM() 
	{
		Matrix F = new Matrix(NTrain+NTest, K);
		
		// count the frequencies and store in a new representation
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int k = 0; k < K; k++)
			{
				double F_ik = 0;
				
				for(int j = 0; j < NSegments; j++)
					F_ik += D[i][j][k];
				
				F.set(i, k, F_ik);
			}
		}
		
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y, 0, NTrain);
	    DataSet testSetHist = new DataSet();
	    testSetHist.LoadMatrixes(F, Y, NTrain, NTrain+NTest); 
	    
	    Instances trainWeka = trainSetHist.ToWekaInstances();
		Instances testWeka = testSetHist.ToWekaInstances();

		SMO svm = WekaClassifierInterface.getPolySvmClassifier(svmC, svmDegree);
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
	
	public void VerifyMembershipConstraint()
	{
		double sum_D_ijk = 0;
		for(int i = 0; i < NTrain+NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				sum_D_ijk = 0;
				
				for(int k = 0; k < K; k++)
					sum_D_ijk += D[i][j][k];
				
				if( Math.abs( sum_D_ijk - 1 ) > 0.001 )
					System.out.println( sum_D_ijk );
			}
		}
	}
	
	

	private double ApplyVariableSubsetSelection() 
	{
		// run variable selection
		Matrix F = new Matrix(NTrain+NTest, K);
		
		// count the frequencies and store in a new representation
		for(int i = 0; i < NTrain+NTest; i++) 
		{
			for(int k = 0; k < K; k++)
			{
				double F_ik = 0;
				
				for(int j = 1; j < NSegments; j++)
					F_ik += D[i][j][k];
				
				F.set(i, k, F_ik);
			}
		}
		
		// compute the best subset on validation data
		VariableSubsetSelection vss = new VariableSubsetSelection();
		int validationTest = NTrain/4;
		int remainingTrain = NTrain - validationTest;

		Logging.println("remainingTrain=" + remainingTrain + ", validationTest="+validationTest, LogLevel.DEBUGGING_LOG);
		
		vss.Init(F, Y, remainingTrain, validationTest);
		int [] bestSubset = vss.SelectVariables();
		
		// get the mcr on test
		vss.Init(F, Y, NTrain, NTest);
		
		return vss.EvaluateCandidate(bestSubset);
		
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
			// measure the loss and the accuracy, if running on debugging log level
			if( Logging.currentLogLevel.compareTo(LogLevel.DEBUGGING_LOG) > 0 )
			{
				double reconstructionLoss = MeasureRecontructionLoss(); 
				double mcr = ClassifyNearestNeighbor(); 
				Logging.println("It=" + iter + ", LR="+reconstructionLoss + ", MCR="+mcr, LogLevel.DEBUGGING_LOG);
			}
			// fix the reconstruction error of every cell
			// for both reconstruction and accuracy losses
			
			UpdateMembershipsCoordDesc();
			UpdatePatternsCoordDesc(); 
			
		}
		
		
		// print the patterns
		
		//Logging.print( D_i[0], System.out, LogLevel.DEBUGGING_LOG );
		//Logging.print( P, System.out, LogLevel.DEBUGGING_LOG );		
		//Logging.print(T.getRow(0), LogLevel.DEBUGGING_LOG );
		
		
		return ClassifySVM(); 
		//return ClassifySVM();
	}


}
