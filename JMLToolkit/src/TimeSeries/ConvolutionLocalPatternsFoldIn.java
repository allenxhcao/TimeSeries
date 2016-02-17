package TimeSeries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.swing.text.Utilities;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.clusterers.EM;
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


public class ConvolutionLocalPatternsFoldIn 
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
	double STrain[][][];
	double STest[][][];
	
	// latent patterns
	double P[][];
	
	// degrees of membership
	double DTrain[][][];	
	double DTest[][][];
	
	// the residual errors
	double ETrain[][][];
	double ETest[][][];
	
	double svmC, svmDegree;
	
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y;
		
	// the number of iterations
	public int maxIter;
	
	// the regularization parameters
	public double lambdaP; 
	
	// the delta increment between segments of the sliding window
	public int deltaT;
	
	Random rand = new Random();
	
	public String dataset, fold;
	
	// constructor
	public ConvolutionLocalPatternsFoldIn()
	{
		deltaT = 1;
		
		svmC = 1.0;
		svmDegree = 3.0;
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		if( deltaT < 1) deltaT = 1;
		else if(deltaT > L ) deltaT = L;
		
		//Logging.println("dataset="+ dataset + ", fold="+ fold, LogLevel.DEBUGGING_LOG);
		
		//Logging.println("maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		//Logging.println("lambdaD="+ lambdaD + ", lambdaF="+ lambdaF + ", lamdaW="+lambdaW + ", alpha="+alpha, LogLevel.DEBUGGING_LOG);
		
		
		
		// if the stepsize is equal to the sliding window size,
		// i.e. no overlap between sliding windows
		if( deltaT == L )
			NSegments = M/L;
		else 
			NSegments = (M-L)/deltaT;
		
		// enforce at least one segment is set
		if(L <= M && NSegments <= 0) NSegments = 1;
		
		Logging.println("NTrain="+NTrain + ", NTest="+NTest + ", M="+M + ", L="+ L + ", K="+K, LogLevel.DEBUGGING_LOG);
		//Logging.println("deltaT="+ deltaT + ", NSegments="+ NSegments, LogLevel.DEBUGGING_LOG);
		
		SegmentTimeSeriesDatasetTrain();
		SegmentTimeSeriesDatasetTest();
		
		InitializePatternsProbabilityDistance();
		InitializeHardMembershipsToClosestPatternTrain();
		
	
		InitializeErrorsTrain();		
		
		
		
		//double initialError = MeasureRecontructionLossTrain();
		//double mcr = ClassifySVM();
		
		//Logging.println("InitRecErr="+initialError+", InitMCR=" +mcr, LogLevel.DEBUGGING_LOG);
	}
	
	// initialize the patterns from random segments
	public void InitializePatternsProbabilityDistance()
	{
		double [][] segments = new double[NTrain*NSegments][L];
		for(int i= 0; i < NTrain; i++) 
			for(int j= 0; j < NSegments; j++) 
				for(int l = 0; l < L; l++)
					segments[i*NSegments + j][l] = STrain[i][j][l];
		
		KMeans kmeans = new KMeans(); 
		P = kmeans.InitializeKMeansPP(segments, K);
		
		//System.out.println(P.length);
		
		if( P == null)
			System.out.println("P not set");
		
	}
	
	
	// initialize the degree of membership to the closest pattern
	public void InitializeHardMembershipsToClosestPatternTrain()
	{
		// initialize the degree of membership to 1 for the closest initial pattern
		// and 0 for all the others
		DTrain = new double[NTrain][NSegments][K];
				
		for(int i = 0; i < NTrain; i++)
		{
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
						//System.out.println(k + " " + l + " " + j);
						
						double err = STrain[i][j][l] - P[k][l];
						dist += err*err;
					}
					
					if(dist < minDist)
					{
						minDist = dist;
						closestPattern = k;
					}
					
				}	
				
				
				for(int k = 0; k < K; k++)
					if( k == closestPattern)
						DTrain[i][j][k] = 1.0;
					else
						DTrain[i][j][k] = 0.0;
				
			}
		}
	}
		
	public void InitializeHardMembershipsToClosestPatternTest()
	{
		DTest = new double[NTest][NSegments][K];
		
		for(int i = 0; i < NTest; i++)
		{
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
						double err = STest[i][j][l] - P[k][l];
						dist += err*err;
					}
					
					if(dist < minDist)
					{
						minDist = dist;
						closestPattern = k;
					}
					
				}	
				
				
				for(int k = 0; k < K; k++)
					if( k == closestPattern)
						DTest[i][j][k] = 1.0;
					else
						DTest[i][j][k] = 0.0; 
			}
		}
		
		
	}
	
	// partition the time series into segments
	public void SegmentTimeSeriesDatasetTrain()
	{
		
		STrain = new double[NTrain][NSegments][L]; 
		
		for(int i = 0; i < NTrain; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int l = 0; l < L; l++)
				{
					STrain[i][j][l] = T.get(i, (j*deltaT) + l); 
				}
				
				// normalize the segment 
				STrain[i][j] = StatisticalUtilities.Normalize(STrain[i][j]);
			}
		}
	}
	
	public void SegmentTimeSeriesDatasetTest()
	{		
		STest = new double[NTest][NSegments][L]; 
		
		for(int i = 0; i < NTest; i++)
		{
			for(int j = 0; j < NSegments; j++)
			{
				for(int l = 0; l < L; l++)
				{
					STest[i][j][l] = T.get(i+NTrain, (j*deltaT) + l); 
				}
				
				// normalize the segment 
				STest[i][j] = StatisticalUtilities.Normalize(STest[i][j]); 
			}
		}
	}
	
	
 	// update all the D_ijk cells in a coordinate descent fashion
 	public void UpdateMembershipsCoordDescTrain()
 	{
 		double z = 0, nominator = 0, denominator = 0, Q = 0, diff_P_w_k = 0;
 		double epsilon = 0.0000001;
 		
 		int i=0, j=0, k=0, w = 0;
		
		// update all D_ijk
		for(int iIdx = 0; iIdx < NTrain; iIdx++)
		{
			for(int jIdx = 0; jIdx < NSegments; jIdx++)
			{
				for(int kIdx = 0; kIdx < K; kIdx++)
				{
					i = rand.nextInt(NTrain);
					j = rand.nextInt(NSegments);
					
					// pick two indexes k&w which are not zero
					do{ 
						k = rand.nextInt(K); 
						w = rand.nextInt(K);
						Q = DTrain[i][j][k] + DTrain[i][j][w];
					}
					while( w == k || Q == 0);
					
					
					// compute optimal D_ijk denoted as z
					z = 0; 
					nominator = epsilon; 
					denominator = epsilon;
					
					for(int l = 0; l < L; l++)
					{
						diff_P_w_k = P[w][l] - P[k][l];
						
						nominator += -(ETrain[i][j][l] - DTrain[i][j][k]*diff_P_w_k)* diff_P_w_k;  
						denominator += diff_P_w_k*diff_P_w_k; 
					}
					
					
					z = nominator/denominator;
					
					z = Math.max(0, Math.min(z,Q)); 
					
					//update the errors 
					for(int l = 0; l < L; l++) 
					{ 
						ETrain[i][j][l] = ETrain[i][j][l] - (z - DTrain[i][j][k])*P[k][l]; 
						ETrain[i][j][l] = ETrain[i][j][l] - (Q - z - DTrain[i][j][w])*P[w][l]; 
					} 
					
					// update the membership
					DTrain[i][j][k] = z; 
					DTrain[i][j][w] = Q - z; 
					
				}
			}
		}
		
 	}
 	
 	public void UpdateMembershipsCoordDescTest()
 	{
 		double z = 0, nominator = 0, denominator = 0, Q = 0, diff_P_w_k = 0;
 		double epsilon = 0.0000001;
 		
 		int i=0, j=0, k=0, w = 0;
		
		// update all D_ijk
		for(int iIdx = 0; iIdx < NTest; iIdx++)
		{
			for(int jIdx = 0; jIdx < NSegments; jIdx++)
			{
				for(int kIdx = 0; kIdx < K; kIdx++)
				{
					i = rand.nextInt(NTest);
					j = rand.nextInt(NSegments);
					
					// pick two indexes k&w which are not zero
					do{ 
						k = rand.nextInt(K); 
						w = rand.nextInt(K);
						Q = DTest[i][j][k] + DTest[i][j][w];
					}
					while( w == k || Q == 0);
					
					
					// compute optimal D_ijk denoted as z
					z = 0; 
					nominator = epsilon; 
					denominator = epsilon;
					
					for(int l = 0; l < L; l++)
					{
						diff_P_w_k = P[w][l] - P[k][l];
						
						nominator += -(ETest[i][j][l] - DTest[i][j][k]*diff_P_w_k)* diff_P_w_k;  
						denominator += diff_P_w_k*diff_P_w_k; 
					}
					
					z = nominator/denominator;
					
					z = Math.max(0, Math.min(z,Q)); 
					
					//update the errors 
					for(int l = 0; l < L; l++) 
					{ 
						ETest[i][j][l] = ETest[i][j][l] - (z - DTest[i][j][k])*P[k][l]; 
						ETest[i][j][l] = ETest[i][j][l] - (Q - z - DTest[i][j][w])*P[w][l]; 
					} 
					
					// update the membership
					DTest[i][j][k] = z; 
					DTest[i][j][w] = Q - z; 
					
				}
			}
		}
		
 	}
 	
 	// update all the P_kl cells in a coordinate descent fashion
 	public void UpdatePatternsCoordDescTrain()
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
				
				// compute the optimal value of P_kl denoted as z
				z= 0; nominator=0; denominator=0;
				for(int i = 0; i < NTrain; i++)
				{
					for(int j = 0; j < NSegments; j++)
					{
						nominator += (ETrain[i][j][l] + DTrain[i][j][k]*P[k][l])*DTrain[i][j][k];
						denominator += DTrain[i][j][k]*DTrain[i][j][k]; 
					}
				}
				
				z = (nominator)/( lambdaP + denominator); 
				
				// update the errors
				for(int i = 0; i < NTrain; i++)
					for(int j = 0; j < NSegments; j++)
						ETrain[i][j][l] = ETrain[i][j][l] - (z - P[k][l])*DTrain[i][j][k]; 
				
				P[k][l] = z;
				
				 
			}
		}
 	}
 	
 	// reconstruct the point l of the j-th segment of the i-th time series
	public double ReconstructTrain(int i, int j, int l)
	{
		double S_ijl = 0;

		for(int k = 0; k < K; k++)
			S_ijl += DTrain[i][j][k] * P[k][l];
		
		return S_ijl;
	}
	
 	// reconstruct the point l of the j-th segment of the i-th time series
	public double ReconstructTest(int i, int j, int l)
	{
		double S_ijl = 0;

		for(int k = 0; k < K; k++)
			S_ijl += DTest[i][j][k] * P[k][l]; 
		
		return S_ijl;
	}
	
	// measure the reconstruction loss
	public double MeasureRecontructionLossTrain()
	{
		double reconstructionLoss = 0;
		
		// iterate through all the time series
		for(int i = 0; i < NTrain; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
					reconstructionLoss += ETrain[i][j][l]*ETrain[i][j][l]; 
		
		return reconstructionLoss;
	}
	
	// measure the reconstruction loss
	public double MeasureRecontructionLossTest()
	{
		double reconstructionLoss = 0;
		
		// iterate through all the time series
		for(int i = 0; i < NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
					reconstructionLoss += ETest[i][j][l]*ETest[i][j][l]; 
		
		return reconstructionLoss;
	}
	
	
	// initalize the error matrix
	public void InitializeErrorsTrain()
	{
		ETrain = new double[NTrain][NSegments][L];
		
		for(int i = 0; i < NTrain; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
					ETrain[i][j][l]  = STrain[i][j][l] - ReconstructTrain(i,j,l);
	}
	
	// initalize the error matrix
	public void InitializeErrorsTest()
	{
		ETest = new double[NTest][NSegments][L];
		
		for(int i = 0; i < NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
					ETest[i][j][l]  = STest[i][j][l] - ReconstructTest(i,j,l);
	}
	
	private double ClassifySVM() 
	{
		Matrix F = new Matrix(NTrain+NTest, K);
		
		// count the frequencies and store in a new representation
		// for the training series
		for(int i = 0; i < NTrain; i++)
		{
			for(int k = 0; k < K; k++)
			{
				double F_ik = 0;
				
				for(int j = 0; j < NSegments; j++)
					F_ik += DTrain[i][j][k];
				
				F.set(i, k, F_ik);
			}
		}
		// and similarly for the testing series
		for(int i = 0; i < NTest; i++)
		{
			for(int k = 0; k < K; k++)
			{
				double F_ik = 0;
				
				for(int j = 0; j < NSegments; j++)
					F_ik += DTest[i][j][k];
				
				F.set(i+NTrain, k, F_ik);
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
	
	private double ClassifySVMTrain() 
	{
		Matrix F = new Matrix(NTrain, K);
		
		// count the frequencies and store in a new representation
		// for the training series
		for(int i = 0; i < NTrain; i++)
		{
			for(int k = 0; k < K; k++)
			{
				double F_ik = 0;
				
				for(int j = 0; j < NSegments; j++)
					F_ik += DTrain[i][j][k];
				
				F.set(i, k, F_ik);
			}
		}
		
		DataSet trainSetHist = new DataSet();
	    trainSetHist.LoadMatrixes(F, Y);
	    
	    Instances trainWeka = trainSetHist.ToWekaInstances();

		SMO svm = WekaClassifierInterface.getPolySvmClassifier(svmC, svmDegree);
		Evaluation eval = null;
		
		try
		{
			svm.buildClassifier(trainWeka);
			eval = new Evaluation(trainWeka);
			
			eval.crossValidateModel(svm, trainWeka, NTrain, rand ); 
			
		}
		catch(Exception exc) 
		{
			Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
		}
		
	    
		return eval.errorRate();
	}
	
	
	// optimize the objective function
	public double Learn()  
	{ 
		// initialize the data structures
		Initialize(); 
		
		List<Double> lossHistory = new ArrayList<Double>(); 
		lossHistory.add(Double.MIN_VALUE); 
		
		
		// learn the memberships and the patterns from the training data
		// apply the stochastic gradient descent in a series of iterations
		for(int iter = 0; iter < maxIter; iter++) 
		{ 
			//Logging.println("It=" + iter + ", LRTrain="+MeasureRecontructionLossTrain(), LogLevel.DEBUGGING_LOG);
			
			UpdateMembershipsCoordDescTrain(); 
			UpdatePatternsCoordDescTrain();  
		} 
		
		// then learn the membership for the test instances only
		
		// initialize memberships for the test predictors
		InitializeHardMembershipsToClosestPatternTest();
		// initialize the reconstruction errors
		InitializeErrorsTest();
		
		// iterate and learn the memberships
		
		for(int iter = 0; iter < maxIter; iter++) 
		{ 
			//Logging.println("It=" + iter + ", LRTest="+MeasureRecontructionLossTest(), LogLevel.DEBUGGING_LOG);
			
			UpdateMembershipsCoordDescTest(); 
		} 
		

		// Logging.print(P, System.out, LogLevel.DEBUGGING_LOG );
		
		
		return ClassifySVM(); 
	} 


}
