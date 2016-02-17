package TimeSeries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.swing.text.Utilities;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import Regression.TimeSeriesPolynomialApproximation;
import TimeSeries.BagOfPatterns.RepresentationType;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;
import Classification.NearestNeighbour;
import Classification.VariableSubsetSelection;
import Classification.WekaClassifierInterface;
import DataStructures.Coordinate;
import DataStructures.DataSet;
import DataStructures.Matrix;


public class ConvolutionLocalPolynomials 
{
	// number of training and testing instances
	public int NTrain, NTest;
	// length of a time-series
	public int M;
	// length of a segment
	public int L;
	// number of latent patterns
	public int K;
	// degree of the polynomial
	public int degree;
	
	// local segments
	int NSegments;
	double S[][][];
	
	// polynomial coefficients
	double beta[][];
	
	// degrees of membership
	double D[][][];
	
	// the residual errors
	double E[][][];
	
	double powers[][];
	double P[][]; 
	
	double eta = 0.000001;
	
	// time series data and the label 
	public Matrix T;
	public Matrix Y;
		
	// the number of iterations
	public int maxIter;
	
	// the regularization parameters
	public double lambdaD, lambdaP, lambdaBeta; 
	
	// the delta increment between segments of the sliding window
	public int deltaT; 
	
	Random rand = new Random();
	
	public String dataset, fold;
	
	// constructor
	public ConvolutionLocalPolynomials()
	{
		deltaT = 1; 
	}
	
	// initialize the data structures
	public void Initialize()
	{ 
		Logging.println("dataset="+ dataset + ", fold="+ fold, LogLevel.DEBUGGING_LOG);
		Logging.println("NTrain="+NTrain + ", NTest="+NTest + ", M_i="+M, LogLevel.DEBUGGING_LOG);
		Logging.println("K="+K + ", L="+L + ", degree="+degree, LogLevel.DEBUGGING_LOG); 
		Logging.println("maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		Logging.println("lambdaD="+ lambdaD + ", lambdaBeta="+ lambdaBeta, LogLevel.DEBUGGING_LOG);
		
		if( deltaT < 1) deltaT = 1;
		else if(deltaT > L ) deltaT = L;
		
		
		NSegments = (M-L)/deltaT; 
		
	
		Logging.println("deltaT="+ deltaT + ", NSegments="+ NSegments, LogLevel.DEBUGGING_LOG);
		 
		//InitializeCoefficientsFromBOP();

		// initialize the patterns
		powers = new double[L][degree+1]; 
		for(int l = 0; l < L; l++) 
			for(int d = 0; d < degree+1; d++) 
				powers[l][d] = StatisticalUtilities.PowerInt(l, d); 
		
		SegmentTimeSeriesDataset();	
		InitializeCoefficientsRandomly(); 
		InitializeHardMembershipsToClosestPattern();  
		//InitializeMembershipsRandomly(); 
		
		
		E = new double[NTrain+NTest][NSegments][L];
		
		// initialize the errors tensor E_i
		InitializeErrors();
		
		double initialError = MeasureRecontructionLoss(); 
		double mcr = ClassifyNearestNeighbor(); 
		
		Logging.println("InitRecErr="+initialError+", InitMCR=" +mcr, LogLevel.DEBUGGING_LOG); 
		Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG); 
	}
	
	// initialize the patterns from random segments
	public void InitializeCoefficientsRandomly()
	{
		TimeSeriesPolynomialApproximation tspa = new TimeSeriesPolynomialApproximation(L, degree);
		// initialize the patterns to some random segments 
		beta = new double[K][degree+1]; 
		
		for(int k = 0; k < K; k++)
		{
			int i = rand.nextInt(NTrain + NTest);
			int j = rand.nextInt(NSegments);
			
			double [] coeffs_ij = tspa.FitPolynomialToSubSeries(S[i][j]);
			for(int d = 0; d < degree+1; d++)
				beta[k][d] = coeffs_ij[d];
		}
				
		P = new double[K][L]; 
		
	}
	
	public void InitializeCoefficientsFromBOP()
	{
		BagOfPatterns bop = new BagOfPatterns();
        
		int initL = 0, initAlphSize = 0, initDegree = 0;
		
		if(dataset.compareTo("ECG2")==0){
			if(fold.compareTo("1")==0)	   { initL=100; initAlphSize=4; initDegree=3; }
			else if(fold.compareTo("2")==0){ initL=100; initAlphSize=4; initDegree=3; }
			else if(fold.compareTo("3")==0){ initL=100; initAlphSize=4; initDegree=4; }
			else if(fold.compareTo("4")==0){ initL=100; initAlphSize=4; initDegree=3; }
			else if(fold.compareTo("5")==0){ initL=100; initAlphSize=4; initDegree=3; } 
		}else if(dataset.compareTo("ratbp")==0){
			if(fold.compareTo("1")==0)	   { initL=100; initAlphSize=6; initDegree=4; }
			else if(fold.compareTo("2")==0){ initL=100; initAlphSize=6; initDegree=7; }
			else if(fold.compareTo("3")==0){ initL=100; initAlphSize=4; initDegree=6; }
			else if(fold.compareTo("4")==0){ initL=200; initAlphSize=4; initDegree=5; }
			else if(fold.compareTo("5")==0){ initL=100; initAlphSize=6; initDegree=6; }
		}else if(dataset.compareTo("gaitpd")==0){
			if(fold.compareTo("1")==0)	   { initL=100; initAlphSize=4; initDegree=7; }
			else if(fold.compareTo("2")==0){ initL=100; initAlphSize=6; initDegree=7; }
			else if(fold.compareTo("3")==0){ initL=100; initAlphSize=4; initDegree=7; }
			else if(fold.compareTo("4")==0){ initL=100; initAlphSize=4; initDegree=8; }
			else if(fold.compareTo("5")==0){ initL=100; initAlphSize=6; initDegree=7; }
		}else if(dataset.compareTo("nesfdb")==0){
			if(fold.compareTo("1")==0)	   { initL=100; initAlphSize=8; initDegree=4; }
			else if(fold.compareTo("2")==0){ initL=100; initAlphSize=6; initDegree=4; }
			else if(fold.compareTo("3")==0){ initL=200; initAlphSize=6; initDegree=3; } 
			else if(fold.compareTo("4")==0){ initL=100; initAlphSize=4; initDegree=5; }
			else if(fold.compareTo("5")==0){ initL=100; initAlphSize=6; initDegree=4; }
		}else if(dataset.compareTo("OSULeaf")==0){
			if(fold.compareTo("1")==0)	   { initL=64; initAlphSize=4; initDegree=4; } 
			else if(fold.compareTo("2")==0){ initL=64; initAlphSize=4; initDegree=4; } 
			else if(fold.compareTo("3")==0){ initL=64; initAlphSize=4; initDegree=4; } 
			else if(fold.compareTo("4")==0){ initL=64; initAlphSize=4; initDegree=4; } 
			else if(fold.compareTo("5")==0){ initL=64; initAlphSize=4; initDegree=4; } 
		}
        
		bop.slidingWindowSize = initL; 
        bop.representationType = RepresentationType.Polynomial; 
        bop.alphabetSize = initAlphSize;
        bop.polyDegree = initDegree;
        
        Logging.println("Selected parameters: initL="+initL+", initInnDim="+initAlphSize 
        					+ ", initAlphSize="+initDegree, LogLevel.DEBUGGING_LOG);
        
		Matrix H = bop.CreateWordFrequenciesMatrix( T );
		
        L = initL;
        K = bop.dictionary.size(); 
        NSegments = (M-L)/deltaT; 
        degree = initDegree;
		Logging.println("After Poly Initialization, L="+L+", K="+K + ", NSegments="+NSegments, LogLevel.DEBUGGING_LOG); 
        
        
        beta = new double[K][degree]; 
        
        for(int polyWordIdx = 0; polyWordIdx < K; polyWordIdx++)
        {
        	String polyWord = bop.dictionary.get(polyWordIdx);
        	
        	beta[polyWordIdx] = bop.pr.ConvertWordToCoeffs(polyWord, initAlphSize); 
        }
        
        // initialize the patterns
        P = new double[K][L];
        
	}
	
	
	public void InitializeMembershipsRandomly()
	{
		D = new double[NTrain+NTest][NSegments][K];
		
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int k = 0; k < K; k++)
					D[i][j][k] = 1.0 / K;
				
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
						double err = S[i][j][l] - ComputeP(k, l);  
						dist += err*err;
					}
					
					if(dist < minDist)
					{
						minDist = dist;
						closestPattern = k;
					}
					
					//D_i[i][j][k] = rand.nextDouble();
				}	
				
				
				for(int k = 0; k < K; k++)
					if( k == closestPattern)
						D[i][j][k] = 1.0;
					else
						D[i][j][k] = 0.0;
				
				
				
			}
	}

	public double ComputeP(int k, int l)
	{
		double P_kl = 0;
		for(int d = 0; d < degree+1; d++)
			P_kl += beta[k][d]*powers[l][d]; 
		
		return P_kl;
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
				double [] normalizedSegment = StatisticalUtilities.Normalize(S[i][j]);
				for(int l = 0; l < L; l++)
				{
					S[i][j][l] = normalizedSegment[l];
				}
				
			}
		}
		
		Logging.println("Partion to Normalized Segments Completed!", LogLevel.DEBUGGING_LOG);
	}
	
	// update the patterns P
	public void UpdateP()
	{
 		for(int k = 0; k < K; k++)
			for(int l = 0; l < L; l++)
				P[k][l] = ComputeP(k, l);
	}
	
 	// update all the D_ijk cells in a coordinate descent fashion
 	public void UpdateMembershipsCoordDesc() 
 	{
 		// update values of P
 		UpdateP();
 		
 		double z = 0, nominator = 0, denominator = 0;
 		
 		int i=0, j=0, k=0;
		
		// update all D_ijk
		for(int iIdx = 0; iIdx < NTrain+NTest; iIdx++)
		{
			for(int jIdx = 0; jIdx < NSegments; jIdx++)
			{
				for(int kIdx = 0; kIdx < K; kIdx++)
				{
					i = rand.nextInt(NTrain+NTest);
					j = rand.nextInt(NSegments);
					k = rand.nextInt(K); 
					
					// compute optimal D_ijk denoted as z
					z = 0; nominator = 0; denominator = 0;
					for(int l = 0; l < L; l++)
					{
						nominator += (E[i][j][l] + D[i][j][k]*P[k][l])*P[k][l];
						denominator += P[k][l]*P[k][l];
					}
					
					z = (nominator)/( lambdaD +  denominator);
					
					//update the errors
					for(int l = 0; l < L; l++)
						E[i][j][l] = E[i][j][l] - (z - D[i][j][k])*P[k][l]; 
					
					// update the membership
					D[i][j][k] = z;
				}
			}
		}
		
 	}
 	
 	// update all the P_kl cells in a coordinate descent fashion
 	public void UpdateCoefficientsCoordDesc()
 	{
 		double z = 0, nominator = 0, denominator = 0, cte = 0, epsilon = 0.000001;
 		int k = 0, d = 0;
 		
 		for(int kIdx = 0; kIdx < K; kIdx++)
 		{
 			for(int dIdx = 0; dIdx < degree+1; dIdx++)
 			{
 				// compute optimal beta_kd
 				z=0; nominator=0; denominator=0;
 				
 				k = rand.nextInt(K);
 				d = rand.nextInt(degree+1);
 				
 				for(int i = 0; i < NTrain+NTest; i++)
 				{
 					for(int j = 0; j < NSegments; j++)
 					{
 						for(int l = 0; l < L; l++)
 						{
 							cte = D[i][j][k]*powers[l][d];
 							
 							nominator = nominator + (E[i][j][l] + cte*beta[k][d])*cte;
 							denominator = denominator + cte*cte;
 						}
 					}
 				}
 				
 				//System.out.println("nom="+nominator + ", den="+denominator);
 				
 				z = (nominator)/( lambdaBeta +  denominator);
 				
 				for(int i = 0; i < NTrain+NTest; i++)
 					for(int j = 0; j < NSegments; j++)
 						for(int l = 0; l < L; l++)
 							E[i][j][l] = E[i][j][l] - (z - beta[k][d])*D[i][j][k]*powers[l][d];
				
 				
 				beta[k][d] = z;
 				
 				//System.out.println("Update " + k + "+" + d);
 			}
 		}	
 	}
 	
 	// reconstruct the point l of the j-th segment of the i-th time series
	public double Reconstruct(int i, int j, int l)
	{
		double S_ijl = 0;
		
		// apply a convolution of k-many patterns and their degrees of membership
		// use the sum of the l-th points from each pattern P
		for(int k = 0; k < K; k++)
		{
			S_ijl += D[i][j][k] * P[k][l]; 
		}
		
		return S_ijl;
	}
	
	// measure the reconstruction loss
	public double MeasureRecontructionLoss()
	{
		double reconstructionLoss = 0;
		
		// iterate through all the time series
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
					reconstructionLoss += E[i][j][l]*E[i][j][l]; 
			
		return reconstructionLoss;
	}
	
	// measure the reconstruction loss
	public double MeasureRecontructionLossRecompute()
	{
		// update values of P
 		UpdateP();
		 		
		double reconstructionLoss = 0, err = 0;
		
		// iterate through all the time series
		for(int i = 0; i < NTrain+NTest; i++)
			for(int j = 0; j < NSegments; j++)
				for(int l = 0; l < L; l++)
				{
					err = S[i][j][l] - Reconstruct(i, j, l);
					reconstructionLoss += err*err;
				} 
			
		return reconstructionLoss;
	}
	
	// initalize the error matrix
	public void InitializeErrors()
	{
		// update values of P
 		UpdateP();
 		
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
					//F_ik += Math.abs(D_i[i][j][k])*Math.abs(D_i[i][j][k]);
					F_ik += D[i][j][k];
				
				//F_ik = Math.sqrt(F_ik);
				
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
			// measure the loss
			double reconstructionLoss = MeasureRecontructionLoss(); 
			//double reconstructionLoss = MeasureRecontructionLossRecompute();
			double mcr = ClassifyNearestNeighbor();
			Logging.println("It=" + iter + ", LR="+reconstructionLoss + ", MCR="+mcr, LogLevel.DEBUGGING_LOG);
			
			//InitializeErrors();
			//reconstructionLoss = MeasureRecontructionLoss();
			//Logging.println("Validate="+ reconstructionLoss + ", MCR="+mcr, LogLevel.DEBUGGING_LOG);
			
			// fix the reconstruction error of every cell
			// for both reconstruction and accuracy losses
			
			UpdateCoefficientsCoordDesc();
			UpdateMembershipsCoordDesc();
			
			
			//lossHistory.add(reconstructionLoss+accuracyLoss);
		}
		
		ApplyVariableSubsetSelection();
		
		//Logging.print( T.getRow(0),LogLevel.DEBUGGING_LOG );
		//Logging.print( D_i[0], System.out, LogLevel.DEBUGGING_LOG );
		
		return ClassifyNearestNeighbor();
	}

	private void ApplyVariableSubsetSelection() 
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
		
		VariableSubsetSelection vss = new VariableSubsetSelection();
		vss.Init(F, Y, NTrain, NTest);
		int [] bestSubset = vss.SelectVariables(); 
		
	}
	
	
}
