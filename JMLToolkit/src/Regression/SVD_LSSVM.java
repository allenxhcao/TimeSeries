package Regression;

import java.util.Random;

import Classification.Kernel;
import DataStructures.Matrix;
import MatrixFactorization.MatrixUtilities;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class SVD_LSSVM 
{
	// the cumulative predictors matrix 
	Matrix predictors;
	// the train targets matrix
	double [] trainTargets;
	
	public double [] testTargets;
	
	// the kernel of the post decomposition
	public LSSVM lssvm;
	int [] targetIndices;
	
	// the matrix X, the low rank data U and the regression weights W
	Matrix U, V;
	// predictors bias 
	double [] V0;
	
	
	// N instances and M_i+1 variables, D_i latent dimensions
	int N, M, NTrain;
	public int D;
	
	// the alpha, all the others will be (1-target) 
	public double beta;
	
	// the regularization penalties
	public double lambda, gamma;

	// the number of iterations
	public int numIter;
	
	// the learn rate, and the rate for the bias
	public double eta;
	
	// the reciprocal of the sizes
	double recN, recNTrain, recM;
	
	boolean isSemiSupervised = false;
	
	
	public SVD_LSSVM()
	{
		eta = 0.001;
		beta = 0.75;
		numIter = 100;
	}

	// initialize the model
	private void Initialize(Matrix trainPredictors, double [] trainTargets, Matrix testPredictors)
	{
		predictors = new Matrix(trainPredictors);
		
		if( testPredictors == null )
		{
			isSemiSupervised = true;
		}
		else
		{
			predictors.AppendMatrix(testPredictors);
			isSemiSupervised = false;
		}
		
		this.trainTargets = trainTargets;
		
		M = predictors.getDimColumns();
		N = predictors.getDimRows();
		NTrain = trainTargets.length;
		
		recN = 1.0/(double)N;
		recNTrain = 1.0/(double)NTrain;
		recM = 1.0/(double)M;
		
		
		Random rand = new Random();
		
		//randomly initialize latent matrices
		U = new Matrix(N, D);
		U.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
		
		V = new Matrix(D, M);
		V.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
		// randomly initialize predictor weights biases
		V0 = new double[M];
		for(int p = 0; p < M; p++)
			V0[p] = -GlobalValues.SMALL_EPSILON + rand.nextDouble()*2*GlobalValues.SMALL_EPSILON;
		
		targetIndices = new int[NTrain];
		for( int i = 0; i < NTrain; i++ )
			targetIndices[i]=i;

		Logging.println(
				"Initialized SVD-LSSVM, params: " +
				", N="+N+
				", NTrain="+NTrain+
				", M_i="+M+
				", lamU="+lambda+
				", gamma="+gamma+
				", rho="+lssvm.beta+
				", degree="+lssvm.d+
				", D_i="+D+
				", eta="+eta+
				", beta="+beta+
				", iter="+numIter, LogLevel.DEBUGGING_LOG); 
		
		
	}
	
	// predict the value of predictor p of instance i
	public double EstimatePredictor(int i, int p)
	{
		double estimate = 0;
		
		for( int k = 0; k < D; k++)
			estimate += U.get(i,k)*V.get(k,p);
		
		estimate += V0[p];
			
		return estimate; 
	}
	
		// estimate the target of instance Ut
	public double EstimateTarget(double [] Ut)
	{
		return lssvm.PredictInstance(Ut);
	}

	public void Train(Matrix trainPredictors, double [] theTrainTargets)
	{
		Train(trainPredictors,theTrainTargets, null);
	}
	
	// train a regression using Gradient Descent
	public void Train(Matrix trainPredictors, double [] theTrainTargets, Matrix testPredictors)
	{
		// initialize the model
		Initialize(trainPredictors, theTrainTargets, testPredictors);
		
		// iterate a fixed/predefined time
		for(int iter = 0; iter<numIter ; iter++)
		{
			// learn the predictors loss 
			for(int i = 0; i < N; i++)
			{
				for(int p = 0; p < M; p++) 
				{
					// avoid missing values
					if( predictors.get(i,p) == GlobalValues.MISSING_VALUE ) continue;
					
					// compute the error in predicting variable c of instance r
					double e_ip = predictors.get(i, p) - EstimatePredictor(i, p); 

			        for(int k = 0; k < D; k++)
			        {
			            double u_ik = U.get(i, k), v_kp = V.get(k, p);
			            U.set(i, k, u_ik - eta*( -beta*gamma*e_ip*v_kp + lambda*recM*u_ik));
			            V.set(k, p, v_kp - eta*( -beta*gamma*e_ip*u_ik + recN*v_kp));
			        }		 
			        
			        V0[p] = V0[p] + eta*beta*gamma*e_ip;
				}
			}
			
			// check target MAE with a frequency
			if (iter % 1000 == 0)
			{
				// train the least square svm on the latent factors
				lssvm.Train(U, trainTargets, targetIndices); 
	
				
				// the error per loss
				Logging.println(
						"Iter " + iter + 
						", loss: " + PredictorsLoss() + 
						", trainMSE=" + PredictTrainSet() + 
						", testMSE=" + PredictTestSet(testTargets), LogLevel.DEBUGGING_LOG);
			}
		}
	}

	
	public double PredictTestSet( double [] testTargets )
	{

		double mse = 0; 
		
		int Ntest = testTargets.length;
	
		for( int i = 0; i < Ntest; i++) 
		{
			double err = testTargets[i]-EstimateTarget(U.getRow(NTrain+i));  
			mse += err*err;
		}

		return  mse / Ntest;
	}


	public double PredictTestSet(Matrix testPredictors, double [] testTargets )
	{

		double rmse = 0; 
		
		int NTest = testPredictors.getDimRows();
		
		for( int i = 0; i < NTest; i++)
		{
			double [] Ut = FoldIn( testPredictors.getRow(i) );
			double err = testTargets[i]-EstimateTarget(Ut);
			
			//System.out.println(testTargets[i] + " " + EstimateTarget(Ut));
			
			rmse += err*err;
		}
			
		return rmse / NTest;
	}
	
	public double PredictTrainSet( )
	{

		double rmse = 0;
		
		for( int i = 0; i < NTrain; i++)
		{
			double err = trainTargets[i] - EstimateTarget( U.getRow(i) );
			rmse += err*err;
		}
		
		return  rmse/NTrain;
	}
	
	// Get overall rmse loss
	public double PredictorsLoss()
	{
		double loss = 0;
		int numObservedCells = 0;
		
		for(int i = 0; i < N; i++)
			for(int p = 0; p < M; p++)
			{
				if( predictors.get(i, p) != GlobalValues.MISSING_VALUE )
				{
					double e_rc = predictors.get(i, p) - EstimatePredictor(i, p); 
					loss += e_rc * e_rc;
					numObservedCells++;
				}
			}
		
		return loss/numObservedCells; 
	}
	
	// fold in a new test instance into the latent dimensionality
	public double [] FoldIn( double [] Xt)
	{
		double [] Ut = new double[D+1];

		Ut[D] = 1.0;
		
		for(int iter = 0; iter < numIter; iter++)
		{
			for(int p = 0; p < M; p++) 
			{
				// avoid missing values
				if( Xt[p] == GlobalValues.MISSING_VALUE ) continue;
				
				// compute the error in reconstructing the p-th predictor value of 
				// the new X instance
				double estimated_Xt_p = 0;  
				for(int k = 0; k < D; k++)  
					estimated_Xt_p += Ut[k]*V.get(k, p); 
				
				estimated_Xt_p += V0[p];
				
				double e_Xt_p = Xt[p] - estimated_Xt_p; 
				
	        	for(int k = 0; k < D; k++) 
		        {
		            double v_kp = V.get(k, p);
	            	Ut[k] = Ut[k] - 2*eta*( -beta*e_Xt_p*v_kp + recM*Ut[k]*lambda );
		        }
			}
		}
		
		return Ut;
	}
	
	
}
