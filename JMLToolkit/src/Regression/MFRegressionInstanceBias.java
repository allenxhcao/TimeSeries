package Regression;

import java.util.Random;

import DataStructures.Matrix;
import MatrixFactorization.MatrixUtilities;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class MFRegressionInstanceBias 
{
	// the cumulative predictors matrix 
	Matrix predictors;
	// the train targets matrix
	double [] trainTargets;
	
	// the matrix X, the low rank data U and the regression weights W
	Matrix U, V;
	
	// the weights of the target variable 
	double [] W;
	
	// N instances and M_i+1 variables, D_i latent dimensions
	
	int N, M, NTrain;
	public int D;
	
	// the alpha, all the others will be (1-target) 
	public double alpha;
	
	// the regularization penalties
	public double lamU, lamV, lamW, lamBI, lamBV;

	// the number of iterations
	public int numIter;
	
	// the learn rate, and the rate for the bias
	public double eta;
	
	// the reciprocal of the sizes
	double recN, recNTrain, recM;
	
	boolean isSemiSupervised = false;
	
	
	public MFRegressionInstanceBias()
	{
		eta = 0.001;
		alpha = 0.75;
		numIter = 100;
	}

	// initialize the model
	private void Initialize(Matrix trainPredictors, double [] theTrainTargets, Matrix testPredictors)
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
		
		this.trainTargets = theTrainTargets.clone();
		
		M = predictors.getDimColumns();
		N = predictors.getDimRows();
		NTrain = trainTargets.length;
		
		recN = 1.0/(double)N;
		recNTrain = 1.0/(double)NTrain;
		recM = 1.0/(double)(M+1);
		
		double eps = 0.001;
		
		//randomly initialize latent matrices
		U = new Matrix(N, D+2);
		U.RandomlyInitializeCells(-eps, eps);
		
		V = new Matrix(D+2, M);
		V.RandomlyInitializeCells(-eps, eps);
		
		Random rand = new Random();
		W = new double[D+2];
		for(int k = 0; k < D+2; k++)
			W[k] = -eps + rand.nextDouble()*2*eps;
	
		// set to 1 weights:  U(:,D_i+1), Psi_i(D_i,:), W(D_i)
		for(int p = 0; p < M; p++)
			V.set(D, p, 1.0);
		
		W[D] = 1.0;
		
		for(int i = 0; i < N; i++)
			U.set(i, D+1, 1.0);
	
	
		Logging.println(
				"Initialized MFRegression, params: " +
				", N="+N+
				", NTrain="+NTrain+
				", M_i="+M+
				", lamU="+lamU+
				", lamV="+lamV+
				", lamW="+lamW+
				", lamBI="+lamBI+
				", lamBP="+lamBV+
				", D_i="+D+
				", eta="+eta+
				", alpha="+alpha+
				", iter="+numIter, LogLevel.DEBUGGING_LOG); 
		
		
	}
	
	// predict the value of predictor p of instance i
	public double EstimatePredictor(int i, int p)
	{
		return MatrixUtilities.getRowByColumnProduct(U, i, V, p);
	}
	
		// estimate the target of instance Ut
	public double EstimateTarget(double [] Ut)
	{
		double estimate = 0;
		
		for( int k = 0; k < D+2; k++)
			estimate += Ut[k]*W[k];
			
		return estimate; 
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
		
		// iterate a fixed/predifined time
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

					
			        for(int k = 0; k < D+2; k++)
			        {
			            double u_ik = U.get(i, k), v_kp = V.get(k, p);
			            
			            if( k != D+1 )
			            {
			            	U.set(i, k, u_ik - 2*eta*( -alpha*e_ip*v_kp 
			            			+ recM*u_ik* ( k != D ? lamU : lamBI) ) );
			            }
			            if( k != D)
			            {
			            	V.set(k, p, v_kp - 2*eta*( -alpha*e_ip*u_ik 
			            			+ recN*v_kp* ( k != D+1 ? lamV : lamBV) ) );
			            }
			        }
		        	
				}
			}
			
			// learn the targets loss
			for(int i = 0; i < NTrain; i++) 
			{
				// compute the error in predicting variable c of instance r
				double e_ip = trainTargets[i] - EstimateTarget( U.getRow(i) );
		        
		        for(int k = 0; k < D+2; k++)
		        {
		            double u_ik = U.get(i, k);
		            
		            if( k != D+1)
		            	U.set(i, k, u_ik - 2*eta*( -e_ip*W[k] 
		            			+ recM*u_ik* ( k != D ? lamU : lamBI) ) );
		            if( k != D)
		            	W[k] = W[k] - 2*eta*( -e_ip*u_ik 
		            			+ recN*W[k]* ( k != D+1 ? lamV : lamBV) );
		        }
		        
			}
		
			
			// the error per loss
			System.out.println(
					"Iter " + iter + 
					", loss: " + PredictorsLoss() + 
					", trainRMSE=" + PredictTrainSet());
		}
	}

	
	public double PredictTestSet( double [] testTargets )
	{

		double rmse = 0; 
		
		int Ntest = testTargets.length;
	
		for( int i = 0; i < Ntest; i++)
		{
			double err = testTargets[i]-EstimateTarget(U.getRow(NTrain+i));  
			rmse += err*err;
		}

		return  rmse / (N-NTrain);
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
		double [] Ut = new double[D+2];

		Ut[D+1] = 1.0;
		
		for(int iter = 0; iter < numIter; iter++)
		{
			for(int p = 0; p < M; p++) 
			{
				// avoid missing values
				if( Xt[p] == GlobalValues.MISSING_VALUE ) continue;
				
				// compute the error in reconstructing the p-th predictor value of 
				// the new X instance
				double estimated_Xt_p = 0;
				for(int k = 0; k < D+2; k++)
					estimated_Xt_p += Ut[k]*V.get(k, p);

				
				double e_Xt_p = Xt[p] - estimated_Xt_p; 
				
	        	for(int k = 0; k < D+2; k++) 
		        {
		            double v_kp = V.get(k, p);
		            
		            if( k != D+1 )
		            	Ut[k] = Ut[k] - 2*eta*( -alpha*e_Xt_p*v_kp 
		            			+ recM*Ut[k]* ( k != D ? lamU : lamBI) );
		        }
			}
		}
		
		return Ut;
	}
	
	
}
