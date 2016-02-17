package Regression;

import java.util.Random;

import DataStructures.Matrix;
import MatrixFactorization.MatrixUtilities;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.StatisticalUtilities;

public class CopyOfSRLP 
{
	// the cumulative predictors matrix 
	Matrix predictors;
	// the train targets matrix
	double [] trainTargets;
	
	public double [] testTargets;
	
	// the matrix X, the low rank data U and the regression weights W
	Matrix U, V;
	// predictors bias 
	double [] V0;
	
	// the weights of the target variable 
	double [] W;
	// target bias
	double W0;
	
	// N instances and M_i+1 variables, D_i latent dimensions
	int N, M, NTrain;
	public int D;
	
	// the alpha, all the others will be (1-target) 
	public double beta;
	
	// the regularization penalties
	public double lambdaU, lambdaV, lambdaW;

	// the number of iterations
	public int numIter;
	
	// the learn rate, and the rate for the bias
	public double eta;
	
	// the reciprocal of the sizes
	double recN, recNTrain, recM;
	
	boolean isSemiSupervised = false;
	
	
	
	public CopyOfSRLP()
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
		
		// normalize predictors
		boolean hasIdCol = true;
		for(int predIdx = 0; predIdx < M; predIdx++)
		{
			//if(hasIdCol)
				//if(predIdx == 0)
					//continue;
			
			double oldPredMean=StatisticalUtilities.Mean(predictors.getCol(predIdx));
			double oldPredStd=StatisticalUtilities.StandardDeviation(predictors.getCol(predIdx));
			
			double [] columnValues = predictors.getCol(predIdx);
			
			// replace Missing values with the mean
			for(int rowIdx = 0; rowIdx < N; rowIdx++)
				if(columnValues[rowIdx] == GlobalValues.MISSING_VALUE)
					columnValues[rowIdx] = oldPredMean;
			
			double [] normalizedColumnValues = StatisticalUtilities.Normalize(columnValues);
			
			predictors.setCol(predIdx, normalizedColumnValues);
			
			double newPredMean=StatisticalUtilities.Mean(predictors.getCol(predIdx));
			double newPredStd=StatisticalUtilities.StandardDeviation(predictors.getCol(predIdx));
			
			Logging.println("Predictor="+predIdx+
					", oldMean="+oldPredMean+", oldStd="+oldPredStd +
					", newMean="+newPredMean+", newStd="+newPredStd);
			
		}
		
		Logging.println("Predictors normalized! ");
		
				
		
		Random rand = new Random();
		
		//randomly initialize latent matrices
		U = new Matrix(N, D);
		U.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
		
		V = new Matrix(D, M);
		V.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
		// randomly initialize predictor weights biases
		V0 = new double[M];
		for(int p = 0; p < M; p++)
			V0[p] = trainPredictors.GetColumnMean(p); 
		
		// randomly initialize weights
		W = new double[D];
		for(int k = 0; k < D; k++)
			W[k] = -GlobalValues.SMALL_EPSILON + rand.nextDouble()*2*GlobalValues.SMALL_EPSILON;
		W0 = StatisticalUtilities.Mean( trainTargets );
	
	
		
		Logging.println(
				"Initialized MFRegression, params: " +
				", N="+N+
				", NTrain="+NTrain+
				", M_i="+M+
				", lambdaU="+lambdaU+
				", lambdaV="+lambdaV+
				", lambdaW="+lambdaW+
				", D="+D+
				", eta="+eta+
				", alpha="+beta+
				", iter="+numIter, LogLevel.DEBUGGING_LOG); 
		
		
	}
	
	// predict the value of predictor p of instance i
	public double EstimatePredictor(int i, int p)
	{
		double estimate = 0;
		
		for( int k = 0; k < D; k++)
			estimate += U.get(i,k)*V.get(k,p);
		
		//estimate += V0[p];
			
		return estimate; 
	}
	
		// estimate the target of instance Ut
	public double EstimateTarget(double [] Ut)
	{
		double estimate = 0;
		
		for( int k = 0; k < D; k++)
			estimate += Ut[k]*W[k];
		
		estimate += W0;
			
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
		
		double uReg = lambdaU;
		double wReg = lambdaW;
		double vReg = lambdaV;
		
		// iterate a fixed/predefined time
		for(int iter = 0; iter<numIter ; iter++)
		{
			// learn the predictors loss 
			for(int i = 0; i < N; i++)
			{
				for(int p = 0; p < M; p++) 
				{
					// avoid missing values
					if( predictors.get(i,p) == GlobalValues.MISSING_VALUE )
					{
						System.out.println("Still missing values?");
						continue;
					}
					
					// compute the error in predicting variable c of instance r
					double e_ip = predictors.get(i, p) - EstimatePredictor(i, p); 

			        for(int k = 0; k < D; k++) 
			        {
			            U.set(i, k, U.get(i, k) - eta*( -2*e_ip*V.get(k, p) + 2*lambdaU*U.get(i, k)));
			            V.set(k, p, V.get(k, p) - eta*( -2*e_ip*U.get(i, k) + 2*lambdaV*V.get(k, p)));
			        }		 
			        
			        //V0[p] = V0[p] + eta*e_ip;
				}
			}
			
//			// learn the targets loss
//			for(int i = 0; i < NTrain; i++)  
//			{ 
//				if( trainTargets[i] == GlobalValues.MISSING_VALUE )
//					continue;
//				
//				// compute the error in predicting variable c of instance r
//				double e_i = trainTargets[i] - EstimateTarget( U.getRow(i) );
//		        
//		        for(int k = 0; k < D; k++)
//		        {
//		            double u_ik = U.get(i, k); 
//				    U.set(i, k, u_ik - eta*(-e_i*W[k] + uReg*u_ik));
//	            	W[k] = W[k] - eta*(-e_i*u_ik + wReg*W[k]);
//		        }
//		        
//		       W0 = W0 + eta*e_i;
//			}
			
			//double trainMAE = PredictTrainSetMAE();
			double trainRMSE = PredictTrainSetRMSE();
			
			// if nan detected then re-start from beginning with a smaller learn rate 
			if( Double.isNaN( trainRMSE ) ) 
			{
				//break;
			}
			
			// the error per loss
			Logging.println(
					"Iter " + iter + 
					", loss: " + PredictorsLoss() +
					", trainRMSE=" + trainRMSE + 
					", trainMAE=" + PredictTrainSetMAE() + 
					", testMAE=" + PredictTestSetMAE(testTargets), LogLevel.DEBUGGING_LOG);  
		}
	}

	
	public double PredictTestSetRMSE( double [] testTargets )
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

	public double PredictTestSetMAE( double [] testTargets )
	{
		double mae = 0; 
		
		int Ntest = testTargets.length;
	
		for( int i = 0; i < Ntest; i++) 
		{
			mae += Math.abs( testTargets[i]-EstimateTarget(U.getRow(NTrain+i)) );  
			
		}

		return  mae / Ntest;
	}

	public double PredictTestSetRMSE(Matrix testPredictors, double [] testTargets )
	{
		double rmse = 0; 
		
		int NTest = testPredictors.getDimRows();
		
		for( int i = 0; i < NTest; i++)
		{
			double [] Ut = FoldIn( testPredictors.getRow(i) );
			double err = testTargets[i]-EstimateTarget(Ut);
			
			rmse += err*err;
		}
			
		return rmse / (double)NTest;
	}
	
	public double PredictTestSetMAE(Matrix testPredictors, double [] testTargets )
	{
		double mae = 0; 
		
		int NTest = testPredictors.getDimRows();
		
		for( int i = 0; i < NTest; i++)
		{
			double [] Ut = FoldIn( testPredictors.getRow(i) );
			mae += Math.abs( testTargets[i]-EstimateTarget(Ut) );
		}
			
		return mae / (double)NTest;
	}
	
	public double PredictTrainSetRMSE( )
	{
		double rmse = 0;
		
		for( int i = 0; i < NTrain; i++)
		{
			double err = trainTargets[i] - EstimateTarget( U.getRow(i) );
			rmse += err*err;
		}
		
		return  rmse/(double)NTrain;
	}
	
	public double PredictTrainSetMAE( )
	{
		double mae = 0;
		
		for( int i = 0; i < NTrain; i++)
			mae += Math.abs( trainTargets[i] - EstimateTarget( U.getRow(i) ) );
		
		return  mae/(double)NTrain;
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
	            	Ut[k] = Ut[k] - 2*eta*( -beta*e_Xt_p*v_kp + recM*Ut[k]*lambdaU );
		        }
			}
		}
		
		return Ut;
	}
	
	
}
