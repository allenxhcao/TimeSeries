package Regression;

import java.util.Random;

import DataStructures.Matrix;
import MatrixFactorization.MatrixUtilities;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;

public class LoanDefaultPrediction 
{
	// the cumulative predictors matrix 
	Matrix predictors;
	// the train targets matrix
	double [] trainTargets;
	double [] trainLoss;
	
	public double [] testLoss;
	
	// the matrix X, the low rank data U and the regression weights W
	Matrix U, V;
	
	// the weights of the loss variable 
	double [] W;
	
	// the weights of the binary target variable
	double [] Z;
	

	
	// N instances and M_i+1 variables, D_i latent dimensions
	int N, M, NTrain;
	public int D;
	
	// the alpha, all the others will be (1-target) 
	public double betaL, betaT;
	
	// the regularization penalties
	public double lambdaU, lambdaV, lambdaW, lambdaZ;

	// the number of iterations
	public int numIter;
	
	// the learn rate, and the rate for the bias
	public double eta;
	
	// the reciprocal of the sizes
	double recN, recNTrain, recM;
	
	boolean isSemiSupervised = false;
	
	
	
	public LoanDefaultPrediction()
	{
		eta = 0.001;
		betaL = 3;
		betaT=3;
		numIter = 100;
	}

	// initialize the model
	private void Initialize(Matrix trainPredictors, double [] trainLoss, Matrix testPredictors)
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
		
		
		this.trainLoss = trainLoss;
		
		M = predictors.getDimColumns();
		N = predictors.getDimRows();
		NTrain = this.trainLoss.length;
		
		recN = 1.0/(double)N;
		recNTrain = 1.0/(double)NTrain;
		recM = 1.0/(double)M;
		
		// initialize the train targets
		trainTargets = new double[NTrain]; 
		for(int i = 0; i < NTrain; i++ )
			if( trainLoss[i] > 0 )
				trainTargets[i] = 1.0; 
			else
				trainTargets[i] = 0.0;
		
		// normalize predictors
		
		for(int predIdx = 0; predIdx < M; predIdx++)
		{			
			double oldPredMean=StatisticalUtilities.Mean(predictors.getCol(predIdx));
			
			double [] columnValues = predictors.getCol(predIdx);
			
			// replace Missing values with the mean
			for(int rowIdx = 0; rowIdx < N; rowIdx++)
				if(columnValues[rowIdx] == GlobalValues.MISSING_VALUE)
					columnValues[rowIdx] = oldPredMean;
			
			double [] normalizedColumnValues = StatisticalUtilities.Normalize(columnValues);
			
			predictors.setCol(predIdx, normalizedColumnValues);			
			
		}
		
		Logging.println("Predictors normalized! ");
		
		double meanLoss = StatisticalUtilities.Mean(trainLoss);
		Logging.println("Loss mean: " + meanLoss);
		double meanTarget = StatisticalUtilities.Mean(trainTargets);
		Logging.println("Target mean: " + meanTarget);  
		
		Random rand = new Random();
		
		double eps = 0.0001;
		
		//randomly initialize latent matrices
		U = new Matrix(N, D);
		U.RandomlyInitializeCells(-eps, eps);
		
		V = new Matrix(D, M);
		V.RandomlyInitializeCells(-eps, eps);
		
		// randomly initialize loss weights
		W = new double[D];
		for(int k = 0; k < D; k++)
			W[k] = meanLoss;
				
		// randomly initialize target weights
		Z = new double[D];
		
		for(int k = 0; k < D; k++)
			Z[k] = meanTarget;
		
	
		
		Logging.println(
				"Initialized MFRegression, params: " +
				", N="+N+
				", NTrain="+NTrain+
				", F="+M+
				", lambdaU="+lambdaU+
				", lambdaV="+lambdaV+
				", lambdaW="+lambdaW+
				", lambdaZ="+lambdaZ+
				", D="+D+
				", eta="+eta+
				", betaT="+betaT+
				", betaL="+betaL+
				", iter="+numIter, LogLevel.DEBUGGING_LOG); 
		
		
	}
	
	// predict the value of predictor p of instance i
	public double EstimatePredictor(int i, int p)
	{
		double estimate = 0;
		
		for( int k = 0; k < D; k++)
			estimate += U.get(i,k)*V.get(k,p);
		
		return estimate; 
	}
	
	
	public double EstimateLoss(int i)
	{
		double L_hat_i = 0;
		
		for( int k = 0; k < D; k++)
			L_hat_i += U.get(i,k)*W[k];
		
		return L_hat_i;
		
	}
	
	
	
	// estimate the target of instance Ut
	public double EstimateTarget(int i)
	{
		double T_hat_i = 0;
		
		for( int k = 0; k < D; k++)
			T_hat_i += U.get(i, k)*Z[k];
		
		return T_hat_i; 
	}

	
	public void Train(Matrix trainPredictors, double [] theTrainTargets)
	{
		Train(trainPredictors,theTrainTargets, null);
	}
	
	
	// train a regression using Gradient Descent
	public void Train(Matrix trainPredictors, double [] theTrainLoss, Matrix testPredictors)
	{
		// initialize the model
		Initialize(trainPredictors, theTrainLoss, testPredictors);
		
		Random rand = new Random();
		
		// iterate a fixed/predefined time
		for(int iter = 0; iter<numIter; iter++)
		{
			// the error per loss
			if( iter % 5 == 0)
				Logging.println(
					"Iter " + iter + 
					", eta=" + eta +
					", loss=" + PredictorsMSE() +
					", trainMCR=" +  PredictTrainTargetMCR() + 
					", trainMAE=" + PredictTrainLossMAE() + 
					", testMCR=" +  PredictTestTargetMCR() + 
					", testMAE=" + PredictTestLossMAE(testLoss), 
					LogLevel.DEBUGGING_LOG);  
			
			// learn the predictors loss
			for( int instIdx = 0; instIdx < N; instIdx++ )  
			{ 
				int i = rand.nextInt(N); 
				
				// learn the predictors
				for( int predIdx = 0; predIdx < M; predIdx++ ) 
				{
					int p = rand.nextInt(M);
					
					// avoid missing values
					if( predictors.get(i,p) == GlobalValues.MISSING_VALUE )
						continue;
					
					// compute the error in predicting variable c of instance r
					double e_ip = predictors.get(i, p) - EstimatePredictor(i, p); 

			        for(int k = 0; k < D; k++) 
			        {
			            U.set(i, k, U.get(i, k) - eta*( -2*e_ip*V.get(k, p) + 2*lambdaU*U.get(i, k)));
			            V.set(k, p, V.get(k, p) - eta*( -2*e_ip*U.get(i, k) + 2*lambdaV*V.get(k, p)));
			        } 			        
				
				
					// learn the target and loss 
					if( i < NTrain )
					{
						// compute the error in predicting variable c of instance r
						double e_target_i = trainTargets[i] - Sigmoid.Calculate(EstimateTarget(i)); 
				        
				        for(int k = 0; k < D; k++)
				        {
						    U.set(i, k, U.get(i, k) - eta*(betaL*-e_target_i*Z[k] + 2*lambdaU*U.get(i, k)));
			            	Z[k] = Z[k] - eta*(betaL*-e_target_i*U.get(i, k) + 2*lambdaZ*Z[k]);
				        }
				        
				        // compute the conditional regression on defaulting loans
						if( trainLoss[i] > 0 )
						{
							// compute the error in predicting variable c of instance r
							double e_loss_i = trainLoss[i] - EstimateLoss(i); 
						        
					        for(int k = 0; k < D; k++)
					        {
					        	if(e_loss_i < 0)
					        	{
								    U.set(i, k, U.get(i, k) - eta*(betaT*W[k] + 2*lambdaU*U.get(i, k)));  
					            	W[k] = W[k] - eta*(betaT*U.get(i, k) + 2*lambdaW*W[k]);
					        	}
					        	else if(e_loss_i > 0)
					        	{
								    U.set(i, k, U.get(i, k) - eta*(betaT*-W[k] + 2*lambdaU*U.get(i, k)));  
					            	W[k] = W[k] - eta*(betaT*-U.get(i, k) + 2*lambdaW*W[k]);
					        	}	
					        }
						}
					}
				}
				
			}
			
			// bold driver heuristics
			//eta *= 1.03;
			
		}
	}

	
	public void TrainDirectly(Matrix trainPredictors, double [] theTrainLoss, Matrix testPredictors)
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
		
		
		this.trainLoss = theTrainLoss;
		
		M = predictors.getDimColumns();
		N = predictors.getDimRows();
		NTrain = this.trainLoss.length;
		
		recN = 1.0/(double)N;
		recNTrain = 1.0/(double)NTrain;
		recM = 1.0/(double)M;
		
		// initialize the train targets
		trainTargets = new double[NTrain]; 
		for(int i = 0; i < NTrain; i++ )
			if( trainLoss[i] > 0 )
				trainTargets[i] = 1.0; 
			else
				trainTargets[i] = 0.0;
		
		// normalize predictors
		
		for(int predIdx = 0; predIdx < M; predIdx++)
		{			
			double oldPredMean=StatisticalUtilities.Mean(predictors.getCol(predIdx));
			
			double [] columnValues = predictors.getCol(predIdx);
			
			// replace Missing values with the mean
			for(int rowIdx = 0; rowIdx < N; rowIdx++)
				if(columnValues[rowIdx] == GlobalValues.MISSING_VALUE)
					columnValues[rowIdx] = oldPredMean;
			
			double [] normalizedColumnValues = StatisticalUtilities.Normalize(columnValues);
			
			predictors.setCol(predIdx, normalizedColumnValues);			
			
		}
		
		Logging.println("Predictors normalized! ");
		
		double meanLoss = StatisticalUtilities.Mean(trainLoss);
		Logging.println("Loss mean: " + meanLoss);
		double meanTarget = StatisticalUtilities.Mean(trainTargets);
		Logging.println("Target mean: " + meanTarget);  
		
		Random rand = new Random();
		
		double eps = 0.0001;
		
		// randomly initialize loss weights
		W = new double[M];
		for(int k = 0; k < M; k++)
			W[k] = eps + rand.nextDouble()*2*eps;
				
		// randomly initialize target weights
		Z = new double[M];
		
		for(int k = 0; k < M; k++)
			Z[k] = eps + rand.nextDouble()*2*eps;
		
		 
		
		// learn the target and loss
		for(int iter = 0; iter<numIter*10 ; iter++)
		{
			// the error per loss
			if( iter % 5 == 0)
				Logging.println(
					"Iter " + iter + 
					", trainMCR=" +  PredictTrainTargetMCRDirectly() + 
					", trainMAE=" + PredictTrainLossMAEDirectly() + 
					", testMCR=" +  PredictTestTargetMCRDirectly() + 
					", testMAE=" + PredictTestLossMAEDirectly(testLoss), 
					LogLevel.DEBUGGING_LOG);  
			
			// learn the predictors loss
			for( int i = 0; i < NTrain; i++ )  
			{ 
				// compute the error in predicting variable c of instance r
				
				double target_i = 0;
				for(int k = 0; k < M; k++) 
					target_i += predictors.get(i,k)*Z[k];
				
				double e_target_i = trainTargets[i] - Sigmoid.Calculate(target_i); 
		        
		        for(int k = 0; k < M; k++)
		        	Z[k] = Z[k] - eta*(-e_target_i*predictors.get(i, k) + 2*lambdaZ*Z[k]);
		        
		        // compute the conditional regression on defaulting loans
				if( trainLoss[i] > 0 )
				{
					// compute the error in predicting variable c of instance r
					double loss_i = 0;
					for(int k = 0; k < M; k++) 
						loss_i += predictors.get(i,k)*W[k];
					
					double e_loss_i = trainLoss[i] - loss_i; 
				        
			        for(int k = 0; k < M; k++)
			        {
			        	if(e_loss_i < 0)
			        		W[k] = W[k] - eta*(predictors.get(i, k) + 2*lambdaW*W[k]);
			        	else if(e_loss_i > 0)
			        		W[k] = W[k] - eta*(-predictors.get(i, k) + 2*lambdaW*W[k]);	
			        }
				}
			}
			
			// bold driver heuristics
			//eta *= 1.03;
			
		}
	}

		
	public double PredictTrainTargetMCR( )
	{
		double mcr = 0;
		
		for(int i = 0; i < NTrain; i++)
		{
			double T_hat_i = Sigmoid.Calculate( EstimateTarget(i) );
			
			if(trainLoss[i] < 0 && T_hat_i >= 0.5)
				mcr++;
			else if(trainLoss[i] > 0 && T_hat_i < 0.5)
				mcr++;
		}
		
		return  mcr/(double)NTrain;
	}
	
	public double PredictTrainTargetMCRDirectly( )
	{
		double mcr = 0;
		
		for(int i = 0; i < NTrain; i++)
		{
			double target_i = 0;
			for(int k = 0; k < M; k++)
				target_i += predictors.get(i,k)*Z[k];
			
			double T_hat_i = Sigmoid.Calculate( target_i );
			
			if(trainLoss[i] < 0 && T_hat_i >= 0.5)
				mcr++;
			else if(trainLoss[i] > 0 && T_hat_i < 0.5)
				mcr++;
		}
		
		return  mcr/(double)NTrain;
	}
	
	public double PredictTestTargetMCR( )
	{
		double mcr = 0;
		int Ntest = testLoss.length;
		
		for( int i = 0; i < Ntest; i++) 
		{
			double T_hat_i = Sigmoid.Calculate( EstimateTarget(NTrain+i) );
			
			if(testLoss[i] < 0 && T_hat_i >= 0.5)
				mcr++;
			else if(testLoss[i] > 0 && T_hat_i < 0.5)
				mcr++;
		}
		
		return  mcr/(double)Ntest;
	}
	
	public double PredictTestTargetMCRDirectly( )
	{
		double mcr = 0;
		int Ntest = testLoss.length;
		
		for( int i = 0; i < Ntest; i++) 
		{
			double target_i = 0;
			for(int k = 0; k < M; k++)
				target_i += predictors.get(NTrain+i,k)*Z[k];
			
			double T_hat_i = Sigmoid.Calculate( target_i );
			
			if(testLoss[i] < 0 && T_hat_i >= 0.5)
				mcr++;
			else if(testLoss[i] > 0 && T_hat_i < 0.5)
				mcr++;
		}
		
		return  mcr/(double)Ntest;
	}
	
	
	public double PredictTrainLossMAE( )
	{
		double mae = 0; 
		
		for( int i = 0; i < NTrain; i++)
		{			
			double T_hat_i = Sigmoid.Calculate( EstimateTarget(i) ); 
			
			if( T_hat_i >= 0.5 )
			{				
				mae += Math.abs( trainLoss[i] - EstimateLoss(i) );
			}
			else
				mae += Math.abs( trainLoss[i] - 0);
		}
			
		return  mae/(double) NTrain; 
	} 
	
	
	public double PredictTrainLossMAEDirectly( )
	{
		double mae = 0; 
		
		for( int i = 0; i < NTrain; i++)
		{
			double target_i = 0;
			for(int k = 0; k < M; k++)
				target_i += predictors.get(i,k)*Z[k];
			
			double T_hat_i = Sigmoid.Calculate( target_i );
			
			if( T_hat_i >= 0.5 )
			{
				double loss_i = 0;
				for(int k = 0; k < M; k++)
					loss_i += predictors.get(i,k)*W[k];
				
				mae += Math.abs( trainLoss[i] - loss_i );
			}
			else
				mae += Math.abs( trainLoss[i] - 0);
		}
			
		return  mae/(double) NTrain; 
	} 
	
	public double PredictTestLossMAE( double [] testLoss )
	{
		double mae = 0; 
		
		int Ntest = testLoss.length;
	
		for( int i = 0; i < Ntest; i++) 
		{			
			double T_hat_i = Sigmoid.Calculate( EstimateTarget(NTrain+i) );
			
			if( T_hat_i > 0.5)
			{				
				mae += Math.abs( testLoss[i] - EstimateLoss(NTrain+i) );
			}
			else
				mae += Math.abs( testLoss[i] - 0);
		}

		return  mae / Ntest;
	}

	public double PredictTestLossMAEDirectly( double [] testLoss )
	{
		double mae = 0; 
		
		int Ntest = testLoss.length;
	
		for( int i = 0; i < Ntest; i++) 
		{
			double target_i = 0;
			for(int k = 0; k < M; k++)
				target_i += predictors.get(NTrain+i,k)*Z[k];
			
			double T_hat_i = Sigmoid.Calculate( target_i );
			
			if( T_hat_i > 0.5)
			{
				double loss_i = 0;
				for(int k = 0; k < M; k++)
					loss_i += predictors.get(NTrain+i,k)*W[k];
				
				mae += Math.abs( testLoss[i] - loss_i );
			}
			else
				mae += Math.abs( testLoss[i] - 0);
		}

		return  mae / Ntest;
	}
	
	// Get overall rmse loss
	public double PredictorsMSE()
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

	
	
}
