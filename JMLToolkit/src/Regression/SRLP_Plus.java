package Regression;

import java.util.Random;

import Classification.Kernel;
import Classification.Kernel.KernelType;
import DataStructures.Matrix;
import MatrixFactorization.MatrixUtilities;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.StatisticalUtilities;

public class SRLP_Plus 
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
	LSSVM modelTarget;
	public Kernel kernel;
	// the observed indices of the target cells
	int [] observedTargetIndices;
		
	
	// N instances and M_i+1 variables, D_i latent dimensions
	int N, M, NTrain;
	public int D;
	
	// the alpha, all the others will be (1-target) 
	public double beta;
	
	// the regularization penalties
	public double lambda, gamma, rho;

	// the number of iterations
	public int maxEpocs;
	
	// the learn rate, and the rate for the bias
	public double eta;
	
	// the reciprocal of the sizes
	double recN, recNTrain, recM;
	
	boolean isSemiSupervised = false;
	
	public double presenceRatio;
	
	
	public SRLP_Plus()
	{
		eta = 0.001;
		beta = 0.75;
		maxEpocs = 100;
		
		presenceRatio = 0.2; 
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
		{
			//V0[p] = -GlobalValues.SMALL_EPSILON + rand.nextDouble()*2*GlobalValues.SMALL_EPSILON;
			V0[p] = trainPredictors.GetColumnMean(p); 
		}
		
		// randomly initialize weights
		modelTarget = new LSSVM();
		modelTarget.kernel = kernel;
		modelTarget.beta = rho; 
		
		int presentLabels = (int) Math.ceil( presenceRatio*NTrain );
		observedTargetIndices = new int[presentLabels];
		for(int i = 0; i < presentLabels; i++)
		{
			observedTargetIndices[i]=i;
		} 
	
		
		Logging.println(
				"Initialized MFRegression, params: " +
				", N="+N+
				", NTrain="+NTrain+
				", M_i="+M+
				", lamU="+lambda+
				", gamma="+gamma+
				", rho="+rho+
				", D_i="+D+
				", eta="+eta+
				", alpha="+beta+
				", iter="+maxEpocs, LogLevel.DEBUGGING_LOG); 
		
		
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
		return modelTarget.PredictInstance(Ut, observedTargetIndices);  
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
		for(int iter = 0; iter<maxEpocs ; iter++)
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
			
			// update the target relation
			// first the weights
			modelTarget.Train(U, trainTargets, observedTargetIndices);
			
			// learn the targets loss
			int i=0,l=0;
			double grad = 0, kernelGrad = 0;
			// then the latent predictors
			for(int iIndex = 0; iIndex < observedTargetIndices.length; iIndex++)
			{
				for(int lIndex = 0; lIndex < observedTargetIndices.length; lIndex++)
				{
					i = observedTargetIndices[iIndex];
					l = observedTargetIndices[lIndex];
					
					for(int k = 0; k < D; k++)
					{
						// update U(i,k)
						kernelGrad = ComputeKernelGradient(i, l, i, k);
						
						if( kernelGrad != 0)
						{
							grad = 0.5*modelTarget.alphas[iIndex] 
								* modelTarget.alphas[lIndex]
									* kernelGrad
									- recM*lambda*U.get(i,k);
							
							U.set(i, k,  U.get(i,k) + eta*grad);
						}
						
						// update U(l,k)
						kernelGrad = ComputeKernelGradient(i, l, l, k);
						
						if( kernelGrad != 0)
						{
							grad = 0.5*modelTarget.alphas[iIndex] 
									* modelTarget.alphas[lIndex]
										* kernelGrad 
										- recM*lambda*U.get(l,k);
							
							U.set(l, k,  U.get(l,k) + eta*grad);
						}
						
					}
				}
			}
			
			double trainMAE = PredictTrainSet();
			
			// if nan detected then re-start from beginning with a smaller learn rate 
			if( Double.isNaN( trainMAE ) ) 
			{
				iter = 0;
				eta /= 2;
				Initialize(trainPredictors, theTrainTargets, testPredictors);
				continue;
			}
			
			// the error per loss
			Logging.println(
					"Iter " + iter + 
					", loss: " + PredictorsLoss() + 
					", trainMSE=" + trainMAE + 
					", testMSE=" + PredictTestSet(testTargets), LogLevel.DEBUGGING_LOG);  
		}
	}
	
	public double ComputeKernelGradient(int i, int l, int r, int k)
	{
		double grad = 0;
		
		if( kernel.type == KernelType.Linear)
		{
			if( r == i )
				grad = U.get(l, k);
			else if(r == l)
				grad = U.get(i, k);	
		}
		else if( kernel.type == KernelType.Polynomial)
		{
			if( r == i || r == l )
			{
				grad = kernel.degree * 
						Utilities.StatisticalUtilities.Power(U.RowDotProduct(i, l)+1, kernel.degree-1); 
				
				if( r == i )
					grad *= U.get(l, k);
				else if(r == l)
					grad *= U.get(i, k);
			}			
		}

		else if( kernel.type == KernelType.Gaussian)
		{
			if( r == i || r == l) 
			{
				grad = (r == i) ? 1 : -1;
				
				double kerVal = Math.exp(-kernel.EuclideanDistance(U.getRow(i), U.getRow(l))/kernel.sig2);
				
				grad *= -2*((U.get(i,k) - U.get(l,k))/kernel.sig2) * kerVal;  
				
				//System.out.println(grad + ", diff: " + (U.get(i,k) - U.get(l,k)) + ", kernel" + kerVal + ", sig2" + kernel.sig2);
				
			}
		}

		else if( kernel.type == KernelType.Gaussian)
		{
			
		}
		
		return grad;
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
		
		for(int iter = 0; iter < maxEpocs; iter++)
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
