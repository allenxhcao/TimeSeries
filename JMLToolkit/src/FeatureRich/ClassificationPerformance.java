package FeatureRich;

import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;

public class ClassificationPerformance 
{

	// the classification accuracy measurements
	public double trainLoss;
	public double trainMCR;
	public double testLoss;
	public double testMCR;
	
	public int ITrain;
	public int ITest;
	public int K;
	public int C;
	
	public double [][] F;
	public double [][] Y;
	public double [][] W;
	public double [] biasW;
	
	// classify frequencies F in R^{I x R x K} 
	public void ComputeClassificationAccuracy(double [][][] FInput, int R, double [][] YInput, double [][][] WInput, double [] biasWInput)
	{
		// initialize the 
		this.F = new double[ITrain+ITest][R*K]; 
		this.W = new double[C][R*K]; 
		
		// convert the frequencies vector to 2-dimensions
		for(int i=0; i<(ITrain+ITest); i++)
			for(int r=0; r<R; r++)
				for(int k=0; k<K; k++)
					 this.F[i][r*K+k] = FInput[i][r][k]; 
		
		// convert the weights tensor to matrix
		for(int c=0; c<C; c++)
			for(int r=0; r<R; r++)
				for(int k=0; k<K; k++)
					this.W[c][r*K+k] = WInput[c][r][k];  
		
		
		this.Y = YInput;
		
		this.biasW = biasWInput;
		
		trainLoss = AccuracyLossTrainSet();
		testLoss = AccuracyLossTestSet();
		
		trainMCR = GetMCRTrainSet();
		testMCR = GetMCRTestSet();
	}
	
	// compute the classification accuracy of predictors 
	public void ComputeClassificationAccuracy(double [][] FInput, double [][] YInput, double [][] WInput, double [] biasWInput)
	{
		this.F = FInput;
		this.Y = YInput;
		this.W = WInput;
		this.biasW = biasWInput;
		
		trainLoss = AccuracyLossTrainSet();
		testLoss = AccuracyLossTestSet();
		
		trainMCR = GetMCRTrainSet();
		testMCR = GetMCRTestSet();
	}
	
	
	// compute the estimated target variable
	public double EstimateTarget(int i, int c)
	{
		double y_hat_ic = biasW[c];
		
		for(int k = 0; k < K; k++)
			y_hat_ic += F[i][k]*W[c][k];
		
		return y_hat_ic;
	}
	

	// compute the MCR on the test set
	public double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			double max_Y_hat_ic = -Double.MAX_VALUE;
			int label_i = -1; 
			
			double [] Y_hat_i = new double[C];
			
			for(int c = 0; c < C; c++)
			{
				Y_hat_i[c] = EstimateTarget(i, c);
				
				if( Y_hat_i[c] > max_Y_hat_ic)
				{
					max_Y_hat_ic = Y_hat_i[c]; 
					label_i = c;
				}
				
				//System.out.println("\n"+Y_hat_i[c] + " " + max_Y_hat_ic);
				
			}
			
			//System.out.println("\n"+label_i + " " + max_Y_hat_ic);
			
			if( Y[i][label_i] != 1.0 ) 
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
			double max_Y_hat_ic = -Double.MAX_VALUE;
			int label_i = -1; 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = EstimateTarget(i, c); 
				
				if(Y_hat_ic > max_Y_hat_ic)
				{
					max_Y_hat_ic = Y_hat_ic; 
					label_i = c;
				}
			}
			
			if( Y[i][label_i] != 1.0 ) 
				numErrors++;
		}
		
		return (double)numErrors/(double)ITest; 
	}
	
	// compute the accuracy loss of instance i according to the 
	// smooth hinge loss 
	public double AccuracyLoss(int i, int c)
	{
		double Y_hat_ic = EstimateTarget(i, c);
		double loss = 0;
		
		double z = Y[i][c]*Y_hat_ic;
		
		if( z <= 0 ) loss = 0.5 - z;
		else if( 0 < z && z < 1) loss = 0.5*(1-z)*(1-z);
		else loss = 0;
	
		return loss; 
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
	
	
}
