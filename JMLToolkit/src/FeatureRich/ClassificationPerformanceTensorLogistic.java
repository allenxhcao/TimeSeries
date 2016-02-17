package FeatureRich;

import java.io.PrintStream;
import java.util.Random;

import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;

public class ClassificationPerformanceTensorLogistic 
{

	// the classification accuracy measurements
	public double trainLoss;
	public double trainMCR;
	public double testLoss;
	public double testMCR;
	
	public int ITrain;
	public int ITest;
	public int [] K;
	public int R; 
	public int C;
	
	public double [][][] F;
	public double [][] Y;
	public double [][][] W;
	public double [] biasW;
	
	Random rand = new Random();
	
	// classify frequencies F in R^{I x R x K} 
	public void ComputeClassificationAccuracy(double [][][] FInput, double [][] YInput, double [][][] WInput, double [] biasWInput)
	{
		// initialize the 
		this.F = FInput; 
		this.W = WInput;
		this.Y = YInput;
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
		
		for(int r = 0; r < R; r++)
			for(int k = 0; k < K[r]; k++)
				y_hat_ic += F[i][r][k]*W[c][r][k];
		
		return y_hat_ic;
	}
	

	// compute the MCR on the test set
	public double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			double max_Y_hat_ic = 0;
			int label_i = rand.nextInt(C);
			
			double [] Y_hat_i = new double[C];
			
			for(int c = 0; c < C; c++)
			{
				Y_hat_i[c] = Sigmoid.Calculate( EstimateTarget(i, c) ); 
				
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
			double max_Y_hat_ic = 0; 
			int label_i = rand.nextInt(C); 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = Sigmoid.Calculate( EstimateTarget(i, c) ); 
				
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
		
	// compute the MCR on the test set
	public void PrintEstimatedTestLabels(PrintStream ps) 
	{
		for(int i = ITrain; i < ITrain+ITest; i++)
		{
			double max_Y_hat_ic = 0;
			int label_i = rand.nextInt(C); 
			
			for(int c = 0; c < C; c++)
			{
				double Y_hat_ic = Sigmoid.Calculate( EstimateTarget(i, c) ); 
				
				if(Y_hat_ic > max_Y_hat_ic)
				{
					max_Y_hat_ic = Y_hat_ic; 
					label_i = c;
				}
			}
			
			ps.println( (i-ITrain) + ", " + label_i);
		}
	}
		 
	
	// compute the accuracy loss of instance i according to the 
	// smooth hinge loss 
	public double AccuracyLoss(int i, int c)
	{
		double sig_Y_hat_ic = Sigmoid.Calculate( EstimateTarget(i, c) );
		
		return -Y[i][c]*Math.log(  sig_Y_hat_ic ) - (1-Y[i][c])*Math.log( 1 - sig_Y_hat_ic );		 
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
		
		return accuracyLoss/(double)ITrain;
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
		
		return accuracyLoss/(double)ITest;
	} 	
	
	
}
