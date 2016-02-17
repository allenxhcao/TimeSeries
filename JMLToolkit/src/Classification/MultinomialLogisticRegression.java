package Classification;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class MultinomialLogisticRegression 
{
	// the number of classes
	public int C;
	// the number of training instances
	public int NTrain;
	// the number of training instances
	public int NTest;
	// the number of features
	public int M;
	
	// the training predictors X and targets Y
	public double [][] XTrain;
	public double [][] YTrain;
	// the testing predictors X and targets Y
	public double [][] XTest;
	public double [][] YTest;
	
	// the bias coefficients alpha, one per class
	double [] alpha;
	// the linear interaction coefficients, per class and per feature
	double [][] beta;
	
	// regularization hyper-parameters 
	public double lambdaBeta;
	
	// the learning rate and the number of iterations
	public double eta;
	public int maxIter;
	
	// number of latent dimensions for the feature interactions
	int K;
	
	// a random number generator
	Random rand = new Random(); 
	
	public MultinomialLogisticRegression()
	{
		
	}
	
	public void LoadTrainFile(String filePath)
    {
		System.out.println("Started loading training file: " + filePath); 
		
		XTrain = new double[NTrain][M];
		// initialize the targets
		YTrain = new double[NTrain][C];
		for(int n=0; n<NTrain; n++)
			for(int c=0; c<C; c++)
				YTrain[n][c] = 0;
		
		try
        {
			// a line number reader created from the file path
			LineNumberReader lr = new LineNumberReader(
					new InputStreamReader(
							new BufferedInputStream(
									new FileInputStream(filePath))));
	        
			String delimiters = "\t ,;";
	        
	        String line = null;
	        
	        // index of the instances
	        int n = 0;
	        
	        while ( (line = lr.readLine()) != null )
	        {
	        	try
	        	{
		        	// read tokens and write into the 
		        	StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
		        	// index of the features
		        	int m = 0;
		        	// read all tokens
		        	while( tokenizer.hasMoreTokens() )
		        	{
		        		String token = tokenizer.nextToken();
		        		
		        		// set the predictor, or the target
		        		if (m < M)
		        			XTrain[n][m] = Double.parseDouble( token );
		        		else
		        			YTrain[n][ (Integer.parseInt(token)-1) ] = 1.0;
		        		
		        		m++;		        		
		        	}
		        	
		        	//data.add( instanceVariables );
	        	}
	        	catch(Exception exc)
	        	{
	        		Logging.println("Error Loading CSV: " + exc.getMessage(), LogLevel.ERROR_LOG);
	        	}
	        	
	        	// increment instance counter
	        	n++;	        	
	        }
			
	    }
        catch( Exception exc )
        {
        	Logging.println("RegressionDataSet::LoadFile::" + exc.getMessage(), LogLevel.ERROR_LOG);
        }
		
		System.out.println("Finished loading training file: " + filePath); 
        
	}
	
	public void LoadTestFileNoLabels(String filePath)
    {
		System.out.println("Started loading testing file: " + filePath); 
		
		XTest = new double[NTest][M];
		
		try
        {
			// a line number reader created from the file path
			LineNumberReader lr = new LineNumberReader(
					new InputStreamReader(
							new BufferedInputStream(
									new FileInputStream(filePath))));
	        
			String delimiters = "\t ,;";
	        
	        String line = null;
	        
	        // index of the instances
	        int n = 0;
	        
	        while ( (line = lr.readLine()) != null )
	        {
	        	try
	        	{
		        	// read tokens and write into the 
		        	StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
		        	// index of the features
		        	int m = 0;
		        	// read all tokens
		        	while( tokenizer.hasMoreTokens() )
		        	{
		        		String token = tokenizer.nextToken();
		        		XTest[n][m] = Double.parseDouble( token );
		        		m++;		        		
		        	}
	        	}
	        	catch(Exception exc)
	        	{
	        		Logging.println("Error Loading CSV: " + exc.getMessage(), LogLevel.ERROR_LOG);
	        	}
	        	// increment instance counter
	        	n++;	        	
	        }
	    }
        catch( Exception exc )
        {
        	Logging.println("RegressionDataSet::LoadFile::" + exc.getMessage(), LogLevel.ERROR_LOG);
        }
		
		System.out.println("Finished loading testing file: " + filePath); 
        
	}

	
	// load the training and testing files
	public void Initialize(String trainFile, String testFile)
	{
		// load the training and testing file
		LoadTrainFile(trainFile);
		LoadTestFileNoLabels(testFile);
		
		// initialize the parameters
		alpha = new double[C];
		beta = new double[C][M];
		for(int c=0; c<C; c++)
			for(int m=0; m<M; m++)
				beta[c][m] = 2*rand.nextDouble()-1;
		
	}
	
	// compute Y_hat_nc = alpha_c + SUM_m beta_cm * X_nm 
	// given predictors X 
	public double Predict(int n, int c, double [][] X)
	{
		double Y_hat_nc = 0;
		
		Y_hat_nc += alpha[c];
		
		for(int m=0; m<M; m++)
			Y_hat_nc += beta[c][m]*X[n][m];
				
		return Y_hat_nc;
	}
	
	// compute the softmax of a series of 
	public double ComputeLoss(double [][] softMaxTrain )
	{
		double loss = 0;
		
		for(int n=0; n<NTrain; n++)
			for(int c=0; c<C; c++)
				loss += YTrain[n][c]*Math.log( softMaxTrain[n][c] );
			
		loss *= (-1.0/(double)NTrain); 
		
		return loss;
	}
	
	public void Train()
	{
		double [][] errTrain = new double[NTrain][C];
		double [][] softMaxTrain = new double[NTrain][C];
		double sumSoftMaxDenominator_n = 0;
		// temporary variables for the gradient
		double grad_alpha_c = 0;
		double grad_beta_cm = 0;
				
		for(int iter=0; iter<maxIter; iter++)
		{
			// update the parameters
			for(int n=0; n<NTrain; n++)
			{
				// first compute the predicted targets powered to exp
				// and collect the total denominator
				sumSoftMaxDenominator_n = 0;
				for(int c=0; c<C; c++)
				{
					softMaxTrain[n][c] = Math.exp( Predict(n, c, XTrain) );  
					sumSoftMaxDenominator_n += softMaxTrain[n][c];
				}
				// then normalize the soft max to zero-one
				// and then compute the error to ground truth softMax_nc - Y_nc
				for(int c=0; c<C; c++)
				{ 
					softMaxTrain[n][c] /= sumSoftMaxDenominator_n;
					errTrain[n][c] = softMaxTrain[n][c] - YTrain[n][c];  
				} 
			}
			// then update the parameters
			for(int c=0; c<C; c++)
			{
				// update the bias parameters
				// compute the gradient of the loss w.r.t alpha_c
				grad_alpha_c = 0;
				for(int n=0; n<NTrain; n++)
					grad_alpha_c +=	errTrain[n][c];
				//update the alpha_c
				alpha[c] -= eta*grad_alpha_c; 
				
				// update the linear and pairwise interaction parameters 
				for(int m=0; m<M; m++)
				{
					// compute the gradient of the loss w.r.t beta_cm, plus regularization
					grad_beta_cm = 0;
					for(int n=0; n<NTrain; n++)
						grad_beta_cm +=	errTrain[n][c]*XTrain[n][m];
					
					//update the alpha_c 
					beta[c][m] -= eta*grad_beta_cm;
				}
			}				
			
			
			// compute and print the loss
			double trainLoss = ComputeLoss(softMaxTrain);
			System.out.println("Iteration=" + iter + ", trainLoss=" + trainLoss);
		}
		
	}
	
	public void PredictTestTargets()
	{
		
		double [][] softMaxTest = new double[NTest][C];
		double sumSoftMaxDenominator_n = 0;
		
		for(int n=0; n<NTest; n++)
		{
			System.out.print((n+1)+","); 
			
			sumSoftMaxDenominator_n = 0;
			for(int c=0; c<C; c++)
			{
				softMaxTest[n][c] = Math.exp( Predict(n, c, XTest) );  
				sumSoftMaxDenominator_n += softMaxTest[n][c];
			}
			// then normalize the soft max to zero-one
			// and then compute the error to ground truth softMax_nc - Y_nc
			for(int c=0; c<C; c++)
			{
				softMaxTest[n][c] /= sumSoftMaxDenominator_n;
				
				//print the probability
				System.out.print(softMaxTest[n][c]);
				if(c<C-1)
					System.out.print(",");
				else
					System.out.println("");
			}
		}
	}
	
	public static void main(String [] args)
	{
		MultinomialLogisticRegression fnmlr = new MultinomialLogisticRegression();
		fnmlr.lambdaBeta = 0.0001;
		
		// set the data size
		fnmlr.NTrain = 61878;
		fnmlr.NTest = 144368; 
		fnmlr.M = 93; 
		fnmlr.C = 9; 
		
		// set the learning rate
		fnmlr.eta = 0.00001;
		fnmlr.maxIter = 10000;
		
		// load the data initialize the model 
		fnmlr.Initialize("E:\\Data\\classification\\otto\\train.csv", "E:\\Data\\classification\\otto\\test.csv");
		// train the model
		fnmlr.Train();
		
		// predict test probabilities
		//fnmlr.PredictTestTargets();
		
		//Logging.print( fnmlr.XTrain[34000], LogLevel.DEBUGGING_LOG);
		//Logging.println( "" );
		//Logging.print( fnmlr.YTrain[34000], LogLevel.DEBUGGING_LOG);
		//Logging.println( "" );
		
		
		
	}
	

}
