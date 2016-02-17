package FeatureRich;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.SortedSet;
import java.util.StringTokenizer;
import java.util.TreeSet;

import Utilities.Logging;
import Utilities.Logging.LogLevel;


public class FeatureRichMain  
{
	
	
	public static void main(String [] args)
	{
		
		// in case ones wants to run it from an IDE like eclipse 
		// then the command-line parameters can be set as
		if (args.length == 0) { 
			
			args = new String[] { 
				//"trainSet=H:\\bci-eeg-challenge-2014\\train_exported.txt", 
				//"testSet=H:\\bci-eeg-challenge-2014\\test_exported.txt", 
				"trainSet=E:\\Data\\multimedia\\image\\DataScienceBowl\\trainAugmentedSandbox\\sandboxTrain_hog_exported.txt",
				"testSet=E:\\Data\\multimedia\\image\\DataScienceBowl\\trainAugmentedSandbox\\sandboxTest_hog_exported.txt",
				//"trainSet=E:\\Data\\multimedia\\image\\DataScienceBowl\\trainAugmented_TRAIN_hog_exported.txt",
				//"testSet=E:\\Data\\multimedia\\image\\DataScienceBowl\\trainAugmented_TEST_hog_exported.txt",
				"lambdaW=0.00001",
 				"lambdaP=0.00001",
 				"gamma=-100",
 				"K=200", 
				"eta=0.1", 
				"maxEpochs=300" 
				};
		} 
		
		// values of hyperparameters
		double eta = -1, lambdaW = -1, lambdaP = -1, gamma = -1;
		int maxEpochs = -1, K = -1;
		String trainSetPath = "", testSetPath = "";
		boolean isNonlinear = false;
		
		// read and parse parameters
		for (String arg : args) {
			String[] argTokens = arg.split("=");
			
			if (argTokens[0].compareTo("eta") == 0) 
				eta = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("lambdaW") == 0)
				lambdaW = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("lambdaP") == 0)
				lambdaP = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("gamma") == 0)
				gamma = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("maxEpochs") == 0)
				maxEpochs = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("K") == 0) 
				K = Integer.parseInt(argTokens[1]); 
			else if (argTokens[0].compareTo("trainSet") == 0) 
				trainSetPath = argTokens[1]; 
			else if (argTokens[0].compareTo("testSet") == 0) 
				testSetPath = argTokens[1];
			else if (argTokens[0].compareTo("nonlinear") == 0) 
				isNonlinear = Integer.parseInt(argTokens[1]) == 1; 
		}
		
		long startTime = System.currentTimeMillis();
		
		FeatureRichDataset trainData = new FeatureRichDataset();		
		trainData.LoadDataset(trainSetPath);
		
		FeatureRichDataset testData = new FeatureRichDataset(); 
		testData.LoadDataset(testSetPath);
		
		LearnSupervisedShapeletLogistic lsbwms = new LearnSupervisedShapeletLogistic();
 
		lsbwms.C = trainData.C;
		lsbwms.ITrain = trainData.I; 
		lsbwms.ITest = testData.I; 
		// set scales and lengths at different scales 
		lsbwms.R = trainData.R; 
		lsbwms.L = trainData.L;
		
		lsbwms.S = new double[lsbwms.ITrain+lsbwms.ITest][lsbwms.R][][];
		lsbwms.Y = new double[lsbwms.ITrain+lsbwms.ITest][lsbwms.C];
		
		for(int i=0; i<lsbwms.ITrain; i++)
		{
			lsbwms.Y[i] = trainData.Y[i].clone();
			lsbwms.S[i] = trainData.S[i].clone();	
		}
		for(int i=lsbwms.ITrain; i<lsbwms.ITrain+lsbwms.ITest; i++)
		{
			lsbwms.Y[i] = testData.Y[i-lsbwms.ITrain].clone(); 
			lsbwms.S[i] = testData.S[i-lsbwms.ITrain].clone(); 
		}
		
		lsbwms.K = new int[lsbwms.R];
		for(int r = 0; r < lsbwms.R; r++) 
			lsbwms.K[r] = K; 
		
		lsbwms.eta = eta;
		lsbwms.maxIter = maxEpochs; 
		lsbwms.lambdaW = lambdaW; 
		lsbwms.lambdaP = lambdaP;
		lsbwms.gamma = gamma; 
				
		double err=lsbwms.Learn();   
		
		long endTime = System.currentTimeMillis(); 
		
		System.out.println( 
				String.valueOf(lsbwms.cp.testMCR) + " "  
				+ String.valueOf(lsbwms.cp.trainLoss) + " "
				+ "K=" + K + " "
				+ "lW=" + lambdaW + " "
				+ "lP=" + lambdaP + " "
				+ "gamma=" + gamma + " " 
				+ "eta=" + eta + " " 
				+ "maxEpochs="+ maxEpochs + " " 
				+ "time="+ (endTime-startTime) 
				); 
		
		
	} 
	
}
