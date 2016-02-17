package FeatureRich;

import java.io.File;
import java.util.ArrayList;

import DataStructures.DataSet;
import DataStructures.Matrix;

public class TimeSeriesMain 
{

	// the main execution of the program
	public static void main(String [] args)
	{
		// in case ones wants to run it from an IDE like eclipse 
		// then the command line parameters can be set as
		if (args.length == 0) {
			String dir = "E:\\Data\\classification\\timeseries\\",
			//String dir = "/run/media/josif/E/Data/classification/timeseries/",
			ds = "ECG200"; 

			String sp = File.separator, fold = "default";
			
		
			args = new String[] {  
				"trainSet=" + dir + ds + sp + "folds" + sp + fold + sp  
						+ ds + "_TRAIN",  
				"testSet=" + dir + ds + sp + "folds" + sp + fold + sp  
						+ ds + "_TEST",  
 				"lambdaW=0.0001", 
 				"lambdaP=0.0001", 
				"maxEpochs=1000",  
				"gamma=-10",  
				"delta=0.2", 
				//"K=20", 
				"L=0.2",  
				"R=4",  
				"eta=0.1" 
				}; 
		}			

		// values of hyperparameters
		double eta = -1, lambdaW = -1, lambdaP = -1, gamma = -1, L = -1, K = -1, delta = 0.5;
		int maxEpochs = -1, R = -1;
		String trainSetPath = "", testSetPath = "";
		
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
			else if (argTokens[0].compareTo("delta") == 0)
				delta = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("maxEpochs") == 0)
				maxEpochs = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("R") == 0) 
				R = Integer.parseInt(argTokens[1]); 
			else if (argTokens[0].compareTo("L") == 0) 
				L = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("K") == 0) 
				K = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("trainSet") == 0) 
				trainSetPath = argTokens[1]; 
			else if (argTokens[0].compareTo("testSet") == 0) 
				testSetPath = argTokens[1]; 
		}
		
		// set predefined parameters if none set
		if(R < 0) R = 4;
		if(L < 0) L = 0.15;
		if(eta < 0) eta = 0.01;
		if(gamma > 0) gamma = -1; 
		if(maxEpochs < 0) maxEpochs = 10000;
		
		long startTime = System.currentTimeMillis();
		 
		// load dataset
		DataSet trainSet = new DataSet();
		trainSet.LoadDataSetFile(new File(trainSetPath));
		DataSet testSet = new DataSet();
		testSet.LoadDataSetFile(new File(testSetPath));

		// normalize the data instance
		trainSet.NormalizeDatasetInstances();
		testSet.NormalizeDatasetInstances();
		
		// predictor variables T
		Matrix T = new Matrix();
        T.LoadDatasetFeatures(trainSet, false);
        T.LoadDatasetFeatures(testSet, true);
        // outcome variable O
        Matrix O = new Matrix();
        O.LoadDatasetLabels(trainSet, false);
        O.LoadDatasetLabels(testSet, true);

        LearnSupervisedBagOfWordsMultiScale lsBoW = new LearnSupervisedBagOfWordsMultiScale();   
        // initialize the sizes of data structures
        lsBoW.ITrain = trainSet.GetNumInstances();  
        lsBoW.ITest = testSet.GetNumInstances();
        
        lsBoW.labels = O;
        // set the learn rate and the number of iterations
        lsBoW.maxIter = maxEpochs;
        // the scales of the patterns
        lsBoW.R = R;        
        // set the size of the cascade patterns 
        lsBoW.L = new int[lsBoW.R];
        int initL = (int)(L*T.getDimColumns()); 
        for(int r=0; r<R; r++)
        	lsBoW.L[r] = initL*(r+1); 
        
        // convert the time series dataset into partitioned segments
        TimeSeriesTransformations tst = new TimeSeriesTransformations();  
        lsBoW.S = tst.NormalizationTransformations(T.cells, R, lsBoW.L, delta);   
        
        //lsBoW.S = tst.NormalizedOriginalAndDerivativeTransformations(T.cells, lsBoW.L, delta); 
        
        // set the regularization parameter
        lsBoW.lambdaW = lambdaW; 
        lsBoW.lambdaP = lambdaP; 
        lsBoW.eta = eta; 
        lsBoW.gamma = gamma;   
        
        trainSet.ReadNominalTargets();  
        lsBoW.nominalLabels =  new ArrayList<Double>(trainSet.nominalLabels);  
        
        // learn the model
        lsBoW.Learn(); 
        
        // learn the local convolutions
        long endTime = System.currentTimeMillis(); 
        
		System.out.println( 
				String.valueOf(lsBoW.minTestMCR)  + " " + String.valueOf(lsBoW.minTrainLoss) + " " 
				+ "L=" + L	+ " "  
				+ "R=" + lsBoW.R + " "
				+ "lW=" + lambdaW + " "
				+ "gamma=" + gamma + " " 
				+ "eta=" + eta + " " 
				+ "maxEpochs="+ maxEpochs + " " 
				+ "time="+ (endTime-startTime) 
				); 
	}


}
