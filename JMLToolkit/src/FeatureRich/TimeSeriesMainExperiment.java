package FeatureRich;

import java.io.File;

import DataStructures.DataSet;
import FeatureRich.SupervisedCodebookLearning.LossTypes;
import FeatureRich.SupervisedCodebookLearning.SimilarityTypes;

public class TimeSeriesMainExperiment 
{

	public static void main(String [] args)
	{

		// set the arguments if not run from command line
		if (args.length == 0) {
			args = new String[] {
					"datasetsFolder=/run/media/josif/E/Data/classification/timeseries/",
					"dsName=Gun_Point", 
					"fold=1", 
					"eta=1", 
					"lambda=0.001", 
					"gamma=-1", 
					"maxIters=30000" 
			};
		}
		
		// hyper-parameters
		String datasetsFolder = "", dsName = "", fold = ""; 
		int maxIters = 0;
		double eta = 0, lambda = 0, gamma = 0;

		// parse the command line arguments and initialize the hyper-parameters
		for (String arg : args) {
			String[] argTokens = arg.split("=");
		
			if (argTokens[0].compareTo("datasetsFolder") == 0)
				datasetsFolder = argTokens[1];
			else if (argTokens[0].compareTo("dsName") == 0)
				dsName = argTokens[1];
			else if (argTokens[0].compareTo("fold") == 0)
				fold = argTokens[1];
			else if (argTokens[0].compareTo("maxIters") == 0)
				maxIters = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("eta") == 0)
				eta = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("lambda") == 0)
				lambda = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("gamma") == 0)
				gamma = Double.parseDouble(argTokens[1]);
		}		
		
		// load the dataset
		DataSet dsTrain = new DataSet();
		dsTrain.LoadDataSetFile(new File(datasetsFolder + "/" + dsName + "/folds/" + fold + "/" + dsName + "_TRAIN")); 
		dsTrain.NormalizeDatasetInstances();
		dsTrain.ReadNominalTargets();
		
		DataSet dsTest = new DataSet();
		dsTest.LoadDataSetFile(new File(datasetsFolder + "/" + dsName + "/folds/" + fold + "/" + dsName + "_TEST"));  
		dsTest.NormalizeDatasetInstances();
		dsTest.ReadNominalTargets();
		
		if( dsTrain.nominalLabels.size() != dsTest.nominalLabels.size() )
			System.out.println("Number of labels don't match in train and test splits");
		
		SupervisedCodebookLearning scl = new SupervisedCodebookLearning(); 
		
		int seriesLength = dsTrain.numFeatures;
		
		
		scl.NTrain = dsTrain.GetNumInstances();
		scl.NTest = dsTest.GetNumInstances(); 
		
		scl.Q = seriesLength;
		scl.D = seriesLength/2; 
		
		scl.gamma = gamma;
		scl.lambdaW = lambda;  
		scl.eta = eta; 
		scl.maxIters = maxIters;
		
		scl.C = dsTrain.nominalLabels.size(); 
		
		scl.lossType = LossTypes.HINGE;
		scl.similarityType = SimilarityTypes.GAUSSIAN;
		
		// initialize the model
		scl.Initialize(dsTrain, dsTest);				
		
		

		
		// learn the codebook and the classification weights
		scl.Learn();
	}
}
