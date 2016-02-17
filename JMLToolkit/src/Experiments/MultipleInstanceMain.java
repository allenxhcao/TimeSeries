package Experiments;

import Classification.*;
import Utilities.*;
import Clustering.KMedoids;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.Matrix;
import MatrixFactorization.CollaborativeImputation;
import MatrixFactorization.FactorizationsCache;
import MatrixFactorization.NonlinearlySupervisedMF;
import TimeSeries.*;
import Utilities.ExpectationMaximization;
import Utilities.Logging;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintStream;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class MultipleInstanceMain {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		if (args.length == 0) 
		{
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

			String dir = "C:\\Users\\josif\\Documents\\Data\\classification\\binary\\",
		    ds = "genderPrediction",
			foldNo = "3";  

			String sp = File.separator;
			
			args = new String[] { 
				"fold=" + foldNo,   
				"model=nmi", 
				"runMode=test", 
				"trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp 
						+ ds + "_TRAIN", 
				"validationSet=" + dir + ds + sp + "folds" + sp + foldNo 
						+ sp + ds + "_VALIDATION", 
				"testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp 
						+ ds + "_TEST", 
				"factorizationsCacheFolder=/home/josif/Documents/factorizations_cache", 
				"etaX=0.01", 
				"etaY=0.01", 
				"lambdaU=0.0001",  
				"lambdaV=0.001", 
				"lambdaAlpha=0.0001",   
				"latentDimensionsRatio=0.01",
				"maxEpochs=11", 
				"gamma=0.1" 
			};
			
		}

		if (args.length > 0) {
			// set model values
			String model = "", runMode = "";
			String trainSetPath = "", validationSetPath = "", testSetPath = "", fold = "", factorizationsCacheFolder = "", transformationFieldsFolder = "";
			double etaX = 0.01, etaY = 0.01, etaL = 0.01, lambdaU=0, lambdaV=0, lambdaAlpha = 0;
			double latentDimensionsRatio = 0, gamma=0.1;
			
			int k = 0, maxEpochs = 30;

			// read and parse parameters
			for (String arg : args) {
				String[] argTokens = arg.split("=");

				if (argTokens[0].compareTo("model") == 0)
					model = argTokens[1];
				else if (argTokens[0].compareTo("runMode") == 0)
					runMode = argTokens[1];
				else if (argTokens[0].compareTo("trainSet") == 0)
					trainSetPath = argTokens[1];
				else if (argTokens[0].compareTo("validationSet") == 0) 
					validationSetPath = argTokens[1];
				else if (argTokens[0].compareTo("testSet") == 0)
					testSetPath = argTokens[1];
				else if (argTokens[0].compareTo("factorizationsCacheFolder") == 0)
					factorizationsCacheFolder = argTokens[1];
				else if (argTokens[0].compareTo("fold") == 0)
					fold = argTokens[1];
				else if (argTokens[0].compareTo("latentDimensionsRatio") == 0) 
					latentDimensionsRatio = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("etaX") == 0) 
					etaX = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("etaY") == 0) 
					etaY = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("etaL") == 0) 
					etaL = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaU") == 0) 
					lambdaU = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaV") == 0)
					lambdaV = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaAlpha") == 0)
					lambdaAlpha = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("gamma") == 0)
					gamma = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("maxEpochs") == 0)
					maxEpochs = Integer.parseInt(argTokens[1]);
			}

			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

			// load the train, validation and test sets
			DataSet trainSet = new DataSet();
			trainSet.LoadDataSetFile(new File(trainSetPath));
			DataSet validationSet = new DataSet();
			validationSet.LoadDataSetFile(new File(validationSetPath));
			DataSet testSet = new DataSet();
			testSet.LoadDataSetFile(new File(testSetPath));

			// normalize the data instance
			trainSet.NormalizeDatasetInstances();
			testSet.NormalizeDatasetInstances();
			validationSet.NormalizeDatasetInstances();

			// of k was not initialized from parameters then initialize it from
			// the ratio
	
			k = (int) (trainSet.numFeatures * latentDimensionsRatio);

			String dsName = new File(trainSetPath).getName().split("_TRAIN")[0];

				if (model.compareTo("klr") == 0) {
				try {
					long start = System.currentTimeMillis();
					
					DataSet finalTrainSet = null;
					DataSet finalTestSet = null;

					if (runMode.compareTo("test") == 0) {
						finalTrainSet = new DataSet(trainSet);
						finalTrainSet.AppendDataSet(validationSet);
						finalTestSet = testSet;

					} else {
						finalTrainSet = trainSet;
						finalTestSet = validationSet;
					}
					
					KernelLogisticRegression klr = new KernelLogisticRegression();
					klr.eta = etaY;
					klr.lambda = lambdaAlpha; 
					klr.gamma = gamma;
					klr.maxEpochs = maxEpochs;

			        Matrix trainPredictors = new Matrix();
			        trainPredictors.LoadDatasetFeatures(finalTrainSet, false);
			        Matrix trainLabels = new Matrix();
			        trainLabels.LoadDatasetLabels(finalTrainSet, false);
			        
			        Matrix testPredictors = new Matrix();
			        testPredictors.LoadDatasetFeatures(finalTestSet, false);
			        Matrix testLabels = new Matrix();
			        testLabels.LoadDatasetLabels(finalTestSet, false); 
			        
					klr.Train(trainPredictors, trainLabels);
					
					klr.Test(testPredictors, testLabels);
					
					
					long end = System.currentTimeMillis();  
					double elapsedTime = end - start; 

					

				} catch (Exception exc) {
					exc.printStackTrace();
				}
			}
				
				else if (model.compareTo("nmi") == 0) {
				try {
					long start = System.currentTimeMillis();
					
					DataSet finalTrainSet = null;
					DataSet finalTestSet = null;

					if (runMode.compareTo("test") == 0) {
						finalTrainSet = new DataSet(trainSet);
						finalTrainSet.AppendDataSet(validationSet);
						finalTestSet = testSet;

					} else {
						finalTrainSet = trainSet;
						finalTestSet = validationSet;
					}
					
					NonlinearMultipleInstance nmi = new NonlinearMultipleInstance(k);
					
					nmi.etaX = etaX;
					nmi.etaY = etaY;
					nmi.etaL = etaL;
					
					nmi.lambdaU = lambdaU;
					nmi.lambdaV = lambdaV;
					nmi.lambdaAlpha = lambdaAlpha;
					
					nmi.gamma = gamma;
					
					nmi.maxEpochs = maxEpochs;

			        Matrix predictors = new Matrix();
			        predictors.LoadDatasetFeatures(finalTrainSet, false);
			        predictors.LoadDatasetFeatures(finalTestSet, true);
			        
			        Matrix labels = new Matrix();
			        labels.LoadDatasetLabels(finalTrainSet, false);
			        labels.LoadDatasetLabels(finalTestSet, true); 
			        
			        System.out.println(predictors.getDimRows() + ", " +  labels.getDimRows());
			        

			        // initialize the training and total instances fields
			        nmi.numTrainInstances = finalTrainSet.instances.size();
			        nmi.numTotalInstances = finalTrainSet.instances.size() + finalTestSet.instances.size();
			        nmi.numFeatures = finalTestSet.numFeatures; 
			        		
			        nmi.Decompose(predictors, labels); 
			        
					long end = System.currentTimeMillis();  
					double elapsedTime = end - start; 

					

				} catch (Exception exc) {
					exc.printStackTrace();
				}
			} 
		}
			

	}

}
