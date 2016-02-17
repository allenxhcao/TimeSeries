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
import Utilities.Logging.LogLevel;

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

public class SupervisedDecompositionMain {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		if (args.length == 0) 
		{
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

			String dir = "C:\\Users\\josif\\Documents\\Data\\classification\\longts\\",
			ds = "ratbp",    
			foldNo = "3";   

			String sp = File.separator; 

			args = new String[] { 
				"fold=" + foldNo,   
				"model=tsmrf",  
				"runMode=test",  
				"trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp  
						+ ds + "_TRAIN",  
				"validationSet=" + dir + ds + sp + "folds" + sp + foldNo  
						+ sp + ds + "_VALIDATION",  
				"testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp  
						+ ds + "_TEST", 
				"eta=0.0001", 
				"maxEpochs=10000", 
				"lambda1=0.00001", // ls 
				"lambda2=0.00001", // lp
				"lambda3=0.00001", // lw
				"alpha1=1.0", // h loss
				"alpha2=1.0", // y loss 
				"n=100",
				"w=8",
				"a=4",
				"degree=6",  
				"latentDimensionsRatio=0.03",  
				
			};

		}

		if (args.length > 0) {
			// set model values
			String model = "", runMode = "";
			String trainSetPath = "", validationSetPath = "", testSetPath = "", fold = "", factorizationsCacheFolder = "", transformationFieldsFolder = "";
			double eta = 0.01, etaX = 0.01, etaY = 0.01, etaL = 0.01, 
				lambda1=0, lambda2=0, lambda3 = 0, lambda4 = 0.0,
				alpha1 = 0.0, alpha3 = 0.0, alpha2 = 0.0, alpha4 = 0.0; 
			double latentDimensionsRatio = 0, gamma=0.1;
			int n = 100, w = 8, a = 4, degree = 3;
			
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
				else if (argTokens[0].compareTo("eta") == 0) 
					eta = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("etaX") == 0) 
					etaX = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("etaY") == 0) 
					etaY = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("etaL") == 0) 
					etaL = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambda1") == 0) 
					lambda1 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambda2") == 0)
					lambda2 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambda3") == 0)
					lambda3 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambda4") == 0)
					lambda4 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("alpha1") == 0)
					alpha1 = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("alpha2") == 0) 
					alpha2 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("alpha3") == 0)
					alpha3 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("alpha4") == 0) 
					alpha4 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("gamma") == 0)
					gamma = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("maxEpochs") == 0)
					maxEpochs = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("n") == 0)
					n = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("w") == 0)
					w = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("a") == 0)
					a = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("degree") == 0)
					degree = Integer.parseInt(argTokens[1]);
			}

			//Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

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
					klr.lambda = lambda3; 
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
					
					NonlinearSupervisedDecomposition nmi = new NonlinearSupervisedDecomposition(k);
					
					nmi.etaX = etaX;
					nmi.etaY = etaY;
					nmi.etaL = etaL;
					
					nmi.lambdaU = lambda1;
					nmi.lambdaV = lambda2;
					nmi.lambdaAlpha = lambda3;
					
					nmi.gamma = gamma;
					
					nmi.maxEpochs = maxEpochs;

			        Matrix predictors = new Matrix();
			        predictors.LoadDatasetFeatures(finalTrainSet, false);
			        predictors.LoadDatasetFeatures(finalTestSet, true);
			        
			        Matrix labels = new Matrix();
			        labels.LoadDatasetLabels(finalTrainSet, false);
			        labels.LoadDatasetLabels(finalTestSet, true); 
			        
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
				else if (model.compareTo("sdl") == 0) {
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
						
						SupervisedDecompositionLaplacian sdl = new SupervisedDecompositionLaplacian(k);
						
						sdl.etaX = etaX;
						sdl.etaY = etaY;
						sdl.etaL = etaL;
						
						sdl.lambdaU = lambda1;
						sdl.lambdaV = lambda2;
						sdl.lambdaW = lambda3;						
						
						sdl.maxEpochs = maxEpochs;

				        Matrix predictors = new Matrix();
				        predictors.LoadDatasetFeatures(finalTrainSet, false);
				        predictors.LoadDatasetFeatures(finalTestSet, true);
				        
				        Matrix labels = new Matrix();
				        labels.LoadDatasetLabels(finalTrainSet, false);
				        labels.LoadDatasetLabels(finalTestSet, true); 
				        

				        
				        // initialize the training and total instances fields
				        sdl.numTrainInstances = finalTrainSet.instances.size();
				        sdl.numTotalInstances = finalTrainSet.instances.size() + finalTestSet.instances.size();
				        sdl.numFeatures = finalTestSet.numFeatures; 
				        		
				        sdl.Decompose(predictors, labels); 
				        
						long end = System.currentTimeMillis();  
						double elapsedTime = end - start; 

						

					} catch (Exception exc) {
						exc.printStackTrace();
					}
				} 
				else if (model.compareTo("nsd") == 0) 
				{
					
					try 
					{
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
						
						trainSet = null;
						validationSet = null;
						testSet = null;
						System.gc();
						
						NarySupervisedDecomposition nsd = new NarySupervisedDecomposition(k);
						
						nsd.alpha = alpha1;
						nsd.eta = eta;						
						nsd.lambdaU = lambda1;
						nsd.lambdaV = lambda2;
						nsd.lambdaW = lambda3;	
						nsd.maxEpochs = maxEpochs;
						
				        Matrix predictors = new Matrix();
				        predictors.LoadDatasetFeatures(finalTrainSet, false);
				        predictors.LoadDatasetFeatures(finalTestSet, true);
				        
				        Matrix labels = new Matrix();
				        labels.LoadDatasetLabels(finalTrainSet, false);
				        labels.LoadDatasetLabels(finalTestSet, true); 
				        
				        finalTrainSet.ReadNominalTargets();
				        
				        // initialize the training and total instances fields
				        nsd.numTrainInstances = finalTrainSet.instances.size();
				        nsd.numTotalInstances = finalTrainSet.instances.size() + finalTestSet.instances.size();
				        nsd.numFeatures = finalTestSet.numFeatures; 
				        nsd.numLabels = finalTrainSet.nominalLabels.size();
				        
				        Logging.println("NTrain=" + nsd.numTrainInstances + ", NTest=" + nsd.numTotalInstances, LogLevel.DEBUGGING_LOG);
				        Logging.println("M_i=" + nsd.numFeatures, LogLevel.DEBUGGING_LOG); 
				        Logging.println("NLabels: " + nsd.numLabels, LogLevel.DEBUGGING_LOG); 
					    Logging.println("D_i=" + k, LogLevel.DEBUGGING_LOG); 
					    
				        finalTrainSet = null;
				        finalTestSet = null;
				        System.gc();
				        
				        double accuracy = nsd.Decompose(predictors, labels); 
				        
						long end = System.currentTimeMillis();  
						double elapsedTime = end - start; 
						
						

					} catch (Exception exc) {
						exc.printStackTrace();
					}
				}
				else if (model.compareTo("bsd") == 0) 
				{
					
					try 
					{
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
						
						trainSet = null;
						validationSet = null;
						testSet = null;
						System.gc();
						
						BinarySupervisedDecomposition bsd = new BinarySupervisedDecomposition(k);
						
						bsd.alpha = alpha1;
						bsd.eta = eta;						
						bsd.lambdaU = lambda1;
						bsd.lambdaV = lambda2;
						bsd.lambdaW = lambda3;	
						bsd.maxEpochs = maxEpochs;
						
				        Matrix predictors = new Matrix();
				        predictors.LoadDatasetFeatures(finalTrainSet, false);
				        predictors.LoadDatasetFeatures(finalTestSet, true);
				        
				        Matrix labels = new Matrix();
				        labels.LoadDatasetLabels(finalTrainSet, false);
				        labels.LoadDatasetLabels(finalTestSet, true); 
				        
				        finalTrainSet.ReadNominalTargets();
				        
				        // initialize the training and total instances fields
				        bsd.numTrainInstances = finalTrainSet.instances.size();
				        bsd.numTotalInstances = finalTrainSet.instances.size() + finalTestSet.instances.size();
				        bsd.numFeatures = finalTestSet.numFeatures; 
				        
				        Logging.println("NTrain=" + bsd.numTrainInstances + ", NTest=" + bsd.numTotalInstances, LogLevel.DEBUGGING_LOG);
				        Logging.println("M_i=" + bsd.numFeatures, LogLevel.DEBUGGING_LOG); 
				        Logging.println("D_i=" + k, LogLevel.DEBUGGING_LOG); 
					    
				        finalTrainSet = null;
				        finalTestSet = null;
				        System.gc();
				        
				        double testError = bsd.Decompose(predictors, labels); 
				        
						long end = System.currentTimeMillis();  
						double elapsedTime = end - start; 
						
						

					} catch (Exception exc) {
						exc.printStackTrace();
					}
				}
				else if (model.compareTo("tsmrf") == 0) 
				{
				
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
					
					trainSet = null;
					validationSet = null;
					testSet = null;
					
					BOPSF bopsf = new BOPSF();
					
					// set the impact weight parameters
					bopsf.alphaH = alpha1;
					bopsf.alphaY = alpha2;
					
					// set the regularization parameters
					bopsf.lambdaS = lambda1;
					bopsf.lambdaP = lambda2;
					bopsf.lambdaW = lambda3;
					
					// set the optimization parameters
					bopsf.eta = eta;
					bopsf.maxEpochs = maxEpochs;
					
					// set the bag of patterns parameters
					bopsf.slidingWindowSize = n;
					bopsf.innerDimension = w;
					bopsf.alphabetSize = a;
					bopsf.degree = degree;
					
					// set the latent dimensionality
					bopsf.D = k;
					
					// load the data into matrices form
			        Matrix predictors = new Matrix();
			        predictors.LoadDatasetFeatures(finalTrainSet, false);
			        predictors.LoadDatasetFeatures(finalTestSet, true);
			        
			        Matrix labels = new Matrix();
			        labels.LoadDatasetLabels(finalTrainSet, false);
			        labels.LoadDatasetLabels(finalTestSet, true); 
			        
			        // initialize the training and total instances fields
			        bopsf.numTrainInstances = finalTrainSet.instances.size();
			        bopsf.numTotalInstances = finalTrainSet.instances.size() + finalTestSet.instances.size();
			        bopsf.numPoints = finalTestSet.numFeatures; 

			        // set the number of labels
			        finalTrainSet.ReadNominalTargets();
			        bopsf.numLabels = finalTrainSet.nominalLabels.size();
			        
			        // assign the predictors and labels
			        bopsf.X = predictors;
			        bopsf.Y = labels;
			        
			        
			        Logging.println("Dataset name '"+dsName+"'", LogLevel.DEBUGGING_LOG);
			        Logging.println("NTrain=" + bopsf.numTrainInstances + ", NTest=" + bopsf.numTotalInstances, LogLevel.DEBUGGING_LOG);
			        Logging.println("OrigDim=" + bopsf.numPoints + ", LatentDim=" + bopsf.D, LogLevel.DEBUGGING_LOG);
			        Logging.println("NLabels=" + bopsf.numLabels, LogLevel.DEBUGGING_LOG);
 					Logging.println("Impacts: alphaH=" + bopsf.alphaH + ", alphaY=" + bopsf.alphaY, LogLevel.DEBUGGING_LOG);
					Logging.println("Regularization: lambdaS=" + bopsf.lambdaS + ", lambdaF=" + bopsf.lambdaP + ", lambdaW=" + bopsf.lambdaW, LogLevel.DEBUGGING_LOG);
					Logging.println("Optimization: eta=" + bopsf.eta + ", maxEpochs=" + bopsf.maxEpochs, LogLevel.DEBUGGING_LOG);
					Logging.println("BOP: slidingWindow=" + bopsf.slidingWindowSize + ", innerDimension=" + bopsf.innerDimension + ", alphabetSize=" + bopsf.alphabetSize, LogLevel.DEBUGGING_LOG);
					
					
			        // start the optimization
					double mcr = bopsf.Optimize();
			        

					// measure elapsed time
					long end = System.currentTimeMillis();  
					double elapsedTime = end - start; 
			        
					
			    	//Logging.println( 
			        	System.out.println( String.valueOf(mcr) + " "
			        		+ dsName + " " 
							+ fold + " " 
							+ model + " " 
							+ alpha1 + " " 
							+ alpha2 + " " 
							+ alpha3 + " " 
							+ lambda1 + " " 
							+ lambda2 + " " 
							+ lambda3 + " "
							+ lambda4 + " "
							+ eta + " " 
							+ maxEpochs + " " 
							+ elapsedTime ); //, 
							//Logging.LogLevel.PRODUCTION_LOG);
				}
				else if (model.compareTo("sbsd") == 0) 
				{
				
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
					
					trainSet = null;
					validationSet = null;
					testSet = null;
					System.gc();
					
					StructurePreservingDecomposition sbsd = new StructurePreservingDecomposition(k);
					
					sbsd.alphaR = alpha1;
					sbsd.alphaA = alpha3;
					sbsd.alphaT = alpha4;
					sbsd.alphaD = alpha2;
					
					sbsd.eta = eta;						
					sbsd.lambdaU = lambda1;
					sbsd.lambdaV = lambda2;
					sbsd.lambdaW = lambda3;	
					sbsd.maxEpochs = maxEpochs;
					
			        Matrix predictors = new Matrix();
			        predictors.LoadDatasetFeatures(finalTrainSet, false);
			        predictors.LoadDatasetFeatures(finalTestSet, true);
			        
			        Matrix labels = new Matrix();
			        labels.LoadDatasetLabels(finalTrainSet, false);
			        labels.LoadDatasetLabels(finalTestSet, true); 
			        
			        finalTrainSet.ReadNominalTargets();
			        
			        // initialize the training and total instances fields
			        sbsd.numTrainInstances = finalTrainSet.instances.size();
			        sbsd.numTotalInstances = finalTrainSet.instances.size() + finalTestSet.instances.size();
			        sbsd.numPoints = finalTestSet.numFeatures; 
			        sbsd.numLabels = finalTrainSet.nominalLabels.size();

					
			        
			        Logging.println("NTrain=" + sbsd.numTrainInstances + ", NTest=" + sbsd.numTotalInstances, LogLevel.DEBUGGING_LOG);
			        Logging.println("M_i=" + sbsd.numPoints + "-> D_i=" + k, LogLevel.DEBUGGING_LOG);
			        Logging.println("numLabels=" + sbsd.numLabels, LogLevel.DEBUGGING_LOG); 
				    
			        finalTrainSet = null;
			        finalTestSet = null;
			        
					long end = System.currentTimeMillis();  
					double elapsedTime = end - start; 
			        
					Logging.println("Impacts: aR=" + alpha1 + ", aT=" + alpha4 + ", aD=" + alpha2 + 
									", aA=" + alpha3, LogLevel.DEBUGGING_LOG);
					
			        double testError = sbsd.Decompose(predictors, labels); 
			        
			    	//Logging.println( 
			        	System.out.println( String.valueOf(testError) + " "
			        		+ dsName + " " 
							+ fold + " " 
							+ model + " " 
							+ alpha1 + " " 
							+ alpha2 + " " 
							+ alpha4 + " " 
							+ alpha3 + " " 
							+ lambda1 + " " 
							+ lambda2 + " " 
							+ lambda3 + " " 
							+ eta + " " 
							+ maxEpochs + " " 
							+ elapsedTime ); //, 
							//Logging.LogLevel.PRODUCTION_LOG);
				} 
				else if (model.compareTo("nsp") == 0) 
				{
				
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
					
					trainSet = null;
					validationSet = null;
					testSet = null;
					System.gc();
					
					NonlinearStructurePreservingDecomposition nsp = new NonlinearStructurePreservingDecomposition(k);
					
					nsp.alphaR = alpha1;
					nsp.alphaA = alpha3;
					nsp.alphaT = alpha4;
					nsp.alphaD = alpha2;
					
					nsp.eta = eta;						
					nsp.lambdaU = lambda1;
					nsp.lambdaV = lambda2;
					nsp.lambdaW = lambda3;	
					nsp.maxEpochs = maxEpochs;
					
			        Matrix predictors = new Matrix();
			        predictors.LoadDatasetFeatures(finalTrainSet, false);
			        predictors.LoadDatasetFeatures(finalTestSet, true);
			        
			        Matrix labels = new Matrix();
			        labels.LoadDatasetLabels(finalTrainSet, false);
			        labels.LoadDatasetLabels(finalTestSet, true); 
			        
			        finalTrainSet.ReadNominalTargets();
			        
			        // initialize the training and total instances fields
			        nsp.numTrainInstances = finalTrainSet.instances.size();
			        nsp.numTotalInstances = finalTrainSet.instances.size() + finalTestSet.instances.size();
			        nsp.numPoints = finalTestSet.numFeatures; 
			        nsp.numLabels = finalTrainSet.nominalLabels.size();
			        
			        Logging.println("NTrain=" + nsp.numTrainInstances + ", NTest=" + nsp.numTotalInstances, LogLevel.DEBUGGING_LOG);
			        Logging.println("M_i=" + nsp.numPoints + "-> D_i=" + k, LogLevel.DEBUGGING_LOG);
			        Logging.println("numLabels=" + nsp.numLabels, LogLevel.DEBUGGING_LOG); 
				    
			        finalTrainSet = null;
			        finalTestSet = null;
			        
					long end = System.currentTimeMillis();  
					double elapsedTime = end - start; 
			        
					Logging.println("Impacts: aR=" + alpha1 + ", aT=" + alpha4 + ", aD=" + alpha2 + 
									", aA=" + alpha3, LogLevel.DEBUGGING_LOG);
					
			        double testError = nsp.Decompose(predictors, labels); 
			        
			    	//Logging.println( 
			        	System.out.println( String.valueOf(testError) + " "
			        		+ dsName + " " 
							+ fold + " " 
							+ model + " " 
							+ alpha1 + " " 
							+ alpha2 + " " 
							+ alpha4 + " " 
							+ alpha3 + " " 
							+ lambda1 + " " 
							+ lambda2 + " " 
							+ lambda3 + " " 
							+ eta + " " 
							+ maxEpochs + " " 
							+ elapsedTime ); //, 
							//Logging.LogLevel.PRODUCTION_LOG);
				} 

		}
			

	}

}
