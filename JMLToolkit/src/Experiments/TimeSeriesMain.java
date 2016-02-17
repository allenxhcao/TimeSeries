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
import MatrixFactorization.SimilaritySupervisedMF;
import TimeSeries.*;
import TimeSeries.BagOfPatterns.RepresentationType;
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
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class TimeSeriesMain {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		long programStartTime = System.currentTimeMillis();
		
		if (args.length == 0) {
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG; 
			 //Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG; 

			String dir = "E:\\Data\\classification\\timeseries\\",
			//String dir = "vartheta:\\Data\\classification\\timeseries\\",
					ds = "gasSensor", 
					foldNo = "default";  
			String sp = File.separator;
			
			args = new String[] { 
					"fold=" + foldNo,
					"model=svm", 
					"runMode=test",
					"trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp
							+ ds + "_TRAIN",  
					"validationSet=" + dir + ds + sp + "folds" + sp + foldNo 
							+ sp + ds + "_VALIDATION", 
					"testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp 
							+ ds + "_TEST", 
					
					"paaRatio=1.0",    
					"percentile=35",
					
					"patternRepresentation=poly",
					"n=100",   
					"w=5",  
					"a=6", 
					"degree=1",
					
					"factorizationsCacheFolder=/home/josif/Documents/factorizations_cache", 
					"learnRate=0.001", 
					"learnRate1=0.001", 
					"learnRate2=0.001", 
					"learnRate3=0.0001", 
					"lambdaU=0.0001",  
					"lambdaV=0.0001",  
					"latentDimensionsRatio=0.2",  
					"beta=0.5", 
					"maxEpochs=300", 
					
					
					"svmC=1", 
					"svmRKGamma=1", 
					"alpha=0.5", 
					"warpingWindowFraction=0.5", 
					"svmKernel=polynomial", 
					"distorsion=false",  
					"timeWarping=false", 
					"unsupervisedFactorization=true", 
					"transformationFieldsFolder=/home/josif/Documents/transformation_fields",
					"useTransformationFields=true", 
					"showAllWarnings=false",
					"transformationsRate=0.1"  
					};
		}

		if (args.length > 0) {
			// set model values
			String model = "", runMode = "";
			String trainSetPath = "", validationSetPath = "", testSetPath = "", fold = "", factorizationsCacheFolder = "", transformationFieldsFolder = "";
			double lambda = 0.001, alpha = 0.8, beta = 0.1, learnRate = 0.01, svmC = 1.0, svmRKGamma = 1, maxMargin = 1.0;
			double lambdaU=0, lambdaV=0, learnRate1 = 0, learnRate2 = 0, learnRate3 = 0;
			double latentDimensionsRatio = 0;
			double transformationsRate = 1.0;
			double warpingWindowFraction = 0.1;
			int k = -1, maxEpocs = 20000;
			int degree = 3, percentile = 15;
			double totalMissingRatio = 0.0, gapRatio = 0.0;
			double epsilon = 0.00;
			String interpolationTechnique = "linear";
			String svmKernel = "rbf";
			String patternRepresentation = "poly";
			boolean avoidExtrapolation = false;
			double nr = 0.1, paaRatio = 1.0;

			boolean enableDistorsion = false;
			boolean enableTimeWarping = false;
			boolean enableSupervisedFactorization = false;
			boolean useTransformationFields = false;
			boolean showAllWarnings = false;
			boolean unsupervisedFactorization = false;

			int n = -1, w = -1, a = -1;
			
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
				else if (argTokens[0].compareTo("dimensions") == 0)
  				k = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("learnRate") == 0)
					learnRate = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("learnRate1") == 0)
					learnRate1 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("learnRate2") == 0)
					learnRate2 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("learnRate3") == 0)
					learnRate3 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambda") == 0)
					lambda = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaU") == 0)
					lambdaU = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaV") == 0)
					lambdaV = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("paaRatio") == 0)
					paaRatio = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("alpha") == 0)
					alpha = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("beta") == 0)
					beta = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("maxMargin") == 0)
					maxMargin = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("warpingWindowFraction") == 0)
					warpingWindowFraction = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("transformationsRate") == 0)
					transformationsRate = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("svmKernel") == 0)
					svmKernel = argTokens[1];
				else if (argTokens[0].compareTo("svmC") == 0)
					svmC = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("degree") == 0)
					degree = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("svmRKGamma") == 0)
					svmRKGamma = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("totalMissingRatio") == 0)
					totalMissingRatio = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("gapRatio") == 0)
					gapRatio = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("interpolation") == 0)
					interpolationTechnique = argTokens[1];
				else if (argTokens[0].compareTo("avoidExtrapolation") == 0)
					avoidExtrapolation = argTokens[1].toUpperCase().compareTo(
							"TRUE") == 0 ? true : false;
				else if (argTokens[0].compareTo("unsupervisedFactorization") == 0)
					unsupervisedFactorization = argTokens[1].toUpperCase()
							.compareTo("TRUE") == 0 ? true : false;
				else if (argTokens[0].compareTo("showAllWarnings") == 0)
					showAllWarnings = argTokens[1].toUpperCase().compareTo(
							"TRUE") == 0 ? true : false;
				else if (argTokens[0].compareTo("distorsion") == 0)
					enableDistorsion = argTokens[1].toUpperCase().compareTo(
							"TRUE") == 0 ? true : false;
				else if (argTokens[0].compareTo("timeWarping") == 0)
					enableTimeWarping = argTokens[1].toUpperCase().compareTo(
							"TRUE") == 0 ? true : false;
				else if (argTokens[0].compareTo("supervisedFactorization") == 0)
					enableSupervisedFactorization = argTokens[1].toUpperCase()
							.compareTo("TRUE") == 0 ? true : false;
				else if (argTokens[0].compareTo("useTransformationFields") == 0)
					useTransformationFields = argTokens[1].toUpperCase()
							.compareTo("TRUE") == 0 ? true : false;
				else if (argTokens[0].compareTo("epsilon") == 0)
					epsilon = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("maxEpochs") == 0)
					maxEpocs = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("transformationFieldsFolder") == 0)
					transformationFieldsFolder = argTokens[1];
				else if (argTokens[0].compareTo("n") == 0)
					n = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("nr") == 0)
					nr = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("w") == 0) 
					w = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("a") == 0)
					a = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("percentile") == 0)
					percentile = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("patternRepresentation") == 0)
					patternRepresentation = argTokens[1];
			}

			if (showAllWarnings)
				Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

			long startProgram = System.currentTimeMillis();
			
			// Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

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

			// if there is any missing ratio parameter then sparsify the dataset
			if (totalMissingRatio > 0) {
				MissingPointsGenerator mpg = new MissingPointsGenerator();
				// avoid extreme points
				mpg.avoidExtremes = avoidExtrapolation;
				mpg.GenerateMissingPoints(trainSet, totalMissingRatio, gapRatio);
				mpg.GenerateMissingPoints(validationSet, totalMissingRatio,
						gapRatio);
				mpg.GenerateMissingPoints(testSet, totalMissingRatio, gapRatio);

			}

			// of k was not initialized from parameters then initialize it from
			// the ratio
			if (k <= 0) {
				if (latentDimensionsRatio > 0) {
					k = (int) (trainSet.numFeatures * latentDimensionsRatio);
				} else // set latent dimensions to 2*numLabels
				{
					trainSet.ReadNominalTargets();
					k = ((int) epsilon) * trainSet.nominalLabels.size();
				}
			}
			
			// initialize lambdaU and lambdaV in case just a lambda is specified
			if(lambdaU == 0 && lambda > 0) lambdaU = lambda;
			if(lambdaV == 0 && lambda > 0) lambdaV = lambda;

			String dsName = new File(trainSetPath).getName().split("_TRAIN")[0];

			/*
			Logging.println("DataSet: " + dsName + ", model:" + model
					+ ", runMode:" + runMode + ", k :" + k + ", lambda: "
					+ lambda + ", svmC:" + svmC + ", svmPKExp: " + degree
					+ ", learnRate:" + learnRate,
					Logging.LogLevel.DEBUGGING_LOG);
			*/
			if (model.compareTo("mfsvm") == 0) {
				long start = System.currentTimeMillis();

				MFSVM mfSvm = new MFSVM();
				mfSvm.latentDimensions = k;
				mfSvm.learningRate = learnRate;
				mfSvm.lambda = lambda;
				mfSvm.kernelType = svmKernel;
				mfSvm.svmC = svmC;
				mfSvm.alpha = alpha;
				mfSvm.maxEpocs = maxEpocs;
				mfSvm.svmPolynomialKernelExp = degree;
				mfSvm.svmRBFKernelGamma = svmRKGamma;
				mfSvm.enableTimeWarping = enableTimeWarping;
				mfSvm.maxMargin = maxMargin;
				mfSvm.warpingWindowFraction = warpingWindowFraction;
				mfSvm.unsupervisedFactorization = unsupervisedFactorization;

				// set the cache folder
				FactorizationsCache.getInstance().cacheDir = factorizationsCacheFolder;
				// set the description for the factorization
				mfSvm.factorizationDescription = dsName + "_" + fold + "_"
						+ runMode
						+ (enableSupervisedFactorization ? "_supervised" : "")
						+ (enableTimeWarping ? "_warped" : "")
						+ (enableDistorsion ? "_distorsion" : "");

				double errorRate = 0;

				// check if it is a validation test run
				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						Distorsion.getInstance().Distort(trainSet, epsilon);
					}

					errorRate = mfSvm.Classify(trainSet, validationSet);
				} else if (runMode.compareTo("test") == 0) {
					DataSet mergedTrain = new DataSet(trainSet);
					mergedTrain.AppendDataSet(validationSet);

					if (enableDistorsion) {
						mergedTrain = Distorsion.getInstance().Distort(
								mergedTrain, epsilon);
					}
					errorRate = mfSvm.Classify(mergedTrain, testSet);
				}

				long end = System.currentTimeMillis();
				double elapsedTime = end - start;

				Logging.println(
						errorRate + " " 
						+ dsName + " " 
						+ fold + " "
						+ model + " " 
						+ runMode + " "
						+ (enableTimeWarping ? "warped " : "") 
						+ ", k=" + k
						+ ", lambda=" + lambda 
						+ ", rate=" + learnRate  
						+ ", alpha=" + alpha 
						+ ", margin=" + maxMargin 
						+ ", svmC="+svmC 
						+ ", svmPKExp=" + degree
						+ ", maxEpocs=" + maxEpocs + " "
						+ elapsedTime + " "
						+ mfSvm.avgClassificationTime,
						Logging.LogLevel.PRODUCTION_LOG);

			} else if (model.compareTo("dtwnn") == 0) {
				
				long start = System.currentTimeMillis();

				// interpolate/impute the missing values
				if (totalMissingRatio > 0) {
					if (interpolationTechnique.compareTo("linear") == 0) {
						LinearInterpolation li = new LinearInterpolation(
								gapRatio);
						li.Interpolate(trainSet);
						li.Interpolate(validationSet);

						if (runMode.compareTo("test") == 0) {
							li.Interpolate(testSet);
						}
					}
					if (interpolationTechnique.compareTo("cubicspline") == 0) {
						SplineInterpolation si = new SplineInterpolation(
								gapRatio);
						si.CubicSplineInterpolation(trainSet);
						si.CubicSplineInterpolation(validationSet);

						if (runMode.compareTo("test") == 0) {
							si.CubicSplineInterpolation(testSet);
						}
					} else if (interpolationTechnique.compareTo("bspline") == 0) {
						SplineInterpolation si = new SplineInterpolation(
								gapRatio);
						si.BSplineInterpolation(trainSet);
						si.BSplineInterpolation(validationSet);

						if (runMode.compareTo("test") == 0) {
							si.BSplineInterpolation(testSet);
						}
					} else if (interpolationTechnique.compareTo("em") == 0) {
						DataSet mergedAll = new DataSet(trainSet);
						mergedAll.AppendDataSet(validationSet);
						mergedAll.AppendDataSet(testSet);

						ExpectationMaximization em = new ExpectationMaximization();
						em.ImputeMissing(mergedAll);

						int trainSize = trainSet.instances.size(), validationSize = validationSet.instances
								.size(), testSize = testSet.instances.size();
						// split back the set splits
						trainSet = mergedAll.GetSubset(0, trainSize);
						validationSet = mergedAll.GetSubset(trainSize,
								trainSize + validationSize);
						testSet = mergedAll.GetSubset(trainSize
								+ validationSize, trainSize + validationSize
								+ testSize);

					} else if (interpolationTechnique.compareTo("mbi") == 0) {
						DataSet mergedAll = new DataSet(trainSet);
						mergedAll.AppendDataSet(validationSet);
						mergedAll.AppendDataSet(testSet);

						// trainSet.SaveToArffFile("F:\\a.arff");

						ModelBasedImputation mbi = new ModelBasedImputation();
						mbi.Impute(mergedAll);

						int trainSize = trainSet.instances.size(), validationSize = validationSet.instances
								.size(), testSize = testSet.instances.size();

						// split back the set splits
						trainSet = mergedAll.GetSubset(0, trainSize);
						validationSet = mergedAll.GetSubset(trainSize,
								trainSize + validationSize);
						testSet = mergedAll.GetSubset(trainSize
								+ validationSize, trainSize + validationSize
								+ testSize);

					} else if (interpolationTechnique
							.compareTo("collaborative") == 0) {
						// create an overall merged dataset
						DataSet mergedAll = new DataSet(trainSet);
						mergedAll.AppendDataSet(validationSet);

						if (runMode.compareTo("test") == 0) {
							mergedAll.AppendDataSet(testSet);
						}

						CollaborativeImputation ci = new CollaborativeImputation();
						ci.k = k;
						ci.lambda = lambda;
						ci.learnRate = learnRate;
						// collaboratively impute the missing points
						ci.Impute(mergedAll);

						int trainSize = trainSet.instances.size(), validationSize = validationSet.instances
								.size(), testSize = testSet.instances.size();
						// split back the set splits
						trainSet = mergedAll.GetSubset(0, trainSize);
						validationSet = mergedAll.GetSubset(trainSize,
								trainSize + validationSize);

						if (runMode.compareTo("test") == 0) {
							testSet = mergedAll.GetSubset(trainSize
									+ validationSize, trainSize
									+ validationSize + testSize);
						}
					}
				}

				NearestNeighbour nn = new NearestNeighbour("dtw");

				double errorRate = 0;
				
				long startClassification = System.currentTimeMillis();

				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}

					errorRate = nn.Classify(trainSet, validationSet);
				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					DataSet mergedTrain = new DataSet(trainSet);
					mergedTrain.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						mergedTrain = Distorsion.getInstance().Distort(
								mergedTrain, epsilon);
					}

					errorRate = nn.Classify(mergedTrain, testSet);
				}
				
				long end = System.currentTimeMillis();
				double elapsedDTWNNTime = end - start;

				System.out.println(
				//Logging.println(String.valueOf(errorRate) + " "
						String.valueOf(errorRate) + " "
						+ dsName + " "
						+ fold + " "
						+ model + " "
						+ elapsedDTWNNTime);
						//Logging.LogLevel.PRODUCTION_LOG);

			} else if (model.compareTo("svm") == 0) {
				try {
					
					
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
					

					// PAA paa = new PAA();
					// mergedTrain = paa.generatePAA(mergedTrain,
					// latentDimensionsRatio);
					// testSet = paa.generatePAA(testSet,
					// latentDimensionsRatio);

					Instances trainWeka = finalTrainSet.ToWekaInstances();
					Instances testWeka = finalTestSet.ToWekaInstances();

					SMO svm = null;
					if (svmKernel.compareTo("polynomial") == 0)
						svm = WekaClassifierInterface.getPolySvmClassifier(
								svmC, degree);
					else
						svm = WekaClassifierInterface.getRbfSvmClassifier(svmC,
								svmRKGamma);

					svm.buildClassifier(trainWeka); 

					Evaluation eval = new Evaluation(trainWeka);
					eval.evaluateModel(svm, testWeka);
					
					long end = System.currentTimeMillis();
					double elapsedTime = end - startProgram;

					Logging.println(String.valueOf(eval.errorRate()) + " "
							+ dsName + " " + fold + " " + model + " " + svmC
							+ " " + degree + " " + svmRKGamma + " " + elapsedTime ,
							Logging.LogLevel.PRODUCTION_LOG);

				} catch (Exception exc) {
					exc.printStackTrace();
				}
			}
			else if (model.compareTo("pcasvm") == 0) {
				try {
					
					
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
					
					PCASVM pcasvm = new PCASVM();
					pcasvm.kernelType = svmKernel;
					pcasvm.svmC = svmC;
					pcasvm.svmPolynomialKernelExp = degree;
					pcasvm.variance  = 0.75;
					
					double errorRate = pcasvm.Classify(finalTrainSet, finalTestSet);
					
					long end = System.currentTimeMillis();
					double elapsedTime = end - startProgram;

					Logging.println(String.valueOf(errorRate) + " " 
							+ dsName + " " + fold + " " + model + " " + svmC + " " + svmKernel
							+ " " + degree + " " + svmRKGamma + " " + elapsedTime ,
							Logging.LogLevel.PRODUCTION_LOG);

				} catch (Exception exc) {
					exc.printStackTrace();
				}
			}
			else if (model.compareTo("cidnn") == 0) {
				// System.out.println("Enter");

				long start = System.currentTimeMillis();

				double errorRateTrain = 1.0, errorRateTest = 1.0;
				
				long startClassification = System.currentTimeMillis();
				
				Evaluation eval = null;
				Random rand = new Random();
				
				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}
					
					try
					{
						NearestNeighbour nnClassifier = new NearestNeighbour("cid"); 
						errorRateTrain = nnClassifier.ClassifyLeaveOneOut(trainSet); 
						errorRateTest = nnClassifier.Classify(trainSet, validationSet);  
					}
					catch(Exception exc)
					{
						Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
					}
					
				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					DataSet mergedTrain = new DataSet(trainSet);
					mergedTrain.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						mergedTrain = Distorsion.getInstance().Distort(
								mergedTrain, epsilon); 
					}

					try
					{
						NearestNeighbour nnClassifier = new NearestNeighbour("cid"); 
						errorRateTrain = nnClassifier.ClassifyLeaveOneOut(trainSet); 
						errorRateTest = nnClassifier.Classify(mergedTrain, testSet);  
					}
					catch(Exception exc)
					{
						exc.printStackTrace();
						Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
					}
				}

				long end = System.currentTimeMillis();
				double elapsedTime = end - start;

				System.out.println(String.valueOf(errorRateTrain) + " " + String.valueOf(errorRateTest)
						+ " "
						+ dsName
						+ " "
						+ fold
						+ " "
						+ model
						+ " "
						+ elapsedTime);
						//Logging.LogLevel.PRODUCTION_LOG);

			}
			else if (model.compareTo("enn") == 0) {
				// System.out.println("Enter");

				long start = System.currentTimeMillis();

				double errorRateTrain = 1.0, errorRateTest = 1.0;
				
				long startClassification = System.currentTimeMillis();
				
				Evaluation eval = null;
				Random rand = new Random();
				
				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}
					
					try
					{
						NearestNeighbour nnClassifier = new NearestNeighbour("euclidean"); 
						errorRateTrain = nnClassifier.ClassifyLeaveOneOut(trainSet); 
						errorRateTest = nnClassifier.Classify(trainSet, validationSet);  
					}
					catch(Exception exc)
					{
						Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
					}
					
				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					DataSet mergedTrain = new DataSet(trainSet);
					mergedTrain.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						mergedTrain = Distorsion.getInstance().Distort(
								mergedTrain, epsilon); 
					}

					try
					{
						NearestNeighbour nnClassifier = new NearestNeighbour("euclidean"); 
						//errorRateTrain = nnClassifier.ClassifyLeaveOneOut(trainSet); 
						errorRateTest = nnClassifier.Classify(mergedTrain, testSet);  
					}
					catch(Exception exc)
					{
						exc.printStackTrace();
						Logging.println(exc.getMessage(), LogLevel.ERROR_LOG); 
					}
				}

				long end = System.currentTimeMillis();
				double elapsedTime = end - start;

				System.out.println(String.valueOf(errorRateTrain) + " " + String.valueOf(errorRateTest)
						+ " "
						+ dsName
						+ " "
						+ fold
						+ " "
						+ model
						+ " "
						+ elapsedTime);
						//Logging.LogLevel.PRODUCTION_LOG);

			}else if (model.compareTo("bop") == 0) {
				// System.out.println("Enter");

				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrain = null;
				
				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}
					
					finalTrain = trainSet;

				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					finalTrain = new DataSet(trainSet);
					finalTrain.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						finalTrain = Distorsion.getInstance().Distort(
								finalTrain, epsilon);
					}
				}

				Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrain, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrain, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            if( n <= 0)
				{
					n = (int)(nr * X.getDimColumns()); 
				}
				
	            System.out.println(n);
	            
	            int slidingWindowSize = n;
	            int innerDimension = w;
	            int alphabetSize = a;
	            
	            
	            BagOfPatterns bop = new BagOfPatterns();
	            
	            bop.slidingWindowSize = slidingWindowSize;
	            
	            if( patternRepresentation.compareTo("poly") == 0 )
            		bop.representationType = RepresentationType.Polynomial;
	            else if( patternRepresentation.compareTo("sax") == 0 )
	            	bop.representationType = RepresentationType.SAX; 
	            
	            bop.innerDimension = innerDimension;
	            bop.alphabetSize = alphabetSize;
	            bop.polyDegree = degree;
	            
	            Matrix H = bop.CreateWordFrequenciesMatrix(X);
	            
	            //H.SaveToFile("C:\\Users\\josif\\Desktop\\"+dsName+"_"+bop.representationType+".txt");
	            
	    	    DataSet trainSetHist = new DataSet();
	    	    trainSetHist.LoadMatrixes(H, Y, 0, finalTrain.instances.size());
	    	    DataSet testSetHist = new DataSet();
	    	    testSetHist.LoadMatrixes(H, Y, finalTrain.instances.size(), H.getDimRows());
	    	    
	    	    
	    	    
	    	    NearestNeighbour nn = new NearestNeighbour("euclidean");
				errorRate = nn.Classify(trainSetHist, testSetHist);
				
				// run also an svm over and print the mcr
				/*
	    	    Instances trainWeka = trainSetHist.ToWekaInstances();
				Instances testWeka = testSetHist.ToWekaInstances();

				SMO svm = null;
				if (svmKernel.compareTo("polynomial") == 0)
					svm = WekaClassifierInterface.getPolySvmClassifier(
							1.0, 1.0);
				else
					svm = WekaClassifierInterface.getRbfSvmClassifier(svmC,
							svmRKGamma);

				try{
					svm.buildClassifier(trainWeka); 
					
					
					Evaluation eval = new Evaluation(trainWeka);
					eval.evaluateModel(svm, testWeka);
					errorRate = eval.errorRate();
					*/
	    	    
				long endTime = System.currentTimeMillis();
				double elapsedTime = endTime - startTime;

				System.out.println( String.valueOf(errorRate)
				//Logging.println(String.valueOf(errorRate)
						+ " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model	+ " "
						+ patternRepresentation + " "
						+ n	+ " "
						+ w	+ " "
						+ a	+ " "
						+ degree + " "
						+ elapsedTime);
						//Logging.LogLevel.PRODUCTION_LOG); 
				
				
				
					
					//Logging.println("SVM: " + String.valueOf(eval.errorRate()) + " "
					//		+ dsName + " " + fold + " " + model + " " + svmC + " " + degree + " " + svmRKGamma,
					//		Logging.LogLevel.PRODUCTION_LOG);

				//}
				//catch(Exception exc){}
				
			

				

			}
			else if (model.compareTo("esd") == 0) {
				// System.out.println("Enter");
				
				DataSet finalTrain = null;
				
				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}
					
					finalTrain = trainSet;

				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					finalTrain = new DataSet(trainSet);
					finalTrain.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						finalTrain = Distorsion.getInstance().Distort(
								finalTrain, epsilon);
					}
				} 

				
				int numTrials = 10;
	            double [] accuracies = new double[numTrials];
	            double [] runTimes = new double[numTrials];
	            double [] numAccepted = new double[numTrials];
	            
	            
	            
	            for(int trial = 0; trial < numTrials; trial++)
	            {
					long startMethodTime = System.currentTimeMillis(); 
	            	
					ScalableShapeletDiscovery ssd = new ScalableShapeletDiscovery();
		            ssd.trainSetPath = trainSetPath;
		            ssd.testSetPath = testSetPath;
		            ssd.percentile = percentile;
		            ssd.paaRatio = paaRatio;
		            ssd.Search(); 		            
		            
					double elapsedMethodTime = System.currentTimeMillis() - startMethodTime;  
					
					double errorRate = ssd.ComputeTestError(); 
					
					
					accuracies[trial] = 1-errorRate;
					runTimes[trial] = elapsedMethodTime/1000; // in second
					numAccepted[trial] = ssd.numAcceptedShapelets;
					
					System.out.println( 
							dsName + " Trial="+trial+ ", " +
							"Accuracy="+accuracies[trial] + ", Time=" + runTimes[trial] + " " +
							", nAccepted= " + numAccepted[trial] + " " +
									fold	+ " "
									+ runMode + " "
									+ model	+ " "
									+ ssd.paaRatio	+ " "
									+ ssd.epsilon);
									
	            } 

	            System.out.println(
	            		dsName + " " + paaRatio + ", " + percentile + ", " +
	            				StatisticalUtilities.Mean(accuracies) + ", " + 
	            				StatisticalUtilities.StandardDeviation(accuracies) + ", " +
	            				StatisticalUtilities.Mean(runTimes) + ", " + 
	            				StatisticalUtilities.StandardDeviation(runTimes) + ", " +
	            				StatisticalUtilities.Mean(numAccepted) + ", " + 
            					StatisticalUtilities.StandardDeviation(numAccepted)  ); 
				
						//Logging.LogLevel.PRODUCTION_LOG); 
				
				
				
					
					//Logging.println("SVM: " + String.valueOf(eval.errorRate()) + " "
					//		+ dsName + " " + fold + " " + model + " " + svmC + " " + degree + " " + svmRKGamma,
					//		Logging.LogLevel.PRODUCTION_LOG);

				//}
				//catch(Exception exc){}
				
			

				

			}
			else if (model.compareTo("lc") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}
					
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						finalTrainSet = Distorsion.getInstance().Distort(
								finalTrainSet, epsilon);
					}
				}

					            
	            Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            LocalConvolutions lc = new LocalConvolutions();
	            // initialize the sizes of data structures
	            lc.NTrain = finalTrainSet.GetNumInstances();
	            lc.NTest = testSet.GetNumInstances();
	            lc.M = X.getDimColumns();
	            // set the time series and labels
	            lc.T = X;
	            lc.Y = Y;
	            // set the learn rate and the number of iterations
	            lc.eta = 0.0001;
	            lc.maxIter = 200;
	            // set te number of patterns
	            lc.K = 10;
	            // set the size of segments
	            lc.L = 100;
	            // set the regularization parameter
	            lc.lambdaP = 0.0001;
	            lc.lambdaD = 0.0001;
	            
	            // learn the local convolutions
	            lc.Learn();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model	+ " "
						+ patternRepresentation + " "
						+ n	+ " "
						+ w	+ " "
						+ a	+ " "
						+ degree);
						
			} else if (model.compareTo("cleandtwnn") == 0) {
				// System.out.println("Enter");

				long start = System.currentTimeMillis();

				NearestNeighbour nn = new NearestNeighbour("cleandtw");

				double errorRate = 0;

				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}

					errorRate = nn.Classify(trainSet, validationSet);
				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					DataSet mergedTrain = new DataSet(trainSet);
					mergedTrain.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						mergedTrain = Distorsion.getInstance().Distort(
								mergedTrain, epsilon);
					}

					errorRate = nn.Classify(mergedTrain, testSet);
				}

				long end = System.currentTimeMillis();
				double elapsedTime = end - start;

				Logging.println(String.valueOf(errorRate)
						+ " "
						+ dsName
						+ " "
						+ fold
						+ " "
						+ model
						+ " "
						+ (totalMissingRatio > 0 ? totalMissingRatio + " "
								+ gapRatio + interpolationTechnique
								+ (avoidExtrapolation == true ? " noExt" : "")
								: " ") + " " + elapsedTime,
						Logging.LogLevel.PRODUCTION_LOG);

			} else if (model.compareTo("kf") == 0) {
				// System.out.println("Enter");

				long start = System.currentTimeMillis();

				KernelFactorization kf = new KernelFactorization();

				double errorRate = 0;

				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}

					errorRate = kf.Classify(trainSet, validationSet);
				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					DataSet mergedTrain = new DataSet(trainSet);
					mergedTrain.AppendDataSet(validationSet);

					// distort the training set if enabled
					if (enableDistorsion) {
						mergedTrain = Distorsion.getInstance().Distort(
								mergedTrain, epsilon);
					}

					errorRate = kf.Classify(mergedTrain, testSet);
				}

				long end = System.currentTimeMillis();
				double elapsedTime = end - start;

				Logging.println(String.valueOf(errorRate)
						+ " "
						+ dsName
						+ " "
						+ fold
						+ " "
						+ model
						+ " "
						+ (totalMissingRatio > 0 ? totalMissingRatio + " "
								+ gapRatio + interpolationTechnique
								+ (avoidExtrapolation == true ? " noExt" : "")
								: " ") + " " + elapsedTime,
						Logging.LogLevel.PRODUCTION_LOG);

			} else if (model.compareTo("isvm") == 0) {
				try {
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

					// PAA paa = new PAA();
					// finalTrainSet = paa.generatePAA(finalTrainSet, 0.2);
					// finalTestSet = paa.generatePAA(finalTestSet, 0.2);

					InvariantSVM isvm = new InvariantSVM();
					isvm.svmC = svmC;
					isvm.kernel = svmKernel;
					isvm.svmRKGamma = svmRKGamma;
					isvm.svmPKExp = degree;
					isvm.eps = epsilon;

					if (useTransformationFields) {
						TransformationFieldsGenerator.getInstance().transformationScale = epsilon;
						TransformationFieldsGenerator.getInstance().transformationRate = transformationsRate;
						TransformationFieldsGenerator.getInstance()
								.LoadTransformationFields(trainSet,
										transformationFieldsFolder);
						isvm.eps = epsilon;
					}

					double[] errorRates = isvm.Classify(finalTrainSet,
							finalTestSet);

					Logging.println(errorRates[0] + " " + errorRates[1] + " "
							+ dsName + " " + fold + " " + model + " " + epsilon
							+ " " + svmC + " " + degree + " "
							+ isvm.elapsedSVMTime + " " + isvm.elapsedISVMTime,

					Logging.LogLevel.PRODUCTION_LOG);

				} catch (Exception exc) {
					exc.printStackTrace();
				}
			} else if (model.compareTo("transformationField") == 0) {
				DataSet merged = new DataSet(trainSet);
				merged.AppendDataSet(validationSet);
				merged.AppendDataSet(testSet);

				TransformationFieldsGenerator.getInstance().warpingWindow = 1.0;

				TransformationFieldsGenerator.getInstance().CreateTransformationFields(merged,
							transformationFieldsFolder);

				// Trans
			}
			else if (model.compareTo("ssd") == 0){

				DataSet ultimateTrain = null, 
						ultimateTest = null; 
				
				// check if test or validation mode 
				// and arrange the ultimate data splits accordingly
				if( runMode.compareTo("test") == 0)
				{
					ultimateTrain = new DataSet(trainSet);
					ultimateTrain.AppendDataSet(validationSet);
					ultimateTest = testSet;
				}
				else if( runMode.compareTo("validation") == 0)
				{
					ultimateTrain = trainSet;
					ultimateTest = validationSet;
				}
				
				// initialize a mmsd 	
				
				
				
				SimilaritySupervisedMF sd = new SimilaritySupervisedMF(
						ultimateTrain, ultimateTest, k);
				
				sd.eta1 = learnRate1;
				sd.eta2 = learnRate2;
				sd.eta3 = learnRate3;
				
				
				sd.lambdaU = lambdaU;
				sd.lambdaV = lambdaV;
				sd.C = svmC;
				sd.gamma = svmRKGamma;
				sd.maxEpocs = maxEpocs;
				
				double mcrTest = sd.Optimize();
				
				Logging.println(
						mcrTest + " " 
						+ ", dataSet=" + dsName 
						+ ", fold=" + fold
						+ ", model=" + model 
						+ ", mode=" + runMode 
						+ ", k=" + k
						+ ", lambdaU=" + lambdaU
						+ ", lambdaV=" + lambdaV
						+ ", learnRate1=" + learnRate1
						+ ", learnRate2=" + learnRate2
						+ ", beta=" + beta 
						+ ", svmC="+svmC
						+ ", svmKernel="+svmKernel 
						+ ", svmPKExp=" + degree
						+ ", maxEpocs=" + maxEpocs,
						Logging.LogLevel.PRODUCTION_LOG);
				
			}
			else if (model.compareTo("smo") == 0){
				
				DataSet ultimateTrain = null, 
						ultimateTest = null; 
				
				// check if test or validation mode 
				// and arrange the ultimate data splits accordingly
				if( runMode.compareTo("test") == 0)
				{
					ultimateTrain = new DataSet(trainSet);
					ultimateTrain.AppendDataSet(validationSet);
					ultimateTest = testSet;
				}
				else if( runMode.compareTo("validation") == 0)
				{
					ultimateTrain = trainSet;
					ultimateTest = validationSet;
				}
				
		        
				Matrix X = new Matrix();
		        X.LoadDatasetFeatures(ultimateTrain, false); 
		        X.LoadDatasetFeatures(ultimateTest, true); 
		        
		        List<Double> Y = new ArrayList<Double>();
		        for(int i = 0; i < ultimateTrain.instances.size(); i++)
		        	Y.add( ultimateTrain.instances.get(i).target );
		        for(int i = 0; i < ultimateTest.instances.size(); i++)
		        	Y.add( ultimateTest.instances.get(i).target );
		        
		        int n_train = ultimateTrain.instances.size();
		        int n_test = ultimateTest.instances.size();
		        
		        // set the values of the labels as 1 and -1
		        for(int i = 0; i < n_train+n_test; i++)
		        {
		        	if( Y.get(i) != 1.0 )
		        		Y.set(i, -1.0);
		        }
		        
		        Kernel kernel = new Kernel();
		        if(svmKernel.compareTo("polynomial") == 0)
		        {
		        	kernel.type = Kernel.KernelType.Polynomial;
		        	kernel.degree = (int)degree;
		        }
		        else if(svmKernel.compareTo("gaussian") == 0)
		        {
		        	kernel.type = Kernel.KernelType.Gaussian;
		        	kernel.sig2 = svmRKGamma;  
		        }
		        
				NaiveSmo smo = new NaiveSmo(X, Y, n_train, svmC, kernel); 
				smo.Optimize();  
			}
			else if (model.compareTo("pcasvm") == 0){

				DataSet ultimateTrain = null, 
						ultimateTest = null; 
				
				// check if test or validation mode 
				// and arrange the ultimate data splits accordingly
				if( runMode.compareTo("test") == 0)
				{
					ultimateTrain = new DataSet(trainSet);
					ultimateTrain.AppendDataSet(validationSet);
					ultimateTest = testSet;
				}
				else if( runMode.compareTo("validation") == 0)
				{
					ultimateTrain = trainSet;
					ultimateTest = validationSet;
				}
				
				// initialize a mmsd 				
				PCASVM pcasvm = new PCASVM();
				pcasvm.variance = beta;
				pcasvm.svmC = svmC;
				pcasvm.kernelType = svmKernel;
				pcasvm.svmPolynomialKernelExp = degree;
				pcasvm.svmRBFKernelGamma = svmRKGamma;
				
				double mcrTest = pcasvm.Classify(trainSet, testSet);
				
				Logging.println(
						mcrTest + " " 
						+ ", dataSet=" + dsName 
						+ ", fold=" + fold
						+ ", model=" + model 
						+ ", mode=" + runMode 
						+ ", variance=" + beta
						+ ", svmC="+svmC
						+ ", svmKernel="+svmKernel 
						+ ", svmPKExp=" + degree,
						Logging.LogLevel.PRODUCTION_LOG);
				
			}
			else if (model.compareTo("repetitivity") == 0)
			{
					
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrain = null;
				
				if (runMode.compareTo("validation") == 0) {
					// distort the training set if enabled
					if (enableDistorsion) {
						trainSet = Distorsion.getInstance().DistortMLS(testSet,
								epsilon);
					}
					
					finalTrain = trainSet;
	
				} else if (runMode.compareTo("test") == 0) {
					// merge the train and test
					finalTrain = new DataSet(trainSet);
					finalTrain.AppendDataSet(validationSet);
	
					// distort the training set if enabled
					if (enableDistorsion) {
						finalTrain = Distorsion.getInstance().Distort(
								finalTrain, epsilon);
					}
				}
	
				
	             Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrain, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrain, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            Repetitivity repetitivity = new Repetitivity();
	            double repetitivityScore = repetitivity.MeasureRepetitivity(X);   
	            
	            
				long endTime = System.currentTimeMillis();
				double elapsedTime = endTime - startTime;
	
				System.out.println( String.valueOf(repetitivityScore) 
						+ " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model	+ " "
						+ elapsedTime);
			}

		}

	}

}
