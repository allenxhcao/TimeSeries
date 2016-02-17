package Experiments;

import Classification.*;
import Utilities.*;
import Clustering.KMedoids;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import MatrixFactorization.CollaborativeImputation;
import MatrixFactorization.FactorizationsCache;
import MatrixFactorization.SqueezeKernelFactorization;
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

public class Main {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		if (args.length == 0) {
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;
			// Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

			//String dir = "/home/josif/Dokumente/ucr_ts_data/",
			 String dir = "F:\\data\\ucr_ts_data\\",
			ds = "Coffee",
			foldNo = "3"; 

			String sp = File.separator;

			args = new String[] {
					"fold=" + foldNo,
					"model=sqf",
					// "model=cleandtwnn",
					//"model=dtwnn",
					// "model=enn",
					//"model=svm",
					// "model=transformationField",
					// "model=randTest",
					"runMode=train",
					// /home/josif/Documents/ucr_ts_data/FacesUCR
					// "trainSet=" + dir + ds + sp + ds + "_TRAIN" ,
					"trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp
							+ ds + "_TRAIN",
					"validationSet=" + dir + ds + sp + "folds" + sp + foldNo
							+ sp + ds + "_VALIDATION",
					// "testSet=" + dir + ds + sp + ds + "_TEST",
					"testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp
							+ ds + "_TEST",
					"factorizationsCacheFolder=/home/josif/Documents/factorizations_cache",
					"learnRate=0.01",
					"learnRateR=0.0001",
					"learnRateCA=0.00001", 
					"lambdaU=0.001",
					"lambdaV=0.001", 
					"latentDimensionsRatio=0.05", 
					"beta=0.5", 
					"maxEpocs=1000",
					"svmC=1",
					"svmPKExp=3",
					//"dimensions=10",
					"warpingWindowFraction=0.5",
					"svmKernel=polynomial", 
					"svmRKGamma=0.1",
					"distorsion=false",
					"epsilon=0.3",
					"timeWarping=false",
					"unsupervisedFactorization=false",
					"transformationFieldsFolder=/home/josif/Documents/transformation_fields",
					"useTransformationFields=true", 
					"showAllWarnings=true",
					"transformationsRate=0.1"  
					};
		}

		if (args.length > 0) {
			// set model values
			String model = "", runMode = "";
			String trainSetPath = "", validationSetPath = "", testSetPath = "", fold = "", factorizationsCacheFolder = "", transformationFieldsFolder = "";
			double lambda = 0.001, alpha = 0.8, beta = 0.1, learnRate = 0.01, svmC = 1.0, svmPKExp = 3.0, svmRKGamma = 1, maxMargin = 1.0;
			double lambdaU=0, lambdaV=0, learnRateR = 0, learnRateCA = 0;
			double latentDimensionsRatio = 0;
			double transformationsRate = 1.0;
			double warpingWindowFraction = 0.1;
			int k = -1, maxEpocs = 20000;
			double totalMissingRatio = 0.0, gapRatio = 0.0;
			double epsilon = 0.00;
			String interpolationTechnique = "linear";
			String svmKernel = "rbf";
			boolean avoidExtrapolation = false;

			boolean enableDistorsion = false;
			boolean enableTimeWarping = false;
			boolean enableSupervisedFactorization = false;
			boolean useTransformationFields = false;
			boolean showAllWarnings = false;
			boolean unsupervisedFactorization = false;

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
				else if (argTokens[0].compareTo("learnRateR") == 0)
					learnRateR = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("learnRateCA") == 0)
					learnRateCA = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambda") == 0)
					lambda = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaU") == 0)
					lambdaU = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaV") == 0)
					lambdaV = Double.parseDouble(argTokens[1]);
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
				else if (argTokens[0].compareTo("svmPKExp") == 0)
					svmPKExp = Double.parseDouble(argTokens[1]);
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
				else if (argTokens[0].compareTo("maxEpocs") == 0)
					maxEpocs = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("transformationFieldsFolder") == 0)
					transformationFieldsFolder = argTokens[1];
			}

			if (showAllWarnings)
				Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

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

			Logging.println("DataSet: " + dsName + ", model:" + model
					+ ", runMode:" + runMode + ", k :" + k + ", lambda: "
					+ lambda + ", svmC:" + svmC + ", svmPKExp: " + svmPKExp
					+ ", learnRate:" + learnRate,
					Logging.LogLevel.DEBUGGING_LOG);

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
				mfSvm.svmPolynomialKernelExp = svmPKExp;
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
						+ ", svmPKExp=" + svmPKExp
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
				
				long endClassification = System.currentTimeMillis();
				double avgClassificationTime = (double)(endClassification-startClassification)/(double)testSet.instances.size();
				
				long end = System.currentTimeMillis();
				double elapsedDTWNNTime = end - start;

				Logging.println(String.valueOf(errorRate) + " "
						+ dsName + " "
						+ fold + " "
						+ model + " "
						+ (totalMissingRatio > 0 ? totalMissingRatio + " "
								+ gapRatio + interpolationTechnique
								+ (avoidExtrapolation == true ? " noExt" : "")
								: " ") + " " + 
						elapsedDTWNNTime + " " +
						avgClassificationTime,
						Logging.LogLevel.PRODUCTION_LOG);

			} else if (model.compareTo("svm") == 0) {
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
					// mergedTrain =
					// Distorsion.getInstance().DistortMLS(mergedTrain, eps);

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
								svmC, svmPKExp);
					else
						svm = WekaClassifierInterface.getRbfSvmClassifier(svmC,
								svmRKGamma);

					svm.buildClassifier(trainWeka); 

					long startClassification = System.currentTimeMillis();
					
					Evaluation eval = new Evaluation(trainWeka);
					eval.evaluateModel(svm, testWeka);
					
					long endClassification = System.currentTimeMillis();
					double avgClassificationTime = (double)(endClassification-startClassification)/(double)testSet.instances.size();
					
					long end = System.currentTimeMillis();
					double elapsedTime = end - start;

					Logging.println(String.valueOf(eval.errorRate()) + " "
							+ dsName + " " + fold + " " + model + " " + svmC
							+ " " + svmPKExp + " " + svmRKGamma + " " + elapsedTime + " " + avgClassificationTime,
							Logging.LogLevel.PRODUCTION_LOG);

				} catch (Exception exc) {
					exc.printStackTrace();
				}
			} else if (model.compareTo("enn") == 0) {
				// System.out.println("Enter");

				long start = System.currentTimeMillis();

				NearestNeighbour nn = new NearestNeighbour("euclidean");

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

				long endClassification = System.currentTimeMillis();
				double avgClassificationTime = (double)(endClassification-startClassification)/(double)testSet.instances.size();
				
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
								: " ") + " "
						+ elapsedTime + " "
						+ avgClassificationTime,
						Logging.LogLevel.PRODUCTION_LOG);

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
					isvm.svmPKExp = svmPKExp;
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
							+ " " + svmC + " " + svmPKExp + " "
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

				/*
				TransformationFieldsGenerator.getInstance()
						.LoadTransformationFields(merged,
								transformationFieldsFolder);

				TransformationFieldsGenerator.getInstance().warpingWindow = 1.0;

				DataSet merged0 = merged.FilterByLabel(0.0);
				List<List<List<Integer>>> wps = TransformationFieldsGenerator
						.getInstance().GetAllWarpingPaths(merged0);
				int[][] wm = TransformationFieldsGenerator.getInstance()
						.CreateWarpingMatrix(merged0, wps);
				TransformationFieldsGenerator.getInstance().SaveWarpingMatrix(
						wm,
						"/home/josif/Documents/warping_maps/" + dsName
								+ "_full.txt");

				double step = 1.0 / 6.0;

				int cp1 = (int) (merged.numFeatures * step);
				int cp2 = (int) (merged.numFeatures * 3 * step);
				int cp3 = (int) (merged.numFeatures * 5 * step);

				TransformationFieldsGenerator.getInstance()
						.InitializeTheControlPoints();
				TransformationFieldsGenerator.getInstance()
						.ApplyFilteredWarpingPaths(merged0, cp1, 0, true);

				List<TimeSeriesControlPoints> cps = TransformationFieldsGenerator
						.getInstance().controlPoints;

				for (int cpIndex = 0; cpIndex < cps.size(); cpIndex++) {
					Logging.print("cp1" + (cpIndex + 1) + "=[",
							Logging.LogLevel.INFORMATIVE_LOG);
					TimeSeriesControlPoints cp = cps.get(cpIndex);
					Logging.print(cp.warpedPts,
							Logging.LogLevel.INFORMATIVE_LOG);
					Logging.println("];", Logging.LogLevel.INFORMATIVE_LOG);
				}

				TransformationFieldsGenerator.getInstance()
						.ApplyFilteredWarpingPaths(merged0, cp2, 0, true);

				for (int cpIndex = 0; cpIndex < cps.size(); cpIndex++) {
					Logging.print("cp2" + (cpIndex + 1) + "=[",
							Logging.LogLevel.INFORMATIVE_LOG);
					TimeSeriesControlPoints cp = cps.get(cpIndex);
					Logging.print(cp.warpedPts,
							Logging.LogLevel.INFORMATIVE_LOG);
					Logging.println("];", Logging.LogLevel.INFORMATIVE_LOG);
				}

				TransformationFieldsGenerator.getInstance()
						.ApplyFilteredWarpingPaths(merged0, cp3, 0, true);

				for (int cpIndex = 0; cpIndex < cps.size(); cpIndex++) {
					Logging.print("cp3" + (cpIndex + 1) + "=[",
							Logging.LogLevel.INFORMATIVE_LOG);
					TimeSeriesControlPoints cp = cps.get(cpIndex);
					Logging.print(cp.warpedPts,
							Logging.LogLevel.INFORMATIVE_LOG);
					Logging.println("];", Logging.LogLevel.INFORMATIVE_LOG);
				}
				*/

				/*
				 * List<List<List<Integer>>> wps1r =
				 * TransformationFieldsGenerator
				 * .getInstance().GetFilteredWarpingPaths(merged0, cp1, 0,
				 * true); int [][] wm1r =
				 * TransformationFieldsGenerator.getInstance
				 * ().CreateWarpingMatrix(merged0, wps1r);
				 * TransformationFieldsGenerator
				 * .getInstance().SaveWarpingMatrix(wm1r,
				 * "/home/josif/Documents/warping_maps/"+ dsName +"_1r.txt");
				 * 
				 * List<List<List<Integer>>> wps2r =
				 * TransformationFieldsGenerator
				 * .getInstance().GetFilteredWarpingPaths(merged0, cp2, 0,
				 * true); int [][] wm2r =
				 * TransformationFieldsGenerator.getInstance
				 * ().CreateWarpingMatrix(merged0, wps2r);
				 * TransformationFieldsGenerator
				 * .getInstance().SaveWarpingMatrix(wm2r,
				 * "/home/josif/Documents/warping_maps/"+ dsName +"_2r.txt");
				 * 
				 * List<List<List<Integer>>> wps3r =
				 * TransformationFieldsGenerator
				 * .getInstance().GetFilteredWarpingPaths(merged0, cp3, 0,
				 * true); int [][] wm3r =
				 * TransformationFieldsGenerator.getInstance
				 * ().CreateWarpingMatrix(merged0, wps3r);
				 * TransformationFieldsGenerator
				 * .getInstance().SaveWarpingMatrix(wm3r,
				 * "/home/josif/Documents/warping_maps/"+ dsName +"_3r.txt");
				 */

				/*
				 * List<List<List<Integer>>> wps1r =
				 * TransformationFieldsGenerator
				 * .getInstance().GetFilteredWarpingPaths(merged0, cp1, 0,
				 * true); int [][] wm1r =
				 * TransformationFieldsGenerator.getInstance
				 * ().CreateWarpingMatrix(merged0, wps1r);
				 * TransformationFieldsGenerator
				 * .getInstance().SaveWarpingMatrix(wm1r,
				 * "/home/josif/Documents/warping_maps/"+ dsName +"_1r.txt");
				 * List<List<List<Integer>>> wps1l =
				 * TransformationFieldsGenerator
				 * .getInstance().GetFilteredWarpingPaths(merged0, cp1, 0,
				 * false); int [][] wm1l =
				 * TransformationFieldsGenerator.getInstance
				 * ().CreateWarpingMatrix(merged0, wps1l);
				 * TransformationFieldsGenerator
				 * .getInstance().SaveWarpingMatrix(wm1l,
				 * "/home/josif/Documents/warping_maps/"+ dsName +"_1l.txt");
				 * 
				 * 
				 * int part2 = (int)((2.0*merged.numFeatures)/3.0);
				 * List<List<List<Integer>>> wps2r =
				 * TransformationFieldsGenerator
				 * .getInstance().GetFilteredWarpingPaths(merged0, part2, 0,
				 * true); int [][] wm2r =
				 * TransformationFieldsGenerator.getInstance
				 * ().CreateWarpingMatrix(merged0, wps2r);
				 * TransformationFieldsGenerator
				 * .getInstance().SaveWarpingMatrix(wm2r,
				 * "/home/josif/Documents/warping_maps/"+ dsName +"_2r.txt");
				 * 
				 * List<List<List<Integer>>> wps2l =
				 * TransformationFieldsGenerator
				 * .getInstance().GetFilteredWarpingPaths(merged0, part2, 0,
				 * false); int [][] wm2l =
				 * TransformationFieldsGenerator.getInstance
				 * ().CreateWarpingMatrix(merged0, wps2l);
				 * TransformationFieldsGenerator
				 * .getInstance().SaveWarpingMatrix(wm2l,
				 * "/home/josif/Documents/warping_maps/"+ dsName +"_2l.txt");
				 */

				/*
				 * BufferedImage bi
				 * =TransformationFieldsGenerator.getInstance().
				 * CreateImage(merged0, wps); BufferedImage bi1r
				 * =TransformationFieldsGenerator
				 * .getInstance().CreateImage(merged0, wps1r); BufferedImage
				 * bi1l
				 * =TransformationFieldsGenerator.getInstance().CreateImage(
				 * merged0, wps1l); BufferedImage bi2r
				 * =TransformationFieldsGenerator
				 * .getInstance().CreateImage(merged0, wps2r); BufferedImage
				 * bi2l
				 * =TransformationFieldsGenerator.getInstance().CreateImage(
				 * merged0, wps2l);
				 * 
				 * 
				 * try { ImageIO.write(bi,"PNG",new File(
				 * "/home/josif/warping_map" + ".png"));
				 * ImageIO.write(bi1r,"PNG",new File(
				 * "/home/josif/warping_map1r" + ".png"));
				 * ImageIO.write(bi1l,"PNG",new File(
				 * "/home/josif/warping_map1l" + ".png"));
				 * ImageIO.write(bi2r,"PNG",new File(
				 * "/home/josif/warping_map2r" + ".png"));
				 * ImageIO.write(bi2l,"PNG",new File(
				 * "/home/josif/warping_map2l" + ".png")); } catch(Exception
				 * exc) { exc.printStackTrace(); }
				 */

				// Trans
			} else if (model.compareTo("randTest") == 0) {
				Random rand = new Random();

				for (int i = 0; i < 20; i++) {
					int n = rand.nextInt();

					System.out.println(n + "\n");
				}
			}
			else if (model.compareTo("sqf") == 0){

				SqueezeKernelFactorization sq = new SqueezeKernelFactorization(trainSet, testSet, k, 
									svmC, svmPKExp, beta, lambdaU, lambdaV, learnRateR, learnRateCA, maxEpocs);   
				
				double mcrTest = sq.Optimize();
				
				Logging.println(
						mcrTest + " " 
						+ ", dataSet=" + dsName 
						+ ", fold=" + fold
						+ ", model=" + model 
						+ ", mode=" + runMode 
						+ ", k=" + k
						+ ", lambdaU=" + lambdaU
						+ ", lambdaV=" + lambdaV
						+ ", learnRateR=" + learnRateR
						+ ", learnRateCA=" + learnRateCA
						+ ", beta=" + beta 
						+ ", margin=" + maxMargin 
						+ ", svmC="+svmC 
						+ ", svmPKExp=" + svmPKExp
						+ ", maxEpocs=" + maxEpocs,
						Logging.LogLevel.PRODUCTION_LOG);
				
			}

		}

	}

}
