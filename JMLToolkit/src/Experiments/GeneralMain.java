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

public class GeneralMain {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		if (args.length == 0) {
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;
			 //Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

			String dir = "/mnt/E/Data/classification/uci_binary/", 
			ds = "ionosphere", 
			foldNo = "2";  

			String sp = File.separator;
			
			args = new String[] {
					"fold=" + foldNo,  
					"model=nsdr", 
					// "model=cleandtwnn",
					//"model=dtwnn",
					// "model=enn",
					//"model=pcasvm",
					//"model=svm",
					// "model=transformationField",
					// "model=randTest",
					"runMode=test",
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
					"learnRateR=0.001",
					"learnRateCA=0.001",   
					"lambdaU=0.000001", 
					"lambdaV=0.000001", 
					//"latentDimensionsRatio=0.5", 
					"dimensions=17",
					"beta=0.5", 
					"maxEpocs=200",
					"svmC=1",
					"svmKernel=polynomial",
					"svmPKExp=2",  
					
					"learnRate=0.000", 
					"svmRKGamma=1",   
					"warpingWindowFraction=0.5", 
					"svmKernel=polynomial", 
					"distorsion=false", 
					"epsilon=0.8",
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

			}
			else if (model.compareTo("svm") == 0) {
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

			}else if (model.compareTo("kf") == 0) {
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

			}
			else if (model.compareTo("nsdr") == 0){

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
					
				NonlinearlySupervisedMF sd = new NonlinearlySupervisedMF(
								ultimateTrain, ultimateTest, k, 
								svmKernel, svmC, svmPKExp, svmRKGamma,
								beta, lambdaU, lambdaV, 
								learnRateR, learnRateCA, maxEpocs);   
				
				long startTime = System.currentTimeMillis();
				
				double mcrTest = sd.Optimize();
				
				long endTime = System.currentTimeMillis();
				
				Logging.println(
						mcrTest + " " 
						+ ", loss=" + sd.lastLoss
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
						+ ", svmC="+svmC
						+ ", svmKernel="+svmKernel 
						+ ", svmPKExp=" + svmPKExp
						+ ", maxEpocs=" + maxEpocs 
						+ ", time=" + (endTime-startTime), 
						Logging.LogLevel.PRODUCTION_LOG);
					//}
				//}
				
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
		        
		        Kernel kernel = null;
		        if(svmKernel.compareTo("polynomial") == 0)
		        {
		        	kernel = new Kernel(Kernel.KernelType.Polynomial);
		        	kernel.degree = (int)svmPKExp;
		        }
		        else if(svmKernel.compareTo("gaussian") == 0)
		        {
		        	
		        	kernel = new Kernel(Kernel.KernelType.Gaussian);
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
				
				// initialize a pcasvm				
				PCASVM pcasvm = new PCASVM();
				pcasvm.variance = beta;
				pcasvm.svmC = svmC;
				pcasvm.kernelType = svmKernel;
				pcasvm.svmPolynomialKernelExp = svmPKExp;
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
						+ ", svmPKExp=" + svmPKExp,
						Logging.LogLevel.PRODUCTION_LOG);
				
			}

		}

	}

}
