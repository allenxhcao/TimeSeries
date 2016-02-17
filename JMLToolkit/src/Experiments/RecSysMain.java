package Experiments;

import Classification.*;
import Classification.Kernel.KernelType;
import Utilities.*;
import Clustering.KMedoids;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.Matrix;
import DataStructures.Tripples;
import MatrixFactorization.BiasedMatrixFactorization;
import MatrixFactorization.CollaborativeImputation;
import MatrixFactorization.FactorizationsCache;
import MatrixFactorization.NonlinearlySupervisedMF;
import MatrixFactorization.UnbiasedMatrixFactorization;
import Regression.RPNP;
import TimeSeries.*;
import Utilities.ExpectationMaximization;
import Utilities.Logging;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintStream;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

import javax.imageio.ImageIO;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.gui.visualize.PNGWriter;

public class RecSysMain {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		if (args.length == 0) 
		{
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

			String dir = "/media/josif/PhD/data/movielens100k/ml-100k/",  
			//String dir = "F:\\data\\movielens100k\\ml-100k\\",
			//String dir = "F:\\data\\movielens1m\\leave-one-out\\", 
			ds = "movielens", 
			foldNo = "1";  

			String sp = File.separator; 
			
			args = new String[] {
					"fold=" + foldNo,  
					"model=rpnp",
					"trainSet=" + dir + sp + "u"+ foldNo +".base",
					"validationSet=" + dir + sp + "u"+ foldNo +".base",
					"testSet=" + dir + sp + "u"+ foldNo +".test",
					"factorizationsCacheFolder=/home/josif/Documents/factorizations_cache",
					"learnRate=1E-13", 
					"lambdaU=0.001",
					"lambdaV=0.001",  
					"dimensions=20", 
					"maxEpocs=1000", 
					"kernel=polynomial",  
					"degree=2", 
					"unsupervisedFactorization=true", 
					"transformationFieldsFolder=/home/josif/Documents/transformation_fields", 
					"useTransformationFields=true",  
					"showAllWarnings=false", 
					};
		}

		if (args.length > 0) {
			// set model values
			String model = "", runMode = "";
			String trainSetPath = "", validationSetPath = "", testSetPath = "", fold = "", factorizationsCacheFolder = "", transformationFieldsFolder = "";
			double lambda = 0.001, alpha = 0.8, beta = 0.1, learnRate = 0.01, 
					gamma = 1, maxMargin = 1.0;
			double lambdaU=0, lambdaV=0, learnRate2 = 0, learnRateCA = 0;
			double latentDimensionsRatio = 0;
			double transformationsRate = 1.0;
			double warpingWindowFraction = 0.1;
			int k = -1, maxEpocs = 20000, degree = 2;
			double totalMissingRatio = 0.0, gapRatio = 0.0;
			double epsilon = 0.00;
			String interpolationTechnique = "linear";
			String kernel = "rbf";
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
				else if (argTokens[0].compareTo("learnRate2") == 0)
					learnRate2 = Double.parseDouble(argTokens[1]);
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
				else if (argTokens[0].compareTo("kernel") == 0)
					kernel = argTokens[1];
				else if (argTokens[0].compareTo("gamma") == 0) 
					gamma = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("degree") == 0) 
					degree = Integer.parseInt(argTokens[1]);
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
			Tripples trainSet, testSet = null;
			
			if( runMode.compareTo("test") == 0 )
			{
				trainSet = new Tripples(trainSetPath);
				//trainSet.ReadTripples(validationSetPath, true);
				testSet = new Tripples(testSetPath);
			}
			else
			{
				trainSet = new Tripples(trainSetPath);
				testSet = new Tripples(validationSetPath);
			}
			
			

			// initialize lambdaU and lambdaV in case just a lambda is specified
			if(lambdaU == 0 && lambda > 0) lambdaU = lambda;
			if(lambdaV == 0 && lambda > 0) lambdaV = lambda;

			String dsName = new File(trainSetPath).getName().split("_TRAIN")[0];

			Logging.println("DataSet: " + dsName + ", model:" + model
					+ ", runMode=" + runMode + ", k=" + k + ", kernel=" + kernel 
					+ ", lambdaU=" + lambdaU + ", lambdaV=" + lambdaV + ", gamma=" + gamma + ", degree=" + degree 
					+ ", learnRate=" + learnRate + ", maxEpocs=" + maxEpocs,
					Logging.LogLevel.DEBUGGING_LOG);
			
			if (model.compareTo("bmf") == 0) 
			{ 
				BiasedMatrixFactorization bmf = new BiasedMatrixFactorization(); 
				bmf.lambdaU = lambdaU; 
				bmf.lambdaV = lambdaV; 
				bmf.eta = learnRate; 
				bmf.maxEpocs = maxEpocs; 
				bmf.K = k; 
				bmf.trainData = trainSet; 
				bmf.testData = testSet; 

				bmf.FixIndices();
				bmf.Initialize();
				bmf.Decompose(); 
				
			}
			else if (model.compareTo("umf") == 0) 
			{
				UnbiasedMatrixFactorization umf = new UnbiasedMatrixFactorization();
				umf.lambdaU = lambdaU;
				umf.lambdaV = lambdaV;
				umf.eta = learnRate;
				umf.maxEpocs = maxEpocs;
				umf.K = k;
				umf.trainData = trainSet;
				umf.testData = testSet; 

				umf.FixIndices();
				umf.Initialize();
				umf.Decompose();
				
				double testRMSE = umf.ComputeRMSE(1);
				
				Logging.println(
						testRMSE + " " 
						+ ", dataSet=" + dsName 
						+ ", fold=" + fold
						+ ", model=" + model 
						+ ", mode=" + runMode 
						+ ", lambdaU=" + lambdaU
						+ ", lambdaV=" + lambdaV
						+ ", k=" + k
						+ ", eta=" + learnRate
						+ ", maxEpocs=" + maxEpocs,
						Logging.LogLevel.PRODUCTION_LOG); 
				
			}
			else if (model.compareTo("rpnp") == 0) 
			{
				RPNP rpnp = new RPNP();
				rpnp.lambdaU = lambdaU;
				rpnp.lambdaV = lambdaV;
				rpnp.eta = learnRate;
				rpnp.D = k;
				rpnp.maxEpocs = maxEpocs;
				
				rpnp.kernel = new Kernel();
				if( kernel.compareTo("polynomial") == 0 )
				{
					rpnp.kernel.type = KernelType.Polynomial;
					rpnp.kernel.degree = degree;
				}
				
				rpnp.Train(trainSet, testSet);
				
				double testMAE = rpnp.Predict(testSet);
				
				Logging.println(
						testMAE + " " 
						+ ", dataSet=" + dsName 
						+ ", fold=" + fold
						+ ", model=" + model 
						+ ", mode=" + runMode 
						+ ", lambdaU=" + lambdaU
						+ ", lambdaV=" + lambdaV
						+ ", k=" + k
						+ ", eta=" + learnRate
						+ ", maxEpocs=" + maxEpocs,
						Logging.LogLevel.PRODUCTION_LOG); 
				
			}

		}

	}

}
