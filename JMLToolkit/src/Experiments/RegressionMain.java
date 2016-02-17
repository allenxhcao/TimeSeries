package Experiments;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import Classification.Kernel;
import Classification.Kernel.KernelType;
import Classification.KernelFactorization;
import Classification.MFSVM;
import Classification.NaiveSmo;
import Classification.NearestNeighbour;
import Classification.PCASVM;
import Classification.WekaClassifierInterface;
import DataStructures.DataSet;
import DataStructures.Matrix;
import MatrixFactorization.FactorizationsCache;
import MatrixFactorization.NonlinearlySupervisedMF;
import Regression.LSSVM;
import Regression.MotifDiscoveryRegression;
import Regression.SRLP;
import Regression.SRLP_Plus;
import Regression.SRNP;
import Regression.SVD_LSSVM;
import Regression.SVMRegression;
import TimeSeries.Distorsion;
import Utilities.IOUtils;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class RegressionMain 
{
	public static void main(String[] args) 
	{
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		
		if (args.length == 0) 
		{
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;
			
			String dir = "E:\\Data\\regression\\",
			ds = "restaurantRevenue", 
			foldNo = "competition";
 
			String sp = File.separator; 
			
			args = new String[] {  

				"fold=" + foldNo,  
				"model=srnp", 
				//"model=svm",
				//"model=lssvm",  
				//"model=srnp", 
				"runMode=test",    

				"trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp   
						+ ds + "_TRAIN",  
				"validationSet=" + dir + ds + sp + "folds" + sp + foldNo  
						+ sp + ds + "_VALIDATION",  
				"testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp  
						+ ds + "_TEST",   
				"presenceRatio=1.0", 
				"learnRate=0.001",  
				"maxEpocs=3000", 
				"lambdaU=0.0001",  
				"lambdaV=0.001", 
				"lambdaW=0.001", 
				"beta=0.5", 
				"latentDimensionsRatio=0.5",
				"semiSupervised=true", 
				"kernel=polynomial", 
				"svmC=1",
				"svmDegree=3",
				"sig2=0.1",
				"n=40",
				"a=6",
				"bopDegree=3",
				"innerDimension=3"
			};
		}

		if (args.length > 0) {
			
			// set model values
			String model = "", runMode = ""; 
			String trainSetPath = "", validationSetPath = "", testSetPath = "", 
					fold = "", factorizationsCacheFolder = "", transformationFieldsFolder = "";
			
			int svmDegree = 3;
			int n = 20, a=6, bopDegree=4, innerDimension = 4;
			
			double alpha = 0.8, 
					beta = 0.1, 
					learnRate = 0.01, svmC = 1.0, 
					sig2 = 1.0, 
					lambdaU = 0.01, lambdaV = 0.01, lambdaW = 0.01, 
					lambdaV0 = 0.01, lambdaW0 = 0.01,
					presenceRatio = 1.0; 
			
			double latentDimensionsRatio = 0; 
			
			boolean semiSupervised = true; 
			
			int k = -1, maxEpocs = 100; 
			String kernel = "polynomial"; 

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
				else if (argTokens[0].compareTo("fold") == 0)
					fold = argTokens[1];
				else if (argTokens[0].compareTo("latentDimensionsRatio") == 0)
					latentDimensionsRatio = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("dimensions") == 0)
					k = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("learnRate") == 0)
					learnRate = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("alpha") == 0)
					alpha = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("beta") == 0)
					beta = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("kernel") == 0)
					kernel = argTokens[1];	
				else if (argTokens[0].compareTo("svmC") == 0)  
					svmC = Integer.parseInt(argTokens[1]); 
				else if (argTokens[0].compareTo("svmDegree") == 0)
					svmDegree = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("sig2") == 0)
					sig2 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaU") == 0) 
					lambdaU = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaV0") == 0)
					lambdaV0 = Double.parseDouble(argTokens[1]); 
				else if (argTokens[0].compareTo("lambdaW0") == 0)
					lambdaW0 = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("presenceRatio") == 0)
					presenceRatio = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaV") == 0)
					lambdaV = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaW") == 0)
					lambdaW = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("maxEpocs") == 0)
					maxEpocs = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("n") == 0)
					n = Integer.parseInt(argTokens[1]); 
				else if (argTokens[0].compareTo("a") == 0)
					a = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("bopDegree") == 0)
					bopDegree = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("innerDimension") == 0)
					innerDimension = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("semiSupervised") == 0) 
					semiSupervised = argTokens[1].toUpperCase().compareTo(
							"TRUE") == 0 ? true : false;
			}
			
			
			
			// read the train validation and test set
			List<List<Double>> trainSet = IOUtils.LoadFile(trainSetPath),
						validationSet = IOUtils.LoadFile(validationSetPath),
						testSet = IOUtils.LoadFile(testSetPath); 
			
				
			// the dataset predictors and targets

			Matrix tmpTrainPredictors = null,
					tmpValidationPredictors = null,
					tmpTestPredictors = null;
					
			boolean ommitTargets = true;
			
			// load the predictors, ommit targets if required
			tmpTrainPredictors = new Matrix();
			tmpTrainPredictors.LoadRegressionData(trainSet, ommitTargets);
			tmpValidationPredictors = new Matrix();
			tmpValidationPredictors.LoadRegressionData(validationSet, ommitTargets);
			tmpTestPredictors = new Matrix();
			tmpTestPredictors.LoadRegressionData(testSet, ommitTargets);
			
			Logging.println("tmpTrainPredictors: " + tmpTrainPredictors.getDimRows() + ", " + tmpTrainPredictors.getDimColumns());
			Logging.println("tmpTestPredictors: " + tmpTestPredictors.getDimRows() + ", " + tmpTestPredictors.getDimColumns());
			
			Matrix trainPredictors = null,
					testPredictors = null;
			
			double [] trainTargets = null,
					testTargets = null;
		
			if( runMode.compareTo("validation") == 0)
			{
				trainPredictors = new Matrix(tmpTrainPredictors);
				testPredictors = new Matrix(tmpValidationPredictors);
				
				// load train as train 
				int nTrain = trainSet.size();
				trainTargets = new double[nTrain];
				for(int i = 0; i < nTrain; i++) 
					trainTargets[i] = trainSet.get(i).get(trainSet.get(i).size()-1);
				
				// load test as validation
				int nTest = validationSet.size(); 
				testTargets = new double[nTest];
				for(int i = 0; i < nTest; i++)
					testTargets[i] = validationSet.get(i).get(validationSet.get(i).size()-1);
					
			}
			else
			{
				trainPredictors = new Matrix(tmpTrainPredictors);
				trainPredictors.AppendMatrix(tmpValidationPredictors);
				testPredictors = new Matrix(tmpTestPredictors);
		
				// load train as train 
				int nTrain = trainSet.size();
				int nValidation = validationSet.size();
				
				trainTargets = new double[nTrain+nValidation];
				
				for(int i = 0; i < nTrain; i++) 
					trainTargets[i] = trainSet.get(i).get(trainSet.get(i).size()-1);
				
				for(int i = nTrain; i < nTrain+nValidation; i++) 
					trainTargets[i] = validationSet.get(i-nTrain).get(validationSet.get(i-nTrain).size()-1);
				
				// load test as test
				int nTest = tmpTestPredictors.getDimRows();
				testTargets = new double[nTest];
				for(int i = 0; i < nTest; i++)
					testTargets[i] = testSet.get(i).get(testSet.get(i).size()-1);
				
				Logging.println("nTrain=" + nTrain + ", nTest=" + nTest);
				Logging.println("Train dim: " + trainPredictors.getDimRows() + ", " + trainPredictors.getDimColumns());
				
			}
		
			/*
			Matrix alltogether = new Matrix(trainPredictors);
			alltogether.AppendMatrix(testPredictors);
			alltogether.SaveTripples("/home/josif/a.tripples");
			
			System.out.println("---------------");
			for(int i = 0; i < trainTargets.length;i++)
				System.out.println(trainTargets[i]);
			for(int i = 0; i < testTargets.length;i++)
				System.out.println(testTargets[i]);
			System.out.println("---------------");
			
			if(true)
				return;
			*/
		
			// either the latend dimensions are set directly or the latentDimRatio is provided
			if ( k < 0)
			{
				k = (int)Math.ceil( (latentDimensionsRatio * trainPredictors.getDimColumns()) );
			}
			
			
			if (model.compareTo("lssvm") == 0) {
				
				int NTrain = trainPredictors.getDimRows(); 
				int NTest = testPredictors.getDimRows(); 
				 
				int M = trainPredictors.getDimColumns(); 
				int presentLabels = (int) Math.ceil( presenceRatio * NTrain ); 
				
				Matrix newTrainPredictors = new Matrix( presentLabels, M);
				double [] newTrainTargets = new double[presentLabels];
				Matrix newTestPredictors = new Matrix(NTest + NTrain - presentLabels, M);
				double [] newTestTargets = new double[NTest + NTrain - presentLabels];

				// set the train predictors and targets
				for(int i = 0; i < presentLabels; i++)
				{
					newTrainPredictors.SetRow(i, trainPredictors.getRow(i));
					newTrainTargets[i] = trainTargets[i];
				}
				
				// set the test predictors and targets
				for(int i = 0; i < NTest; i++)
				{
					newTestPredictors.SetRow(i, testPredictors.getRow(i)); 
					newTestTargets[i] = testTargets[i];
				}
				for(int i = 0; i < NTrain-presentLabels; i++)
				{
					newTestPredictors.SetRow(NTest+i, trainPredictors.getRow(presentLabels + i));
					newTestTargets[NTest+i] = trainTargets[presentLabels + i]; 
				}
				
				LSSVM lssvm = new LSSVM();
				lssvm.beta = 1.0;
				lssvm.lambda = svmC;
				
				lssvm.kernel = new Kernel();
			
				if( kernel.compareTo("polynomial") == 0 )
				{
					lssvm.kernel.type = KernelType.Polynomial;
					lssvm.kernel.degree = svmDegree;
				}
				else if( kernel.compareTo("gaussian") == 0 )
				{
					lssvm.kernel.type = KernelType.Gaussian;
					lssvm.kernel.sig2 = sig2;
				}
				else if( kernel.compareTo("linear") == 0 )
				{
					lssvm.kernel.type = KernelType.Linear;
				}
				
				lssvm.Train(newTrainPredictors, newTrainTargets); 
				
				double mseTrain = lssvm.PredictTrainSet(); 
				double mseTest = lssvm.PredictTestSet(newTestPredictors, newTestTargets);
				
				Logging.println(
						mseTest +
						" " + model+ 
						" fold="+fold+
						" mode="+runMode+
						" svmC="+svmC+
						" svmDegree="+svmDegree+
						" kernel="+kernel+						
						" sig2="+sig2 + 
						" presence=" + presenceRatio,
						LogLevel.PRODUCTION_LOG);  
				
			}
			else if( model.compareTo("srlp") == 0)
			{
				SRLP srlp = new SRLP();	
				srlp.lambdaU = lambdaU;	
				srlp.lambdaV = lambdaV;	
				srlp.lambdaW = lambdaW; 	
				srlp.D = k; 
				srlp.numIter = maxEpocs; 
				srlp.beta = beta; 
				srlp.eta = learnRate; 
				srlp.presentLabelsRatio = presenceRatio;
				
				double mseTrain=0, mseTest=0; 
				
				if( semiSupervised )
				{	
					srlp.testTargets = testTargets;
					
					srlp.Train(trainPredictors, trainTargets, testPredictors);
					mseTrain = srlp.PredictTrainSet();
					mseTest = srlp.PredictTestSet(testTargets);
				}
				else
				{
					srlp.Train(trainPredictors, trainTargets);
					mseTrain = srlp.PredictTrainSet();
					mseTest = srlp.PredictTestSet(testPredictors, testTargets);
				}
				
				System.out.printf("%.12f"+  
						" " + model+
						" fold="+fold+
						" mode="+runMode+
						" lambdaU="+lambdaU+ 
						" lambdaV="+lambdaV+ 
						" lambdaW="+lambdaW+ 
						" dim="+srlp.D+
						" eta="+learnRate+
						" beta="+beta+
						" maxEpocs="+maxEpocs+
						" presenceRatio="+presenceRatio+ 
						" mseTrain="+mseTrain + "\n", mseTest); 
				
			}
			else if( model.compareTo("srnp") == 0)
			{
				SRNP srnp = new SRNP();	
				srnp.lambdaU = lambdaU;	
				srnp.lambdaV = lambdaV;	
				srnp.lambdaW = lambdaW;	
				srnp.D = k;				
				srnp.maxEpocs = maxEpocs;
				srnp.eta = learnRate; 
				srnp.beta = beta;
				srnp.presenceRatio = presenceRatio;
				
				double mseTrain=0, mseTest=0; 
				
				srnp.kernel = new Kernel();
				if( kernel.compareTo("polynomial") == 0 )
				{
					srnp.kernel.type = KernelType.Polynomial;
					srnp.kernel.degree = svmDegree;
				}
				else if( kernel.compareTo("gaussian") == 0 )
				{
					srnp.kernel.type = KernelType.Gaussian;
					srnp.kernel.sig2 = sig2;
				}
				else if( kernel.compareTo("linear") == 0 )
				{
					srnp.kernel.type = KernelType.Linear;
				}
				
				srnp.testTargets = testTargets;
				
				srnp.Train(trainPredictors, trainTargets, testPredictors);
				
				mseTest = srnp.PredictTestSet(testTargets);
				mseTrain = srnp.PredictTrainSet();
				
				System.out.printf("%.12f"+ 
						" " + model+ 
						" fold="+fold+ 
						" mode="+runMode+ 
						" lambdaU="+lambdaU+ 
						" lambdaV="+lambdaV+ 
						" lambdaW="+lambdaW+ 
						" dim="+srnp.D+ 
						" eta="+learnRate+ 
						" beta="+beta+ 
						" degree="+svmDegree+ 
						" maxEpocs="+maxEpocs+ 
						" presenceRatio="+presenceRatio+ 
						" mseTrain="+mseTrain + "\n", mseTest);
			}
			else if (model.compareTo("svm") == 0) { 
				
				SVMRegression svmRegression = new SVMRegression(); 
				svmRegression.kernel = kernel; 
				svmRegression.svmC = lambdaV; 
				svmRegression.degree = svmDegree; 
				svmRegression.gamma = sig2; 
				
				int NTrain = trainPredictors.getDimRows(); 
				int NTest = testPredictors.getDimRows(); 
				 
				int M = trainPredictors.getDimColumns(); 
				int presentLabels = (int) Math.ceil( presenceRatio * NTrain ); 
				
				
				Matrix newTrainPredictors = new Matrix( presentLabels, M);
				double [] newTrainTargets = new double[presentLabels];
				Matrix newTestPredictors = new Matrix(NTest + NTrain - presentLabels, M);
				double [] newTestTargets = new double[NTest + NTrain - presentLabels];

				// set the train predictors and targets
				for(int i = 0; i < presentLabels; i++)
				{
					newTrainPredictors.SetRow(i, trainPredictors.getRow(i));
					newTrainTargets[i] = trainTargets[i];
				}
				
				// set the test predictors and targets
				for(int i = 0; i < NTest; i++)
				{
					newTestPredictors.SetRow(i, testPredictors.getRow(i)); 
					newTestTargets[i] = testTargets[i];
				}
				for(int i = 0; i < NTrain-presentLabels; i++)
				{
					newTestPredictors.SetRow(NTest+i, trainPredictors.getRow(presentLabels + i));
					newTestTargets[NTest+i] = trainTargets[presentLabels + i]; 
				}
				
				double mseTest = svmRegression.Regress(newTrainPredictors, newTrainTargets, 
						newTestPredictors, newTestTargets);  
				 
				Logging.println(
						mseTest +
						" " + model+ 
						" fold="+fold+
						" mode="+runMode+
						" lambdaV="+lambdaV+
						" kernel="+kernel+
						" degree="+svmDegree+
						" sig2="+sig2 + 
						" presence=" + presenceRatio,
						LogLevel.PRODUCTION_LOG);  
				
			}
			else if (model.compareTo("mdr") == 0) {
			
				MotifDiscoveryRegression mdr = new MotifDiscoveryRegression();
				// parameters of the bag of pattern
				mdr.n = n;
				mdr.alpha = a;
				mdr.bopPolyDegree = bopDegree;
				mdr.innerDimension = innerDimension;
				
				// parameters of the ls-svm
				mdr.C = svmC;
				mdr.svmPolyDegree = svmDegree;
				
				Matrix allPred = new Matrix(trainPredictors);
				allPred.AppendMatrix(testPredictors);
				allPred.SaveToFile("C:\\Users\\josif\\Desktop\\infati.csv"); 
				
				// create the histogram from the data
				mdr.CreateHistogram(trainPredictors, testPredictors, trainTargets, testTargets);
				
				int numVars = 196;
				int [] subsetVars = mdr.SelectVariables(numVars);
				
				for(int v = 0; v < numVars; v++)
				{
					System.out.println("Pattern: " + mdr.bop.dictionary.get( subsetVars[v] ) );
				}
				
				// debug print labels
				//for(int i = 0; i<trainTargets.length;i++) System.out.println(trainTargets[i]);
				//for(int i = 0; i<testTargets.length;i++) System.out.println(testTargets[i]);
				
				
				double mse = 0;
			
				
				Logging.println(
						mse +
						" " + model+ 
						" fold="+fold+
						" mode="+runMode+
						" lambdaV="+lambdaV+
						" kernel="+kernel+
						" degree="+svmDegree+
						" sig2="+sig2 + 
						" presence=" + presenceRatio,
						LogLevel.PRODUCTION_LOG);  
				
			}
		}

	}

}
