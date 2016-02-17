package Experiments;

import DataStructures.DataSet;
import DataStructures.Matrix;
import TimeSeries.*;
import TimeSeries.LearnUnivariateShapelets.LossTypes;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import java.io.File;
import java.util.ArrayList;

public class LocalConvolutionsMain 
{
	public static void main(String[] args) {
		//Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;
		Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;

		if (args.length == 0) {
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG; 

			//String dir = "vartheta:\\Data\\classification\\timeseries\\",
			//String dir = "vartheta:\\Data\\classification\\longts\\",
			String dir = "/mnt/E/Data/classification/timeseries/",
			
			ds = "StarLightCurves",  
			//ds = "Otoliths", 
			foldNo = "default"; 
			
			String sp = File.separator;  
			
			args = new String[] {  
					"fold=" + foldNo, 
					"model=lsg",  
					"runMode=test", 
					"trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp  
							+ ds + "_TRAIN",  
					"validationSet=" + dir + ds + sp + "folds" + sp + foldNo  
							+ sp + ds + "_VALIDATION", 
					"testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp  
							+ ds + "_TEST",  
					//"K=0.3",    
					"lambdaW=0.01",       
					"maxEpochs=10000",    
					"alpha=-30",   
					"eta=0.01",   
					"L=0.15",
					"R=4"
					};  
			 
  
		} 
		
		if (args.length > 0) { 
			// set model values 
			String model = "", runMode = ""; 
			String trainSetPath = "", validationSetPath = "", testSetPath = "", fold = "", factorizationsCacheFolder = "", transformationFieldsFolder = "";
			double eta  =0.001, etaR = 0.001, etaA = 0.001, svmC  = 1.0;
			int  maxEpochs = 1000, degree = 3, M = 3, R = 2, scale=3, alpha = 0;
			double lambdaP=0, lambdaD=0, lambdaBeta=0, lambdaW = 0, lambdaS = 0, 
					gamma=1,beta = 0;
			double L = 0.1, K=0.5;
			double latentDimensionsRatio = 0.2;
			double deltaT = 0;

			
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
				else if (argTokens[0].compareTo("K") == 0)
					K = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("eta") == 0) 
					eta = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("etaR") == 0)
					etaR = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("etaA") == 0)
					etaA = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaF") == 0)
					lambdaP = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaD") == 0)
					lambdaD = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaW") == 0)
					lambdaW = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("svmC") == 0)
					svmC = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaS") == 0)
					lambdaS = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("lambdaBeta") == 0)
					lambdaBeta = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("alpha") == 0)
					alpha = Integer.parseInt(argTokens[1]); 
				else if (argTokens[0].compareTo("gamma") == 0)
					gamma = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("beta") == 0)
					beta = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("maxEpochs") == 0)
					maxEpochs = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("deltaT") == 0)
					deltaT = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("degree") == 0)
					degree = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("R") == 0)
					R = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("M") == 0)
					M = Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("L") == 0)
					L = Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("scale") == 0)
					scale = Integer.parseInt(argTokens[1]);
			}


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

			String dsName = new File(trainSetPath).getName().split("_TRAIN")[0];

			if (model.compareTo("lc") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

				Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            ConvolutionLocalPatterns lc = new ConvolutionLocalPatterns();
	            // initialize the sizes of data structures
	            lc.NTrain = finalTrainSet.GetNumInstances();
	            lc.NTest = testSet.GetNumInstances();
	            lc.M = X.getDimColumns();
	            // set the time series and labels
	            lc.T = X;
	            lc.Y = Y;
	            // set the learn rate and the number of iterations
	            lc.maxIter = maxEpochs;
	            // set te number of patterns
	            lc.K = (int)(K*X.getDimColumns());
	            // set the size of segments
	            lc.L = (int)(L*X.getDimColumns());
	            // set the regularization parameter
	            lc.lambdaP = lambdaP;
	            lc.lambdaD = lambdaD;
	            lc.lambdaW = lambdaW;
	            lc.alpha = alpha;
	            
	            lc.deltaT = (int)(deltaT*lc.L);
	            
	            lc.dataset = dsName;
	            lc.fold = fold;
	            
	            // learn the local convolutions
	            errorRate = lc.Learn();

	            
				System.out.println( String.valueOf(errorRate)
						+ " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model	+ " "
						+ lambdaD + " "
						+ lambdaP + " "
						+ K	+ " "
						+ L	+ " "
						+ eta + " "
						+ maxEpochs);
						
			}
			else if (model.compareTo("lcc") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
	            Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true); 
	            
	            ConvolutionLocalPatterns lcnr = new ConvolutionLocalPatterns();
	            // initialize the sizes of data structures
	            lcnr.NTrain = finalTrainSet.GetNumInstances();
	            lcnr.NTest = testSet.GetNumInstances();
	            lcnr.M = X.getDimColumns();
	            // set the time series and labels
	            lcnr.T = X;
	            lcnr.Y = Y;
	            // set the learn rate and the number of iterations
	            lcnr.maxIter = maxEpochs;
	            // set te number of patterns
	            lcnr.K = (int)(K*X.getDimColumns());
	            // set the size of segments
	            lcnr.L = (int)(L*X.getDimColumns());
	            // set the regularization parameter
	            lcnr.lambdaP = lambdaP;
	            lcnr.lambdaD = lambdaD;
	            lcnr.lambdaW = lambdaW;
	            lcnr.alpha = alpha; 	            
	            lcnr.deltaT = (int)(deltaT*lcnr.L);
	            lcnr.dataset = dsName;
	            lcnr.fold = fold;	            
	            
	            // learn the local convolutions
	            errorRate = lcnr.Learn();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model	+ " "
						+ lambdaD + " "
						+ lambdaP + " "
						+ K	+ " "
						+ L	+ " "
						+ eta + " "
						+ maxEpochs);
						
			} 
			else if (model.compareTo("ic") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}
					            
	            Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true); 
	            
	            InvariantConvolution ic = new InvariantConvolution(); 
	            // initialize the sizes of data structures
	            ic.NTrain = finalTrainSet.GetNumInstances(); 
	            ic.NTest = testSet.GetNumInstances(); 
	            ic.M = X.getDimColumns(); 
	            // set the time series and labels 
	            ic.T = X; 
	            ic.Y = Y; 
	            // set the learn rate and the number of iterations 
	            ic.maxIter = maxEpochs; 
	            // set te number of patterns 
	            ic.K = (int)(K*X.getDimColumns()); 
	            // set the size of segments 
	            ic.minWindowSize = (int)(L*X.getDimColumns()); 

	            ic.maxScale = scale; 
	            
	            ic.deltaT = (int)(deltaT*ic.minWindowSize); 
	            
	            ic.lambdaP = lambdaP; 
	            ic.svmC = svmC; 
	            
	            // learn the local convolutions
	            
	            double [] errorRates = ic.Learn();
	            
				System.out.println( String.valueOf(errorRates[0]) + " " + String.valueOf(errorRates[1]) + " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model
						+ " lambdaF=" + lambdaP
						+ " svmC=" + svmC
						+ " K=" + K	
						+ " L=" + L
						+ " scale=" + scale  
						+ " deltaT="+deltaT 
						+ " time="	+ (System.currentTimeMillis() - startTime)  
						+ " epochs=" + maxEpochs);   
						
			}
			else if (model.compareTo("icfoldin") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
	            Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true); 
	            
	            InvariantConvolution ic = new InvariantConvolution(); 
	            // initialize the sizes of data structures
	            ic.NTrain = finalTrainSet.GetNumInstances(); 
	            ic.NTest = testSet.GetNumInstances(); 
	            ic.M = X.getDimColumns(); 
	            // set the time series and labels 
	            ic.T = X; 
	            ic.Y = Y; 
	            // set the learn rate and the number of iterations 
	            ic.maxIter = maxEpochs; 
	            // set te number of patterns 
	            ic.K = (int)(K*X.getDimColumns()); 
	            // set the size of segments 
	            ic.minWindowSize = (int)(L*X.getDimColumns());

	            ic.maxScale = scale; 
	            
	            ic.deltaT = (int)(deltaT*ic.minWindowSize);
	            
	            ic.lambdaP = lambdaP; 
	            ic.svmC = svmC;
	            
	            // learn the local convolutions
	            
	            double [] errorRates = ic.LearnFoldIn();
	            
				System.out.println( String.valueOf(errorRates[0]) + " " + String.valueOf(errorRates[1]) + " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model
						+ " lambdaF=" + lambdaP
						+ " svmC=" + svmC
						+ " K=" + K	
						+ " L=" + L
						+ " scale=" + scale  
						+ " deltaT="+deltaT 
						+ " time="	+ (System.currentTimeMillis() - startTime)  
						+ " epochs=" + maxEpochs);   
						
			}
			else if (model.compareTo("dp") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
	            Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            DiscriminativePatterns dp = new DiscriminativePatterns();
	            // initialize the sizes of data structures
	            dp.NTrain = finalTrainSet.GetNumInstances(); 
	            dp.NTest = testSet.GetNumInstances();
	            dp.M = X.getDimColumns();
	            // set the time series and labels
	            dp.T = X;
	            dp.Y = Y;
	            // set the learn rate and the number of iterations
	            dp.maxIter = maxEpochs;
	            // set te number of patterns
	            dp.K = (int)(K*X.getDimColumns());
	            // set the size of segments
	            dp.L = (int)(L*X.getDimColumns());
	            // set the regularization parameter
	            dp.lambdaP = lambdaP;
	            dp.lambdaD = lambdaD;
	            dp.lambdaW = lambdaW;
	            dp.beta = beta;
	            dp.deltaT = (int)(deltaT*dp.L);
	            dp.eta = eta;
	            
	            // learn the local convolutions
	            errorRate = dp.Learn();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "								
						+ "ds=" + dsName + " "
						+ "fold=" + fold	+ " "
						+ "mode=" + runMode + " "
						+ "model=" + model	+ " "
						+ "K=" + K	+ " "
						+ "L=" + L	+ " "
						+ lambdaD + " "
						+ lambdaP + " "
						+ lambdaW + " "
						+ eta + " "
						+ "epochs="+ maxEpochs
						);
				
			}
			else if (model.compareTo("clp") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
	            Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            ConvolutionLocalPolynomials bcp = new ConvolutionLocalPolynomials();
	            // initialize the sizes of data structures
	            bcp.NTrain = finalTrainSet.GetNumInstances();
	            bcp.NTest = testSet.GetNumInstances();
	            bcp.M = X.getDimColumns();
	            // set the time series and labels
	            bcp.T = X;
	            bcp.Y = Y;
	            // set the learn rate and the number of iterations
	            bcp.maxIter = maxEpochs;
	            // set te number of patterns
	            bcp.K = (int)(K*X.getDimColumns());
	            // set the size of segments
	            bcp.L = (int)(L*X.getDimColumns());
	            // set the regularization parameter
	            bcp.lambdaBeta = lambdaBeta;
	            bcp.lambdaD = lambdaD;
	            bcp.lambdaP = lambdaP;
	            bcp.degree = degree;
	            bcp.deltaT = (int)(deltaT*bcp.L);
	            
	            bcp.dataset = dsName;
	            bcp.fold = fold;
	            
	            // learn the local convolutions
	            errorRate = bcp.Learn();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "
						+ dsName + " "
						+ fold	+ " "
						+ runMode + " "
						+ model	+ " "
						+ lambdaD + " "
						+ lambdaP + " "
						+ K	+ " "
						+ L	+ " "
						+ eta + " "
						+ maxEpochs);
						
			}
			else if (model.compareTo("hlc") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
	            Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            HardLocalConvolutions lc = new HardLocalConvolutions();
	            // initialize the sizes of data structures
	            lc.NTrain = finalTrainSet.GetNumInstances();
	            lc.NTest = testSet.GetNumInstances();
	            lc.Q = X.getDimColumns();
	            // set the time series and labels
	            lc.T = X;
	            lc.Y = Y;
	            // set te number of patterns
	            lc.K = (int)(K*X.getDimColumns());
	            // set the size of segments
	            lc.L = (int)(L*X.getDimColumns());
	            // set the regularization parameter

	            // learn the local convolutions
	            errorRate = lc.Learn();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "
						+ "ds=" + dsName + " "
						+ "fold=" + fold	+ " "
						+ "mode=" + runMode + " "
						+ "model=" + model	+ " "
						+ "K=" + K	+ " "
						+ "L=" + L	+ " "
						+ "epochs="+ maxEpochs);						
			}
			else if (model.compareTo("dps") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
				Matrix X = new Matrix();
	            X.LoadDatasetFeatures(finalTrainSet, false);
	            X.LoadDatasetFeatures(testSet, true);
	            Matrix Y = new Matrix();
	            Y.LoadDatasetLabels(finalTrainSet, false);
	            Y.LoadDatasetLabels(testSet, true);
	            
	            DiscriminativePatternsScales dp = new DiscriminativePatternsScales();
	            // initialize the sizes of data structures
	            dp.ITrain = finalTrainSet.GetNumInstances(); 
	            dp.ITest = testSet.GetNumInstances();
	            dp.Q = X.getDimColumns();
	            // set the time series and labels
	            dp.T = X;
	            dp.Y = Y;
	            // set the learn rate and the number of iterations
	            dp.maxIter = maxEpochs;
	            // set te number of patterns
	            dp.K = (int)(K*X.getDimColumns());
	            // set the size of segments
	            dp.initL = (int)(L*X.getDimColumns());
	            // set the regularization parameter
	            dp.lambdaP = lambdaP;
	            dp.lambdaD = lambdaD;
	            dp.lambdaW = lambdaW;
	            dp.beta = beta;
	            dp.deltaT = (int) Math.ceil(deltaT*dp.initL); 
	            dp.eta = eta; 
	            dp.M = M; 
	            
	            // learn the local convolutions
	            errorRate = dp.Learn();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "								
						+ "ds=" + dsName + " "
						+ "fold=" + fold	+ " "
						+ "mode=" + runMode + " "
						+ "model=" + model	+ " "
						+ "K=" + K	+ " "
						+ "L=" + L	+ " "
						+ "M_i=" + M + " "
						+ "lD=" + lambdaD + " "
						+ "lP=" + lambdaP + " "
						+ "lW=" + lambdaW + " "
						+ "eta=" + eta + " "
						+ "epochs="+ maxEpochs
						);
				
			}
			else if (model.compareTo("dpmc") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
				// predictor variables T
				Matrix T = new Matrix();
	            T.LoadDatasetFeatures(finalTrainSet, false);
	            T.LoadDatasetFeatures(testSet, true);
	            // outcome variable O
	            Matrix O = new Matrix();
	            O.LoadDatasetLabels(finalTrainSet, false);
	            O.LoadDatasetLabels(testSet, true);
	            
	            DiscriminativePatternsMultiClass dpmc = new DiscriminativePatternsMultiClass();
	            // initialize the sizes of data structures
	            dpmc.ITrain = finalTrainSet.GetNumInstances(); 
	            dpmc.ITest = testSet.GetNumInstances();
	            dpmc.Q = T.getDimColumns();
	            // set the time series and labels
	            dpmc.T = T;
	            dpmc.O = O;
	            // set the learn rate and the number of iterations
	            dpmc.maxIter = maxEpochs;
	            // set te number of patterns
	            dpmc.K = (int)(K*T.getDimColumns());
	            // set the size of segments
	            dpmc.initL = (int)(L*T.getDimColumns());
	            // set the regularization parameter
	            dpmc.lambdaP = lambdaP;
	            dpmc.lambdaD = lambdaD;
	            dpmc.lambdaW = lambdaW;
	            dpmc.beta = beta;
	            dpmc.deltaT = (int) Math.ceil(deltaT*dpmc.initL); 
	            dpmc.eta = eta; 
	            dpmc.M = M; 
	            dpmc.R = R;
	            

	            finalTrainSet.ReadNominalTargets();
	            dpmc.nominalLabels =  new ArrayList<Double>(finalTrainSet.nominalLabels);
	            
	            // learn the local convolutions
	            errorRate = dpmc.Learn();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "								
						+ "ds=" + dsName + " "
						+ "fold=" + fold	+ " "
						+ "mode=" + runMode + " "
						+ "model=" + model	+ " "
						+ "K=" + K	+ " "
						+ "L=" + L	+ " "
						+ "M=" + M + " "
						+ "lD=" + lambdaD + " "
						+ "lP=" + lambdaP + " "
						+ "lW=" + lambdaW + " "
						+ "eta=" + eta + " "
						+ "epochs="+ maxEpochs
						);
				
			}
			else if (model.compareTo("ls") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

					            
				// predictor variables T
				Matrix T = new Matrix();
	            T.LoadDatasetFeatures(finalTrainSet, false);
	            T.LoadDatasetFeatures(testSet, true);
	            // outcome variable O
	            Matrix O = new Matrix();
	            O.LoadDatasetLabels(finalTrainSet, false);
	            O.LoadDatasetLabels(testSet, true);
	            
	            LearnShapelets ls = new LearnShapelets();
	            // initialize the sizes of data structures
	            ls.ITrain = finalTrainSet.GetNumInstances(); 
	            ls.ITest = testSet.GetNumInstances();
	            ls.Q = T.getDimColumns();
	            // set the time series and labels 
	            ls.T = T;
	            ls.Y = O;
	            // set the learn rate and the number of iterations
	            ls.maxIter = maxEpochs;
	            // set te number of patterns
	            ls.K = (int)(K*T.getDimColumns());
	            // set the size of segments
	            ls.L = (int)(L*T.getDimColumns());
	            // set the regularization parameter

	            ls.lambdaS = lambdaS;
	            ls.lambdaW = lambdaW;
	            ls.eta = eta; 
	            ls.alpha = alpha;
	            
	            // learn the local convolutions
	            errorRate = ls.Learn(); 
	            
	            long endTime = System.currentTimeMillis();
	            
				System.out.println( String.valueOf(errorRate)
						+ " "								
						+ "ds=" + dsName + " "
						+ "fold=" + fold	+ " "
						+ "mode=" + runMode + " "
						+ "model=" + model	+ " "
						+ "K=" + K	+ " "
						+ "L=" + L	+ " "
						+ "lS=" + lambdaS + " "
						+ "lW=" + lambdaW + " "
						+ "alpha=" + alpha + " "
						+ "eta=" + eta + " "
						+ "epochs="+ maxEpochs + " "
						+ "time="+ (endTime-startTime)
						); 
				
			}
			else if (model.compareTo("lsg") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

				// predictor variables T
				Matrix T = new Matrix();
	            T.LoadDatasetFeatures(finalTrainSet, false);
	            T.LoadDatasetFeatures(testSet, true);
	            // outcome variable O
	            Matrix O = new Matrix();
	            O.LoadDatasetLabels(finalTrainSet, false);
	            O.LoadDatasetLabels(testSet, true);

	            LearnShapeletsGeneralized lsg = new LearnShapeletsGeneralized();   
	            // initialize the sizes of data structures
	            lsg.ITrain = finalTrainSet.GetNumInstances();  
	            lsg.ITest = testSet.GetNumInstances();
	            lsg.Q = T.getDimColumns();
	            // set the time series and labels
	            lsg.T = T;
	            lsg.Y = O;
	            // set the learn rate and the number of iterations
	            lsg.maxIter = maxEpochs;
	            // set te number of patterns 
	            lsg.K = (int)(K*T.getDimColumns());
	            lsg.L_min = (int)(L*T.getDimColumns());
	            lsg.R = R;
	            // set the regularization parameter
	            lsg.lambdaW = lambdaW;  
	            lsg.eta = eta;  
	            lsg.alpha = alpha; 
	            finalTrainSet.ReadNominalTargets();
	            lsg.nominalLabels =  new ArrayList<Double>(finalTrainSet.nominalLabels);
	            
	            
	            // learn the local convolutions
	            errorRate = lsg.Learn(); 
	            double trainErrorRate = lsg.GetMCRTrainSet();
	            
	            long endTime = System.currentTimeMillis(); 
	            
				System.out.println( 
						String.valueOf(errorRate)  + " " + String.valueOf(trainErrorRate) + " "
						+ "ds=" + dsName + " " 
						+ "fold=" + fold	+ " " 
						+ "mode=" + runMode + " " 
						+ "model=" + model	+ " " 
						+ "K=" + K	+ " " 
						+ "L=" + L	+ " " 
						+ "R=" + R	+ " " 
						+ "lW=" + lambdaW + " "
						+ "lS=" + lambdaS + " " 
						+ "alpha=" + alpha + " " 
						+ "eta=" + eta + " " 
						+ "epochs="+ maxEpochs + " " 
						+ "time="+ (endTime-startTime) 
						); 
				
			}
			else if (model.compareTo("ldm") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

				// predictor variables T
				Matrix T = new Matrix();
	            T.LoadDatasetFeatures(finalTrainSet, false);
	            T.LoadDatasetFeatures(testSet, true);
	            // outcome variable O
	            Matrix O = new Matrix();
	            O.LoadDatasetLabels(finalTrainSet, false);
	            O.LoadDatasetLabels(testSet, true);

	            LearnDiscriminativeMotifs ldm = new LearnDiscriminativeMotifs(); 
	            // initialize the sizes of data structures
	            ldm.ITrain = finalTrainSet.GetNumInstances(); 
	            ldm.ITest = testSet.GetNumInstances();
	            ldm.Q = T.getDimColumns();
	            // set the time series and labels
	            ldm.T = T;
	            ldm.Y = O;
	            // set the learn rate and the number of iterations
	            ldm.maxIter = maxEpochs;
	            // set te number of patterns
	            ldm.K = (int)(K);
	            // set the size of segments
	            ldm.L_min = (int)(L); 
	            ldm.R = R; 
	            // set the regularization parameter
	            ldm.lambdaW = lambdaW;
	            ldm.eta = eta; 
	            ldm.gamma = gamma; 
	            ldm.kMeansIter = 50;

	            finalTrainSet.ReadNominalTargets();
	            ldm.nominalLabels =  new ArrayList<Double>(finalTrainSet.nominalLabels);
	            
	            // learn the local convolutions
	            errorRate = ldm.Learn(); 
	            double errorSVM = 0; //ldm.ClassifySVM();
	            
	            long endTime = System.currentTimeMillis(); 
	            
				System.out.println( String.valueOf(errorRate)  + " "
						+ "ds=" + dsName + " " 
						+ "fold=" + fold	+ " " 
						+ "mode=" + runMode + " " 
						+ "model=" + model	+ " " 
						+ "K=" + K	+ " " 
						+ "L=" + L	+ " " 
						+ "R=" + R	+ " " 
						+ "lW=" + lambdaW + " "
						+ "gamma=" + gamma + " " 
						+ "eta=" + eta + " " 
						+ "epochs="+ maxEpochs + " " 
						+ "time="+ (endTime-startTime) 
						); 
				
			}
			else if (model.compareTo("ldminv") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

				// predictor variables T
				Matrix T = new Matrix();
	            T.LoadDatasetFeatures(finalTrainSet, false);
	            T.LoadDatasetFeatures(testSet, true);
	            // outcome variable O
	            Matrix O = new Matrix();
	            O.LoadDatasetLabels(finalTrainSet, false);
	            O.LoadDatasetLabels(testSet, true);

	            LearnDiscriminativeMotifsInverseEuclidean ldm = new LearnDiscriminativeMotifsInverseEuclidean(); 
	            // initialize the sizes of data structures
	            ldm.ITrain = finalTrainSet.GetNumInstances(); 
	            ldm.ITest = testSet.GetNumInstances();
	            ldm.Q = T.getDimColumns();
	            // set the time series and labels
	            ldm.T = T;
	            ldm.Y = O;
	            // set the learn rate and the number of iterations
	            ldm.maxIter = maxEpochs;
	            // set te number of patterns
	            ldm.K = (int)(K);
	            // set the size of segments
	            ldm.L_min = (int)(L); 
	            ldm.R = R; 
	            // set the regularization parameter
	            ldm.lambdaW = lambdaW;
	            ldm.eta = eta; 
	            ldm.gamma = gamma; 
	            ldm.kMeansIter = 1;

	            finalTrainSet.ReadNominalTargets();
	            ldm.nominalLabels =  new ArrayList<Double>(finalTrainSet.nominalLabels);
	            
	            // learn the local convolutions 
	            errorRate = ldm.Learn(); 
	            
	            long endTime = System.currentTimeMillis(); 
	            
				System.out.println( String.valueOf(errorRate)  + " " 
						+ "ds=" + dsName + " " 
						+ "fold=" + fold	+ " " 
						+ "mode=" + runMode + " " 
						+ "model=" + model	+ " " 
						+ "K=" + K	+ " " 
						+ "L=" + L	+ " " 
						+ "R=" + R	+ " " 
						+ "lW=" + lambdaW + " " 
						+ "gamma=" + gamma + " " 
						+ "eta=" + eta + " " 
						+ "epochs="+ maxEpochs + " " 
						+ "time="+ (endTime-startTime) 
						); 
				
			}
			else if (model.compareTo("lsgSingle") == 0) {
				
				double errorRate = 0;
				
				long startTime = System.currentTimeMillis();
				
				DataSet finalTrainSet = null;
				
				if (runMode.compareTo("validation") == 0) 
				{
					finalTrainSet = trainSet;

				} else if (runMode.compareTo("test") == 0) 
				{
					// merge the train and test
					finalTrainSet = new DataSet(trainSet);
					finalTrainSet.AppendDataSet(validationSet);
				}

				// predictor variables T
				Matrix T = new Matrix();
	            T.LoadDatasetFeatures(finalTrainSet, false);
	            T.LoadDatasetFeatures(testSet, true);
	            // outcome variable O
	            Matrix O = new Matrix();
	            O.LoadDatasetLabels(finalTrainSet, false);
	            O.LoadDatasetLabels(testSet, true);

	            LearnShapeletsGeneralized lsg = new LearnShapeletsGeneralized(); 
	            // initialize the sizes of data structures
	            lsg.ITrain = finalTrainSet.GetNumInstances(); 
	            lsg.ITest = testSet.GetNumInstances();
	            lsg.Q = T.getDimColumns();
	            // set the time series and labels
	            lsg.T = T;
	            lsg.Y = O;
	            // set the learn rate and the number of iterations
	            lsg.maxIter = maxEpochs;
	            // set te number of patterns
	            lsg.K = 2; 
	            // set the size of segments
	            lsg.L_min = (int)(L*T.getDimColumns()); 
	            lsg.R = 1; 
	            // set the regularization parameter
	            lsg.lambdaW = lambdaW; 
	            lsg.eta = eta; 
	            lsg.alpha = alpha; 

	            finalTrainSet.ReadNominalTargets();
	            lsg.nominalLabels =  new ArrayList<Double>(finalTrainSet.nominalLabels);
	            
	            // learn the local convolutions
	            errorRate = lsg.Learn(); 
	            
	            //errorRate = lsg.LearnSearchInitialShapelets();
	            
	            double errorSVM = 0;
	            
	            long endTime = System.currentTimeMillis(); 
	            
				System.out.println( String.valueOf(errorRate)  + " "
						+ "ds=" + dsName + " " 
						+ "fold=" + fold	+ " " 
						+ "mode=" + runMode + " " 
						+ "model=" + model	+ " " 
						+ "L=" + L	+ " " 
						+ "lW=" + lambdaW + " "
						+ "alpha=" + alpha + " " 
						+ "eta=" + eta + " " 
						+ "epochs="+ maxEpochs + " " 
						+ "time="+ (endTime-startTime) 
						); 
				
			}
            
            

		}

	}

}
