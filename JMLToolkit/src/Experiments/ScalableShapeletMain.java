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

public class ScalableShapeletMain {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		long programStartTime = System.currentTimeMillis();
		
		if (args.length == 0) {
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG; 
			 //Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG; 

			String dir = "/mnt/vartheta/Data/classification/timeseries/",
					ds = "MALLAT",
					foldNo = "default";   
			
			String sp = File.separator;
			
			args = new String[] 
					{ 
					"fold=" + foldNo,
					"model=sd",  
					"runMode=test", 
					//"trainSet=/mnt/vartheta/Data/classification/longts/dailySports/dailySport.csv",
					 "trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp
							+ ds + "_TRAIN",  
					 "testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp 
							+ ds + "_TEST", 
					"paaRatio=0.125",
					"percentile=35",
					"numCVFolds=2",
					"maxNumCandidates=2",
					};
		}

		if (args.length > 0) 
		{
			// set model values
			String model = "", runMode = "";
			String trainSetPath = "", 
					testSetPath = "", 
					fold = "";
			
			double paaRatio = 1.0;
			int percentile = 25;
			int numCVFolds = 10; 
			int maxNumCandidates = 100;
			
			// read and parse parameters
			for (String arg : args) {
				String[] argTokens = arg.split("=");
				
				if (argTokens[0].compareTo("model") == 0)
					model = argTokens[1]; 
				else if (argTokens[0].compareTo("trainSet") == 0)
					trainSetPath = argTokens[1];
				else if (argTokens[0].compareTo("testSet") == 0)
					testSetPath = argTokens[1];
				else if (argTokens[0].compareTo("paaRatio") == 0)
					paaRatio =  Double.parseDouble(argTokens[1]);
				else if (argTokens[0].compareTo("percentile") == 0)
					percentile =  Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("numCVFolds") == 0)
					numCVFolds =  Integer.parseInt(argTokens[1]);
				else if (argTokens[0].compareTo("maxNumCandidates") == 0)
					maxNumCandidates =  Integer.parseInt(argTokens[1]);
			}
			
			String dsName = new File(trainSetPath).getName().split("_TRAIN")[0];

			if (model.compareTo("sd") == 0) {
				
				/*
				for(int ratioInverse = 1; ratioInverse <= 10; ratioInverse++ )
            	{
				for( percentile = 5; percentile <= 50; percentile += 5 )
            	{
				paaRatio = 1.0/(double)ratioInverse;
	            */
				
				
				int numTrials = 1;
	            double [] accuracies = new double[numTrials];
	            double [] numAccepted = new double[numTrials];
	            double [] numRejected = new double[numTrials];
	            double [] numRefused = new double[numTrials];
	            
	            double [] trainTimes = new double[numTrials];
	            double [] testTimes = new double[numTrials];
	            
	            for(int trial = 0; trial < numTrials; trial++)
	            {
	            	
	            	ScalableShapeletDiscovery esd = new ScalableShapeletDiscovery();
		            esd.trainSetPath = trainSetPath;
		            esd.testSetPath = testSetPath;
		            
		            esd.percentile = percentile;		            
		            esd.paaRatio = paaRatio;
		            
		            esd.Search();
		            
		            double errorRate = esd.ComputeTestError();  
	
					accuracies[trial] = 1-errorRate;
					numAccepted[trial] = esd.numAcceptedShapelets;
					numRejected[trial] = esd.numRejectedShapelets;
					numRefused[trial] = esd.numRefusedShapelets;
					trainTimes[trial] = esd.trainTime/1000.0;
					testTimes[trial] = esd.testTime/1000.0; 
										
					/*
					System.out.println( 
							accuracies[trial] + "," + runTimes[trial] + " "
									+ dsName + " "
									+ fold	+ " "
									+ runMode + " "
									+ model	+ " "
									+ fdp.shapeletLength	+ " "
									+ fdp.epsilon);
									*/
	            } 

	            System.out.println( dsName + "," + paaRatio + "," + percentile +   
	            					"," + StatisticalUtilities.Mean(accuracies) + 
	            					"," + StatisticalUtilities.StandardDeviation(accuracies) +
	            					
	            					"," + StatisticalUtilities.Mean(trainTimes) + 
	            					"," + StatisticalUtilities.StandardDeviation(trainTimes) +
	            					
	            					"," + StatisticalUtilities.Mean(numAccepted) +
	            					"," + StatisticalUtilities.Mean(numRejected) +
	            					"," + StatisticalUtilities.Mean(numRefused) +
	            					
	            					"," + StatisticalUtilities.Mean(testTimes) + 
	            					"," + StatisticalUtilities.StandardDeviation(testTimes));
            	//}
            	//}
			}
			else if (model.compareTo("sdMulti") == 0) 
			{
//				ScalableShapeletDiscoveryMultichannel ssdm = new ScalableShapeletDiscoveryMultichannel();
//				ssdm.percentile = percentile;
//				
//				ssdm.trainSeries = new TimeSeriesMultichannel(trainSetPath);
//				ssdm.trainSeries.ApplyPiecewiseAggregateApproximation(paaRatio); 
//				
//				
//				double [] accuracies = new double[numCVFolds];
//	            double [] trainTimes = new double[numCVFolds];
//	            double [] testTimes = new double[numCVFolds];
//	            
//	            ssdm.numFolds = numCVFolds;
//	            ssdm.maxNumCandidates = maxNumCandidates; 
//	          
//	            
//				for(int foldNo = 0; foldNo < numCVFolds; foldNo++)
//				{
//					accuracies[foldNo] = 1.0 - ssdm.Search();
//					trainTimes[foldNo] = ssdm.trainTime;
//					testTimes[foldNo] = ssdm.testTime;
//					
//					System.out.println("Fold="+foldNo+ 
//										", paaRatio=" + paaRatio + 
//										", percentile=" + percentile +
//										", numMaxCandidates=" + maxNumCandidates +
//										", Accuracy=" + accuracies[foldNo] + ", TrainTime=" + trainTimes[foldNo]);
//				}
//				
//			    System.out.println( dsName + "," + paaRatio + "," + percentile + "," + maxNumCandidates +   
//    					"," + StatisticalUtilities.Mean(accuracies) + 
//    					"," + StatisticalUtilities.StandardDeviation(accuracies) +
//    					
//    					"," + StatisticalUtilities.Mean(trainTimes) + 
//    					"," + StatisticalUtilities.StandardDeviation(trainTimes) +
//    					
//    					"," + StatisticalUtilities.Mean(testTimes) + 
//    					"," + StatisticalUtilities.StandardDeviation(testTimes));
//	
			}


		}
	}
}
