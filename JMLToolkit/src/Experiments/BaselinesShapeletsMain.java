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
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class BaselinesShapeletsMain {
	public static void main(String[] args) {
		Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;

		long programStartTime = System.currentTimeMillis();
		
		if (args.length == 0) {
			Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG; 
			 //Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG; 

			String dir = "/mnt/vartheta/Data/classification/timeseries/",
					ds = "MedicalImages",
					foldNo = "default";   
			
			/*
			String dir = "/mnt/vartheta/Data/classification/longts/",
					ds = "BIDMC",
					foldNo = "1";   
			*/
			
			String sp = File.separator;
			
			args = new String[] 
					{ 
					"fold=" + foldNo,
					"model=fast",  
					"runMode=test", 
					"trainSet=" + dir + ds + sp + "folds" + sp + foldNo + sp
							+ ds + "_TRAIN",  
					"testSet=" + dir + ds + sp + "folds" + sp + foldNo + sp 
							+ ds + "_TEST", 
					"trainExecutable=/home/jgrabocka/Software/trainFastShapelet",
					"testExecutable=/home/jgrabocka/Software/testFastShapelet"	
					};
		} 

		if (args.length > 0) 
		{
			// set model values
			String model = "", runMode = "";
			String trainSetPath = "", 
					testSetPath = "", 
					fold = "",
					trainExecutablePath="",
					testExecutablePath="";
			
			// read and parse parameters
			for (String arg : args) {
				String[] argTokens = arg.split("=");
				
				if (argTokens[0].compareTo("model") == 0)
					model = argTokens[1]; 
				else if (argTokens[0].compareTo("trainSet") == 0)
					trainSetPath = argTokens[1];
				else if (argTokens[0].compareTo("testSet") == 0)
					testSetPath = argTokens[1];
				else if (argTokens[0].compareTo("trainExecutable") == 0)
					trainExecutablePath = argTokens[1];
				else if (argTokens[0].compareTo("testExecutable") == 0)
					testExecutablePath = argTokens[1];
			}
			
			String dsName = new File(trainSetPath).getName().split("_TRAIN")[0];

			if (model.compareTo("fast") == 0) {
				
				int numTrials = 10;
	            double [] accuracies = new double[numTrials];
	            double [] trainTimes = new double[numTrials];
	            
	            DataSet trainSet = new DataSet();
	    		trainSet.LoadDataSetFile(new File(trainSetPath)); 
	    		trainSet.ReadNominalTargets();
	    		
	    		DataSet testSet = new DataSet();
	    		testSet.LoadDataSetFile(new File(testSetPath));
	    		testSet.ReadNominalTargets(); 
	    		
	    		
	    		
	    		int minLength = (int) (0.2*trainSet.numFeatures);
	    		int maxLength = 3*minLength;
	    		int stepLength = minLength; 
	    		
	    		/*
	    		// default paper parameters
	    		int maxLength = trainSet.numFeatures; 
	    		int minLength = 1;
	    		int stepLength = 1;
	    		*/
	    		
	            
	            for(int trial = 0; trial < numTrials; trial++)
	            {
	            	

	            	String trainCommand = 
	            			trainExecutablePath + " " + 
							trainSetPath + " " +
	            			trainSet.nominalLabels.size() + " " +
							trainSet.instances.size() + " " + 
							maxLength + " " + minLength + " " + stepLength + " " +
							"10 10 " + 
							"tree_"+dsName + " " +
							"time_"+dsName;
	            	
	            	String testCommand =  
	            			testExecutablePath + " " + 
							testSetPath + " " +
	            			testSet.nominalLabels.size() + " " +
							testSet.instances.size() + " " + 
							"tree_"+dsName + " " +
							"time_"+dsName;	;
	            	
	            	//System.out.println("Train command: " + trainCommand);
	            	//System.out.println("Test command: " + testCommand);
	            	
	            	try{
	            		Process p = Runtime.getRuntime().exec( trainCommand	);
	            		p.waitFor();
	            		
	            		BufferedReader br = new BufferedReader( new InputStreamReader(p.getInputStream() ) );
	            		
	            		String trainOutputLine = br.readLine();
	            		
	            		//System.out.println(trainOutputLine);
	            		
	            		trainTimes[trial] = Double.parseDouble(trainOutputLine);
	            		
	            		p = Runtime.getRuntime().exec( testCommand	);
	            		p.waitFor();
	            		
	            		br = new BufferedReader( new InputStreamReader(p.getInputStream() ) );
	            		
	            		String testOutputLine = br.readLine();
	            		//System.out.println(testOutputLine);
	            		
	            		accuracies[trial] = Double.parseDouble(testOutputLine)/100.0;
	            		
	            		//System.out.println(dsName + "," + trial + "," + trainTimes[trial] + "," + accuracies[trial] );
	            		
	            		
	            	}
	            	catch(Exception exc)
	            	{
	            		exc.printStackTrace();
	            	}
	            	
	
					/*
					accuracies[trial] = 1-errorRate;
					trainTimes[trial] = esd.trainTime/1000.0;
					testTimes[trial] = esd.testTime/1000.0; 
						*/				
					
					
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

	            System.out.println( dsName +  ",Aggregated" +  
	            					"," + StatisticalUtilities.Mean(accuracies) + 
	            					"," + StatisticalUtilities.StandardDeviation(accuracies) +
	            					
	            					"," + StatisticalUtilities.Mean(trainTimes) + 
	            					"," + StatisticalUtilities.StandardDeviation(trainTimes) );
            	
			}


		}
	}
}
