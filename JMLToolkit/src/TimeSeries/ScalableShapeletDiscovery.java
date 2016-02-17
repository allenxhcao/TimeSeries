package TimeSeries;


import java.io.File;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import Regression.TimeSeriesPolynomialApproximation;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.StatisticalUtilities;
import Utilities.Logging.LogLevel;
import Classification.WekaClassifierInterface;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;

public class ScalableShapeletDiscovery 
{	
	// series data and labels
	public Matrix trainSeriesData, trainSeriesLabels; 
	
	public Matrix testSeriesData, testSeriesLabels;
	
	public int numTrainInstances;
	int seriesLength;
	
	// the length of the shapelet we are searching for
	int [] shapeletLengths;
	
	// the epsilon parameter to prune candidates being epsilon close to a 
	// rejected or accepted shapelet
	public double epsilon;
	
	// the percentile for the distribution of distances between pairs of segments
	// is used as epsilon
	public int percentile;
	
	// list of accepted and rejected words
	List<double []> acceptedList = new ArrayList<double []>();
	List<double []> rejectedList = new ArrayList<double []>();
	
	
	public String trainSetPath, testSetPath;
	// the paa ratio, i.e. 0.25 reduces the length of series by 1/4
	public double paaRatio;
	
	
	// the histogram contains lists of frequency columns
	List< double [] > distancesShapelets = new ArrayList<double []>();
	// the current classification error of the frequencies histogram
	double currentTrainError = Double.MAX_VALUE;
	
	double [][] seriesDistancesMatrix;
	
	// logs on the number of acceptances and rejections
	public int numAcceptedShapelets, numRejectedShapelets, numRefusedShapelets;
	
	public long trainTime, testTime;
	
	
	public boolean normalizeData;
	
	// random number generator
	Random rand = new Random();
	
	public ScalableShapeletDiscovery()
	{
		normalizeData = false;
	}
	
	public void Search()
    {		
		// load train data
		LoadTrainData();
		
		numAcceptedShapelets = numRejectedShapelets = numRefusedShapelets = 0;
		// num train instances
		numTrainInstances = trainSeriesData.getDimRows();
    	// set the length of series
    	seriesLength = trainSeriesData.getDimColumns();
    	    	
    	// check 20%,40%, 60% shapelet lengths
    	
    	shapeletLengths = new int[3]; 
    	shapeletLengths[0] = (int)(0.20*seriesLength);
    	shapeletLengths[1] = (int)(0.40*seriesLength); 
    	shapeletLengths[2] = (int)(0.60*seriesLength);
    	
    	epsilon = EstimateEpsilon();
    	
    	//System.out.println(epsilon); 
    	
    	// set distances matrix to 0.0
    	seriesDistancesMatrix = new double[numTrainInstances][numTrainInstances];
    	for(int i = 0;  i < numTrainInstances; i++)
			for(int j = i+1;  j < numTrainInstances; j++)
				seriesDistancesMatrix[i][j] = 0.0;
    	
    	// evaluate all the words of all series
    	int numTotalCandidates = numTrainInstances*seriesLength*shapeletLengths.length;
    	
    	//Logging.println("Candidate shapelets:");
    	
    	for( int candidateIdx = 0; candidateIdx < numTotalCandidates; candidateIdx++)
    	{
    		// select a random series
    		int i = rand.nextInt(numTrainInstances);
    		// select a random shapelet length
        	int shapeletLength = shapeletLengths[ rand.nextInt(shapeletLengths.length) ];
        	// select a random segment of the i-th series
        	int j = rand.nextInt(seriesLength-shapeletLength+1); 
    			
			double [] candidateShapelet = new double[shapeletLength];
			for(int k = 0; k < shapeletLength; k++)
				candidateShapelet[k] = trainSeriesData.get(i, j + k);
    			
			EvaluateShapelet(candidateShapelet);
			
			//Logging.println(candidateShapelet); 
			
			//Logging.println(candidateIdx + "," + currentTrainError + "," + numAcceptedShapelets + "," + numRejectedShapelets + "," + numRefusedShapelets);
			
		}
	    
    	

    	/*
    	Logging.println("Considered Candidates:");
    	for(double [] acceptedShapelet : acceptedList)
    		Logging.println(acceptedShapelet);
    	for(double [] rejectedShapelet : rejectedList) 
    		Logging.println(rejectedShapelet);
    	
    	
    	
    	Logging.println("Train Shapelet-Transformed Data:");
    	for(int i = 0; i < numTrainInstances; i++)
    	{
    		System.out.print( trainSeriesLabels.get(i) + " "); 
    		
    		for(int shpltIdx = 0; shpltIdx < acceptedList.size(); shpltIdx++)
	    	{
	    		System.out.print( distancesShapelets.get(shpltIdx)[i] + " " );
	    	}
    		System.out.println("");
    	}
    	*/
    	
    	
    	
    }
    
    // consider a word whether it is part of the accept list of reject list
    // and if not whether it helps reduce the classification error
    public void EvaluateShapelet(double [] candidate)
    {
    	
    	// if the lists are both empty or the candidate is previously not been considered 
    	// then give it a chance
    	if(  !FoundInList(candidate, acceptedList)  &&  !FoundInList(candidate, rejectedList) )  
    	{
    		// compute the soft frequencies of the word in the series data
    		double [] distancesCandidate = ComputeDistances(candidate);
    		
    		// refresh distances
    		AddCandidateDistancesToDistancesMatrix(distancesCandidate);
    		// compute error
    		double newTrainError = ComputeTrainError(); 
    		
    		if( newTrainError < currentTrainError)
    		{
    			// accept the word, which improves the error
    			acceptedList.add(candidate);
    			
    			// add the distances of the shapelet to a list
    			// will be used for testing
    			distancesShapelets.add(distancesCandidate);
    			
    			// set the new error as the current one
    			currentTrainError = newTrainError;
    			
    			// increase the counter of accepted words
    			numAcceptedShapelets++;
    		}
    		else
    		{
    			// the word doesn't improve the error, therefore is rejected
    			rejectedList.add(candidate);
    			
    			// finally remove the distances from the distance matrix
    			RemoveCandidateDistancesToDistancesMatrix(distancesCandidate); 
    			
    			// increase the counter of rejected words
    			numRejectedShapelets++;
    		}
    		
    	}
    	else // word already was accepted and/or rejected before
    	{
    		numRefusedShapelets++;
    	}
    	
				
    }

    // compute the minimum distance of a candidate to the training series
	private double [] ComputeDistances(double [] candidate)  
	{
		double [] distancesCandidate = new double[trainSeriesData.getDimRows()];
		
		double diff = 0, distanceToSegment = 0, minDistanceSoFar = Double.MAX_VALUE;
		
		int shapeletLength = candidate.length;
		
		for( int i = 0; i < trainSeriesData.getDimRows(); i++ ) 
		{ 
			minDistanceSoFar = Double.MAX_VALUE; 
			
			for( int j = 0; j < seriesLength-shapeletLength+1; j++ ) 
    		{
				distanceToSegment = 0; 
				
				for(int k = 0; k < shapeletLength; k++) 
				{
					diff = candidate[k]- trainSeriesData.get(i, j + k); 
					distanceToSegment += diff*diff; 
					
					// if the distance of the candidate to this segment is more than the best so far
					// at point k, skip the remaining points
					if( distanceToSegment > minDistanceSoFar ) 
						break; 
				} 
				
				if( distanceToSegment < minDistanceSoFar) 
					minDistanceSoFar = distanceToSegment; 
    		} 
			
			distancesCandidate[i] = minDistanceSoFar; 
		}
		
		return distancesCandidate;		
	}
    
	// compute the error of the training instances
	// from the distances matrix
	public double ComputeTrainError()
	{		
		int numMissClassifications = 0; 
        double realLabel = -1;         
        double nearestLabel = -1; 
        double nearestDistance = Double.MAX_VALUE; 
		
        // for every test instance 
        for(int i = 0;  i < numTrainInstances; i++)
        {
            realLabel = trainSeriesLabels.get(i); 
            
            nearestLabel = -1; 
            nearestDistance = Double.MAX_VALUE; 

            // iterate through training instances and find the closest neighbours
            for(int j = 0; j < numTrainInstances; j++)
            {   
            	// avoid itself as a neighbor
            	if( i == j) continue;
               
            	double distance = ( i < j ? seriesDistancesMatrix[i][j] : seriesDistancesMatrix[j][i] );
            	
                if( distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestLabel = trainSeriesLabels.get(j); 
                }
            }
            
            if( realLabel != nearestLabel )
                numMissClassifications += 1.0;
        }
        
        // return the error rate
        return (double)numMissClassifications / (double)numTrainInstances;
    }
	
	public void AddCandidateDistancesToDistancesMatrix(double [] candidateDistances)
	{
		double diff = 0;
		
		for(int i = 0;  i < numTrainInstances; i++)
			for(int j = i+1;  j < numTrainInstances; j++)
			{
				diff = candidateDistances[i]-candidateDistances[j];				
				seriesDistancesMatrix[i][j] += diff*diff;
			}
	}
	
	public void RemoveCandidateDistancesToDistancesMatrix(double [] candidateDistances)
	{
		double diff = 0;
		
		for(int i = 0;  i < numTrainInstances; i++)
			for(int j = i+1;  j < numTrainInstances; j++)
			{
				diff = candidateDistances[i]-candidateDistances[j];				
				seriesDistancesMatrix[i][j] -= diff*diff;
			}
	}
	
	// compute the error of the current histogram
	public double ComputeTestError()
	{
		LoadTestData();
    	
		// classify test data
    	int numMissClassifications = 0;
		
		int numTestInstances = testSeriesData.getDimRows();
		int numShapelets = distancesShapelets.size();
		
		int shapeletLength = 0;
		
		double minDistanceSoFar = Double.MAX_VALUE;
		double distanceToSegment = Double.MAX_VALUE;
		double diff = Double.MAX_VALUE;
		
		double [] distTestInstanceToShapelets = new double[numShapelets];
        
        // for every test instance 
        for(int i = 0; i < numTestInstances; i++)
        {
            double realLabel = testSeriesLabels.get(i); 
            
            double nearestLabel = 0;
            double nearestDistance = Double.MAX_VALUE;

            // compute the distances of the test instance to the shapelets
            for(int shapeletIndex = 0; shapeletIndex < numShapelets; shapeletIndex++)
        	{
            	minDistanceSoFar = Double.MAX_VALUE; 
    			// read the shapelet length
            	shapeletLength = acceptedList.get(shapeletIndex).length;
            	
    			for( int j = 0; j < seriesLength-shapeletLength+1; j++ ) 
        		{
    				distanceToSegment = 0; 
    				
    				for(int k = 0; k < shapeletLength; k++) 
    				{
    					diff = acceptedList.get(shapeletIndex)[k] - testSeriesData.get(i, j + k); 
    					distanceToSegment += diff*diff;  
    					
    					// if the distance of the candidate to this segment is more than the best so far
    					// at point k, skip the remaining points
    					if( distanceToSegment > minDistanceSoFar ) 
    						break; 
    				} 
    				
    				if( distanceToSegment < minDistanceSoFar) 
    					minDistanceSoFar = distanceToSegment; 
        		} 
    			
    			distTestInstanceToShapelets[shapeletIndex] = minDistanceSoFar; 
        	}
            
            // iterate through training instances and find the closest neighbours
            for(int j = 0; j < numTrainInstances; j++)
            {   	               
            	double distance = 0;
            	
            	for(int k = 0; k < numShapelets; k++)
            	{
            		double error = distTestInstanceToShapelets[k] - distancesShapelets.get(k)[j];
            		distance += error*error; 
            	}
            	
                
                // if there are less then required candidates then add it as a candidate directly
                if( distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestLabel = trainSeriesLabels.get(j); 
                }
            }
            
            if( realLabel != nearestLabel )
                numMissClassifications += 1.0;
            
        }
        
        return (double) numMissClassifications / (double) numTestInstances;
    }
	
	// is a candidate found in the accepted list of rejected list 
	public boolean FoundInList(double [] candidate, List<double[]> list) 
	{
		double diff = 0, distance = 0; 
		int shapeletLength = candidate.length; 
		
		for(double [] shapelet : list) 
		{ 
			// avoid comparing against shapelets of other lengths
			if(shapelet.length != candidate.length) 
				continue; 
			
			distance = 0;
			for(int k = 0; k < shapeletLength; k++)
			{
				diff = candidate[k]- shapelet[k]; 
				distance += diff*diff; 
				
				// if the distance so far exceeds epsilon then stop
				if(distance/shapeletLength > epsilon) 
					break; 
				
			}
			
			if( distance/shapeletLength < epsilon ) 
					return true;			
		}
		
		return false;
	}
		
	
	// estimate the pruning distance
	public double EstimateEpsilon()
	{
		// return 0 epsilon if no pruning is requested, i.e. percentile=0
		if (percentile == 0)
			return 0;
		
		int numPairs = trainSeriesData.getDimRows()*trainSeriesData.getDimColumns();
				
		double [] distances = new double[numPairs]; 
		
		int seriesIndex1 = -1, pointIndex1 = -1, seriesIndex2 = -1, pointIndex2 = -1;
		double pairDistance = 0, diff = 0;
		int shapeletLength = 0;
		
		DescriptiveStatistics stat = new DescriptiveStatistics();
		
		for(int i = 0; i < numPairs; i++)
		{
			shapeletLength = shapeletLengths[ rand.nextInt(shapeletLengths.length) ];
			
			seriesIndex1 = rand.nextInt( trainSeriesData.getDimRows() );
			pointIndex1 = rand.nextInt( seriesLength - shapeletLength + 1 );
			
			seriesIndex2 = rand.nextInt( trainSeriesData.getDimRows() );
			pointIndex2 = rand.nextInt( seriesLength - shapeletLength + 1 );
			
			pairDistance = 0;
			for(int k = 0; k < shapeletLength; k++)
			{
				diff = trainSeriesData.get(seriesIndex1, pointIndex1 + k) - trainSeriesData.get(seriesIndex2, pointIndex2 + k);
				pairDistance += diff*diff;
			}  
			
			distances[i] = pairDistance/(double)shapeletLength;	
			
			stat.addValue(distances[i]); 
			
		} 
		
		return stat.getPercentile(percentile); 
	}
	
		
	public void LoadTrainData()
	{
		DataSet trainSet = new DataSet();
		trainSet.LoadDataSetFile(new File(trainSetPath));
		
		if(normalizeData)
			trainSet.NormalizeDatasetInstances(); 
		
		trainSeriesLabels = new Matrix();
		trainSeriesLabels.LoadDatasetLabels(trainSet, false);
		
		// if 0< paaRatio < 1 then reduce the data dimensionality otherwise not
		if( paaRatio > 0.0 && paaRatio < 1.0 )
		{
			SAXRepresentation sr = new SAXRepresentation();
			trainSeriesData = sr.generatePAAToMatrix(trainSet, paaRatio);
		}
		else
		{
			trainSeriesData = new Matrix();
            trainSeriesData.LoadDatasetFeatures(trainSet, false);
		}
	}


	public void LoadTestData()
	{
		DataSet testSet = new DataSet();
		testSet.LoadDataSetFile(new File(testSetPath));
		
		if(normalizeData)
			testSet.NormalizeDatasetInstances();   
		
		testSeriesLabels = new Matrix();
		testSeriesLabels.LoadDatasetLabels(testSet, false);
		
		// if 0< paaRatio < 1 then reduce the data dimensionality otherwise not
		if( paaRatio > 0.0 && paaRatio < 1.0 )
		{
			SAXRepresentation sr = new SAXRepresentation();
			testSeriesData = sr.generatePAAToMatrix(testSet, paaRatio);
		}
		else
		{
			testSeriesData = new Matrix();
            testSeriesData.LoadDatasetFeatures(testSet, false);
		}
	}
	
	// the main function
	public static void main(String [] args)
	{
		String sp = File.separator;
		
		if (args.length == 0) {
			args = new String[] { 
					"dir=E:\\Data\\classification\\timeseries\\",
					"ds=NonInvasiveFatalECGThorax2", 
					"paaRatio=0.125", 
					"percentile=25", 
					"numTrials=5" 
			}; 
		}
		
		// initialize variables
		String dir = "", ds = "";
		int percentile = 0, numTrials = 1;
		double paaRatio = 0.0;
		
		// parse command line arguments
		for (String arg : args) 
		{
			String[] argTokens = arg.split("=");

			if (argTokens[0].compareTo("dir") == 0)
				dir = argTokens[1];
			else if (argTokens[0].compareTo("ds") == 0)
				ds = argTokens[1];
			else if (argTokens[0].compareTo("paaRatio") == 0)
				paaRatio = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("percentile") == 0)
				percentile = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("numTrials") == 0)
				numTrials = Integer.parseInt(argTokens[1]);				
		}

		// set the paths of the train and test files
		String trainSetPath = dir + ds + sp + "folds" + sp + "default" + sp + ds + "_TRAIN";
		String testSetPath = dir + ds + sp + "folds" + sp + "default" + sp + ds + "_TEST";
		
		// run the algorithm a number of times times
		double [] errorRates = new double[numTrials];
        double [] trainTimes = new double[numTrials];
        double [] totalTimes = new double[numTrials];
        double [] numAccepted = new double[numTrials];
        
        for(int trial = 0; trial < numTrials; trial++)
        {
			long startMethodTime = System.currentTimeMillis(); 
        	
			ScalableShapeletDiscovery ssd = new ScalableShapeletDiscovery();
            ssd.trainSetPath = trainSetPath;
            ssd.testSetPath = testSetPath; 
            ssd.percentile = percentile;
            ssd.paaRatio = paaRatio;
            ssd.normalizeData = false;  
            ssd.Search(); 		            
            
			double elapsedMethodTime = System.currentTimeMillis() - startMethodTime;
			
			double errorRate = ssd.ComputeTestError(); 
			
			double testTime = System.currentTimeMillis() - startMethodTime; 
			
			errorRates[trial] = errorRate;
			trainTimes[trial] = elapsedMethodTime/1000; // in second
			totalTimes[trial] = testTime/1000; // in second
			numAccepted[trial] = ssd.numAcceptedShapelets;
			
			
			System.out.println( 
					ds + " Trial="+trial+ ", " + 
					"Error="+errorRates[trial] +  
					", TrainTime=" + trainTimes[trial] + " " +
					", TotalTime=" + totalTimes[trial] + " " + 
					", nAccepted= " + numAccepted[trial] + " " +
							+ ssd.paaRatio	+ " "
							+ ssd.epsilon);
						
        } 

        
        System.out.println(
        		ds + " " + paaRatio + ", " + percentile + ", " + numTrials + ", " +
        				StatisticalUtilities.Mean(errorRates) + ", " + 
        				StatisticalUtilities.StandardDeviation(errorRates) + ", " +
        				StatisticalUtilities.Mean(trainTimes) + ", " + 
        				StatisticalUtilities.StandardDeviation(trainTimes) + ", " +
        				StatisticalUtilities.Mean(totalTimes) + ", " + 
        				StatisticalUtilities.StandardDeviation(totalTimes) + ", " + 
        				StatisticalUtilities.Mean(numAccepted) + ", " + 
    					StatisticalUtilities.StandardDeviation(numAccepted)  ); 
		
	}
	

}
