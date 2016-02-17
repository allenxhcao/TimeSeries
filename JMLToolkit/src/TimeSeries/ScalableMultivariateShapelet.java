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
import Utilities.ReadMultivariateSeriesDataset;
import Utilities.StatisticalUtilities;
import Utilities.Logging.LogLevel;
import Classification.WekaClassifierInterface;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;

public class ScalableMultivariateShapelet 
{	
	// series data and labels
	public double[][][] trainSeriesData;
	public double [] trainSeriesLabels; 
	
	public int numTrainInstances;
	public int numChannels;
	public int numPoints;
	// the delta offset 
	
	// the length of the shapelet we are searching for
	public int shapeletLength;
	
	// the epsilon parameter to prune candidates being epsilon close to a 
	// rejected or accepted shapelet
	public double epsilon;
	
	// the percentile for the distribution of distances between pairs of segments
	// is used as epsilon
	public int percentile;
	
	// list of accepted and rejected words
	List<double [][]> acceptedList = new ArrayList<double [][]>();
	List<double [][]> rejectedList = new ArrayList<double [][]>();
	
	
	public String trainSeriesPath, trainLabelsPath;
	
	
	// the histogram contains lists of frequency columns
	List< double [] > distancesShapelets = new ArrayList<double []>();
	// the current classification error of the frequencies histogram
	double currentTrainError = Double.MAX_VALUE;
	
	double [][] seriesDistancesMatrix;
	
	// logs on the number of acceptances and rejections
	public int numAcceptedShapelets, numRejectedShapelets, numRefusedShapelets;
	
	public long trainTime, testTime;
	
	// random number generator
	Random rand = new Random();
	
	public ScalableMultivariateShapelet()
	{
		
	}
	
	public double Search()
    {		
		long startTime = System.currentTimeMillis();
		
		// load train data
		Initialize();
		
		numAcceptedShapelets = numRejectedShapelets = numRefusedShapelets = 0;

    	//System.out.println(epsilon); 
    	
    	// set distances matrix to 0.0
    	seriesDistancesMatrix = new double[numTrainInstances][numTrainInstances];
    	for(int i = 0;  i < numTrainInstances; i++)
			for(int j = i+1;  j < numTrainInstances; j++)
				seriesDistancesMatrix[i][j] = 0.0;
    	
    	// evaluate all the words of all series
    	int numTotalCandidates = numTrainInstances*numPoints;
    	
    	
    	
    	//Logging.println("Candidate shapelets:");
    	
    	for( int candidateIdx = 0; candidateIdx < numTotalCandidates; candidateIdx++)
    	{
    		// select a random series
    		int seriesId = rand.nextInt(numTrainInstances);
    		// select a start time
        	int startTimeId = rand.nextInt(numPoints-shapeletLength+1); 
    			
			double [][] candidateShapelet = new double[numChannels][shapeletLength];
			
			for(int channelId = 0; channelId < numChannels; channelId++)
				for(int timeId = 0; timeId < shapeletLength; timeId++)
					candidateShapelet[channelId][timeId] = trainSeriesData[seriesId][channelId][startTimeId+timeId];
    			
			EvaluateShapelet(candidateShapelet);
			
			//Logging.println(candidateShapelet); 
			
			// if( candidateIdx % 1 == 0)
				Logging.println("candidateIdx="+candidateIdx + ", currentTrainError=" + currentTrainError + ", numAcceptedShapelets=" + numAcceptedShapelets + ", numRejectedShapelets=" + numRejectedShapelets + ", numRefusedShapelets=" + numRefusedShapelets);
			
		}
	    
    	long timeEndTraining = System.currentTimeMillis();
    	
    	trainTime = timeEndTraining - startTime;

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
    	
    	// load test data
    	//LoadTestData();
    	// classify test data
    	double testError = 1.0;//ComputeTestError();
    	
    	testTime = System.currentTimeMillis() - timeEndTraining;
    	
    	//Logging.println("TrainErr=" + currentTrainError+ ", TestErr=" + testError + ",nAcc=" + numAcceptedShapelets + ", nRej=" + numRejectedShapelets + ", nRep=" + numRepeatingShapelets + ", trainTime="+ trainTime/1000.0 + ", testTime="+ testTime/1000.0 ); 
    	 
		return testError;
    }
    
    // consider a word whether it is part of the accept list of reject list
    // and if not whether it helps reduce the classification error
    public void EvaluateShapelet(double [][] candidate)
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
	private double [] ComputeDistances(double [][] candidate)  
	{
		double [] distancesCandidate = new double[numTrainInstances];
		
		double diff = 0, distanceToSegment = 0, minDistanceSoFar = Double.MAX_VALUE;
		
		//System.out.println("shapeletLength="+shapeletLength); 
		
		for( int seriesId = 0; seriesId < numTrainInstances; seriesId++ ) 
		{ 
			minDistanceSoFar = Double.MAX_VALUE; 
			
			for( int startTimeId = 0; startTimeId < numPoints-shapeletLength+1; startTimeId++ ) 
    		{
				distanceToSegment = 0;  
				
				//System.out.println("segment"+segment+ ", segmentStartPoint="+segmentStartPoint); 
				
				for(int channelId=0; channelId<numChannels; channelId++)
				{
					for(int timeId = 0; timeId < shapeletLength; timeId++) 
					{
						//System.out.println("channel="+channel+", time="+time);
						
						diff = candidate[channelId][timeId]- trainSeriesData[seriesId][channelId][startTimeId+timeId];
						
						distanceToSegment += diff*diff;
					}
					
					if( distanceToSegment > minDistanceSoFar ) 
						break;
				}
				
				if( distanceToSegment < minDistanceSoFar) 
					minDistanceSoFar = distanceToSegment; 
    		} 
			
			distancesCandidate[seriesId] = minDistanceSoFar; 
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
        for(int seriesId = 0;  seriesId < numTrainInstances; seriesId++)
        {
            realLabel = trainSeriesLabels[seriesId]; 
            
            nearestLabel = -1; 
            nearestDistance = Double.MAX_VALUE; 

            // iterate through training instances and find the closest neighbours
            for(int j = 0; j < numTrainInstances; j++)
            {   
            	// avoid itself as a neighbor
            	if( seriesId == j) continue;
               
            	double distance = ( seriesId < j ? seriesDistancesMatrix[seriesId][j] : seriesDistancesMatrix[j][seriesId] );
            	
                if( distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestLabel = trainSeriesLabels[j]; 
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
	
//	// compute the error of the current histogram
//	public double ComputeTestError()
//	{
//		int numMissClassifications = 0;
//		
//		int numTestInstances = testSeriesData.getDimRows();
//		int numShapelets = distancesShapelets.size();
//		
//		int shapeletLength = 0;
//		
//		double minDistanceSoFar = Double.MAX_VALUE;
//		double distanceToSegment = Double.MAX_VALUE;
//		double diff = Double.MAX_VALUE;
//		
//		double [] distTestInstanceToShapelets = new double[numShapelets];
//        
//        // for every test instance 
//        for(int i = 0; i < numTestInstances; i++)
//        {
//            double realLabel = testSeriesLabels.get(i); 
//            
//            double nearestLabel = 0;
//            double nearestDistance = Double.MAX_VALUE;
//
//            // compute the distances of the test instance to the shapelets
//            for(int shapeletIndex = 0; shapeletIndex < numShapelets; shapeletIndex++)
//        	{
//            	minDistanceSoFar = Double.MAX_VALUE; 
//    			// read the shapelet length
//            	shapeletLength = acceptedList.get(shapeletIndex).length;
//            	
//    			for( int j = 0; j < seriesLength-shapeletLength+1; j++ ) 
//        		{
//    				distanceToSegment = 0; 
//    				
//    				for(int k = 0; k < shapeletLength; k++) 
//    				{
//    					diff = acceptedList.get(shapeletIndex)[k] - testSeriesData.get(i, j + k); 
//    					distanceToSegment += diff*diff;  
//    					
//    					// if the distance of the candidate to this segment is more than the best so far
//    					// at point k, skip the remaining points
//    					if( distanceToSegment > minDistanceSoFar ) 
//    						break; 
//    				} 
//    				
//    				if( distanceToSegment < minDistanceSoFar) 
//    					minDistanceSoFar = distanceToSegment; 
//        		} 
//    			
//    			distTestInstanceToShapelets[shapeletIndex] = minDistanceSoFar; 
//        	}
//            
//            // iterate through training instances and find the closest neighbours
//            for(int j = 0; j < numTrainInstances; j++)
//            {   	               
//            	double distance = 0;
//            	
//            	for(int k = 0; k < numShapelets; k++)
//            	{
//            		double error = distTestInstanceToShapelets[k] - distancesShapelets.get(k)[j];
//            		distance += error*error; 
//            	}
//            	
//                
//                // if there are less then required candidates then add it as a candidate directly
//                if( distance < nearestDistance)
//                {
//                    nearestDistance = distance;
//                    nearestLabel = trainSeriesLabels.get(j); 
//                }
//            }
//            
//            if( realLabel != nearestLabel )
//                numMissClassifications += 1.0;
//            
//        }
//        
//        return (double) numMissClassifications / (double) numTestInstances;
//    }
//	
	// is a candidate found in the accepted list of rejected list 
	public boolean FoundInList(double [][] candidate, List<double[][]> list) 
	{
		double diff = 0, distance = 0; 
		
		for(double [][] shapelet : list) 
		{ 
			distance = 0;
			for(int channelId=0; channelId<numChannels; channelId++)
			{
				for(int timeId=0; timeId<shapeletLength; timeId++)
				{
					diff = candidate[channelId][timeId]- shapelet[channelId][timeId]; 
					distance += diff*diff; 
					
					// if the distance so far exceeds epsilon then stop
					if(distance > epsilon) 
						break; 
				}
			}
			
			if( distance < epsilon ) 
					return true;			
		}
		
		return false;
	}
		
	
	// estimate the pruning distance
	public double EstimateEpsilon()
	{
		int numPairs = numTrainInstances*numPoints; 
		
		//System.out.println("numPairs="+numPairs);		
				
		double [] distances = new double[numPairs]; 
		
		int seriesIndex1 = -1, pointIndex1 = -1,  
			seriesIndex2 = -1, pointIndex2 = -1;
		double pairDistance = 0, diff = 0;
		
		DescriptiveStatistics stat = new DescriptiveStatistics();
		
		for(int i = 0; i < numPairs; i++)
		{			
			seriesIndex1 = rand.nextInt(numTrainInstances);
			pointIndex1 = rand.nextInt(numPoints-shapeletLength+1);
			
			seriesIndex2 = rand.nextInt(numTrainInstances);
			pointIndex2 = rand.nextInt(numPoints-shapeletLength+1);
			
			//System.out.println("pointIndex1="+pointIndex1+", pointIndex2="+pointIndex2);
			
			pairDistance = 0;
						
			for(int channelId=0; channelId<numChannels; channelId++)
			{
				for(int timeId = 0; timeId < shapeletLength; timeId++)
				{
					diff = trainSeriesData[seriesIndex1][channelId][pointIndex1+timeId] -
							trainSeriesData[seriesIndex2][channelId][pointIndex2+timeId]; 
					
					pairDistance += diff*diff;
				}
			}
			
			distances[i] = pairDistance;	
			
			//System.out.println("pointIndex1="+pointIndex1+", pointIndex2="+pointIndex2 + ", distance="+distances[i]);
			
			stat.addValue(distances[i]); 			
		} 	
		
		return stat.getPercentile(percentile); 
	}
	
		
	public void LoadTrainData()
	{
		ReadMultivariateSeriesDataset rmsd = new ReadMultivariateSeriesDataset();
		rmsd.LoadDatasetTimeSeries(trainSeriesPath);
		rmsd.LoadDatasetLabels(trainLabelsPath); 
		
		rmsd.NormalizeData();
		
		trainSeriesData = rmsd.X;
		trainSeriesLabels = rmsd.Y; 
		
		numTrainInstances = rmsd.numInstances;
		numPoints = rmsd.numPoints;
		numChannels =rmsd.numChannels;
	}

	// initialize the model
	public void Initialize()
	{
		// load the training data
		LoadTrainData();
		
		// estimate the pruning threshold epsilon
		epsilon = EstimateEpsilon();
		Logging.println("percentile="+ percentile + ", epsilon="+ epsilon);
		
	}

	// test the model 
	public static void main(String [] args)
	{
		ScalableMultivariateShapelet sms = new ScalableMultivariateShapelet();
		sms.trainSeriesPath =  "/home/jgrabocka/megData/text/train_subject01_ts.txt";
		sms.trainLabelsPath =  "/home/jgrabocka/megData/text/train_subject01_labels.txt";
		
		sms.shapeletLength = 50; 
		sms.percentile = 5; 
		
		sms.Search(); 
		
	}
}
