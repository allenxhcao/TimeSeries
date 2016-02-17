package MultivariateTimeSeries;


import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import TimeSeries.SAXRepresentation;
import Utilities.StatisticalUtilities;
import DataStructures.MultivariateDataset; 

public class ScalableShapeletDiscoveryMultivariate 
{	
	// series data and labels
	public MultivariateDataset dataSet;
	
	// the length of the shapelet we are searching for
	int [] shapeletLengths;
	
	// the epsilon parameter to prune candidates being epsilon close to a 
	// rejected or accepted shapelet
	public double [] epsilon;
	
	// the percentile for the distribution of distances between pairs of segments
	// is used as epsilon
	public int percentile;
	
	// list of accepted and rejected shapelets, separate per channel
	List<List<double []>> acceptedList;
	List<List<double []>> rejectedList;    

	
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
	
	public ScalableShapeletDiscoveryMultivariate()
	{
		normalizeData = false;
	}
	
	public void LoadData()
	{
		dataSet = new MultivariateDataset(trainSetPath, testSetPath, normalizeData);
		
		// apply the PAA
		SAXRepresentation sr = new SAXRepresentation();
		dataSet = sr.generatePAA(dataSet, paaRatio);
		
	}
	
	public void Search()
    {		
		acceptedList = new ArrayList<List<double[]>>(); 
		rejectedList = new ArrayList<List<double[]>>(); 

		
		// initialize the lists per each channel
		for(int channelIdx = 0; channelIdx < dataSet.numChannels; channelIdx++)
		{
			acceptedList.add( new ArrayList<double[]>() );
			rejectedList.add( new ArrayList<double[]>() );
		}
		
		numAcceptedShapelets = numRejectedShapelets = numRefusedShapelets = 0;
		    	
    	// check 20%,40%, 60% shapelet lengths    	
    	shapeletLengths = new int[3]; 
    	shapeletLengths[0] = (int)(0.20*dataSet.avgLength);
    	shapeletLengths[1] = (int)(0.40*dataSet.avgLength); 
    	shapeletLengths[2] = (int)(0.60*dataSet.avgLength);
    	
    	epsilon = new double[dataSet.numChannels];
    	for(int channelIdx = 0; channelIdx < dataSet.numChannels; channelIdx++)
    	{
    		epsilon[channelIdx] = EstimateEpsilon(channelIdx);
    		//System.out.println("channel=" + channelIdx + ", epsilon=" + epsilon[channelIdx] );
    	}
    	
    	// set distances matrix to 0.0
    	seriesDistancesMatrix = new double[dataSet.numTrain][dataSet.numTrain];
    	for(int i = 0;  i < dataSet.numTrain; i++)
			for(int j = i+1;  j < dataSet.numTrain; j++)
				seriesDistancesMatrix[i][j] = 0.0;
    	
    	// evaluate all the words of all series
    	int numTotalCandidates = dataSet.numTrain*dataSet.minLength*shapeletLengths.length;
    	
		//Logging.println("numTotalCandidates=" + numTotalCandidates );  

    	for( int candidateIdx = 0; candidateIdx < numTotalCandidates; candidateIdx++)
    	{
    		// select a random series
    		int i = rand.nextInt(dataSet.numTrain);
    		// select a random channel
    		int channel = dataSet.numChannels <= 1 ? 0: rand.nextInt(dataSet.numChannels);     
    		// select a random shapelet length 
        	int shapeletLength = shapeletLengths[ rand.nextInt(shapeletLengths.length) ];
        	// select a random segment of the i-th series where the shapelet can be located 
        	int maxTimeIndex = dataSet.timeseries[i][channel].length - shapeletLength + 1;
        	// avoid cases where the shapelet length is longer than the series
        	// because we cannot extract a candidate from that series
        	if( maxTimeIndex <= 0)
        		continue;
    		
        	// pick a start of the time indices
        	int j = rand.nextInt(maxTimeIndex);
        	// set the candidate shapelets
			double [] candidateShapelet = new double[shapeletLength];
			for(int k = 0; k < shapeletLength; k++)
				candidateShapelet[k] = dataSet.timeseries[i][channel][j + k];
			
    		// evaluate the shapelet
			EvaluateShapelet(candidateShapelet, channel);
			
			//if( candidateIdx % (numTotalCandidates/3) == 0) 
			//	Logging.println(candidateIdx + "," + currentTrainError + "," + numAcceptedShapelets + "," + numRejectedShapelets + "," + numRefusedShapelets); 
			
		}
	    	
    }
    
    // consider a word whether it is part of the accept list of reject list
    // and if not whether it helps reduce the classification error
    public void EvaluateShapelet(double [] candidate, int channel)
    {
    	// if the lists are both empty or the candidate is previously not been considered 
    	// then give it a chance
    	if( !FoundInList(candidate, acceptedList.get(channel), channel) &&  
    		!FoundInList(candidate, rejectedList.get(channel), channel) )  
    	{
    		// compute the soft frequencies of the word in the series data
    		double [] distancesCandidate = ComputeDistances(candidate, channel); 
    		
    		// refresh distances
    		AddCandidateDistancesToDistancesMatrix(distancesCandidate);
    		// compute error
    		double newTrainError = ComputeTrainError(); 
    		
    		if( newTrainError < currentTrainError)
    		{
    			// accept the word, which improves the error
    			acceptedList.get(channel).add(candidate);
    			
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
    			rejectedList.get(channel).add(candidate); 
    			
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
	private double [] ComputeDistances(double [] candidate, int channel)  
	{
		double [] distancesCandidate = new double[dataSet.numTrain + dataSet.numTest];
		
		double diff = 0, distanceToSegment = 0, minDistanceSoFar = Double.MAX_VALUE;
		
		for( int i = 0; i < dataSet.numTrain + dataSet.numTest; i++ ) 
		{ 
			// if the candidate is longer than the series then slide the series
			// accross the canidate
			if( candidate.length > dataSet.timeseries[i][channel].length )
			{
				minDistanceSoFar = Double.MAX_VALUE; 
				
				for(int j = 0; j < candidate.length - dataSet.timeseries[i][channel].length + 1; j++)
				{
					distanceToSegment = 0; 
					for(int k = 0; k < dataSet.timeseries[i][channel].length; k++)   
					{
						diff = candidate[j + k] - dataSet.timeseries[i][channel][k];  
						distanceToSegment += diff*diff;
						
						if( distanceToSegment > minDistanceSoFar ) 
							break; 
					} 
					
					if( distanceToSegment < minDistanceSoFar) 
						minDistanceSoFar = distanceToSegment; 
				}
				
				distancesCandidate[i] = minDistanceSoFar; 
			}
			else // slide the candidate accorss the series and keep the smallest distance
			{
				minDistanceSoFar = Double.MAX_VALUE; 
				
				for( int j = 0; j < dataSet.timeseries[i][channel].length-candidate.length+1; j++ ) 
	    		{
					distanceToSegment = 0; 
					
					for(int k = 0; k < candidate.length; k++) 
					{
						diff = candidate[k]- dataSet.timeseries[i][channel][j + k];  
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
        for(int i = 0;  i < dataSet.numTrain; i++)
        {
            realLabel = dataSet.labels[i]; 
            
            nearestLabel = -1; 
            nearestDistance = Double.MAX_VALUE; 

            // iterate through training instances and find the closest neighbours
            for(int j = 0; j < dataSet.numTrain; j++)
            {   
            	// avoid itself as a neighbor
            	if( i == j) continue;
               
            	double distance = ( i < j ? seriesDistancesMatrix[i][j] : seriesDistancesMatrix[j][i] );
            	
                if( distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestLabel = dataSet.labels[j]; 
                }
            }
            
            if( realLabel != nearestLabel )
                numMissClassifications += 1.0;
        }
        
        // return the error rate
        return (double)numMissClassifications / (double) dataSet.numTrain; 
    }
	
	public void AddCandidateDistancesToDistancesMatrix(double [] candidateDistances)
	{
		double diff = 0;
		
		for(int i = 0;  i < dataSet.numTrain; i++)
			for(int j = i+1;  j < dataSet.numTrain; j++)
			{
				diff = candidateDistances[i]-candidateDistances[j];				
				seriesDistancesMatrix[i][j] += diff*diff;
			}
	}
	
	public void RemoveCandidateDistancesToDistancesMatrix(double [] candidateDistances)
	{
		double diff = 0;
		
		for(int i = 0;  i < dataSet.numTrain; i++)
			for(int j = i+1;  j < dataSet.numTrain; j++)
			{
				diff = candidateDistances[i]-candidateDistances[j];				
				seriesDistancesMatrix[i][j] -= diff*diff;
			}
	}
	
	// compute the error of the current 
	
	public double ComputeTestError()
	{
		// classify test data
    	int numMissClassifications = 0;
		// the number of shapelets
		int numShapelets = distancesShapelets.size();
		
        // for every test instance 
        for(int i = dataSet.numTrain; i < dataSet.numTrain+dataSet.numTest; i++)
        {
            double realLabel = dataSet.labels[i];  
            
            double nearestLabel = 0;
            double nearestDistance = Double.MAX_VALUE;

            // iterate through training instances and find the closest neighbours
            for(int j = 0; j < dataSet.numTrain; j++)
            {
            	// compute the distance between the train and test instances
            	// in the shapelet-transformed space
            	double distance = 0;
            	for(int k = 0; k < numShapelets; k++)
            	{
            		double error = distancesShapelets.get(k)[i] - distancesShapelets.get(k)[j];
            		distance += error*error; 
            		// stop measuring the distance if it already exceeds the nearest distance so far
            		if(distance > nearestDistance)
            			break;
            	}
            	
                // if the distance is 
                if( distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestLabel = dataSet.labels[j]; 
                }
            }
            
            if( realLabel != nearestLabel )
                numMissClassifications += 1.0;
            
        }
        
        return (double) numMissClassifications / (double) dataSet.numTest;
    }
	
	
	// is a candidate found in the accepted list of rejected list 
	public boolean FoundInList(double [] candidate, List<double[]> list, int channel) 
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
				if( (distance/shapeletLength) > epsilon[channel] ) 
					break; 
				
			}
			
			if( (distance/shapeletLength) < epsilon[channel] ) 
					return true;			
		}
		
		return false;
	}
		
	
	// estimate the pruning distance
	public double EstimateEpsilon(int channel)
	{
		// return 0 epsilon if no pruning is requested, i.e. percentile=0
		if (percentile == 0)
			return 0;
		
		int numPairs = dataSet.numTrain*dataSet.minLength; 
				
		double [] distances = new double[numPairs]; 
		
		int seriesIndex1 = -1, pointIndex1 = -1, seriesIndex2 = -1, pointIndex2 = -1;
		double pairDistance = 0, diff = 0;
		int shapeletLength = 0;
		
		DescriptiveStatistics stat = new DescriptiveStatistics();
		
		for(int i = 0; i < numPairs; i++)
		{
			shapeletLength = shapeletLengths[ rand.nextInt(shapeletLengths.length) ];
			
			seriesIndex1 = rand.nextInt( dataSet.numTrain );
			int maxPoint1 = dataSet.timeseries[seriesIndex1][channel].length - shapeletLength + 1;
			seriesIndex2 = rand.nextInt( dataSet.numTrain );
			int maxPoint2 = dataSet.timeseries[seriesIndex2][channel].length - shapeletLength + 1;
			
			// avoid series having length less than the shapeletLength
			if( maxPoint1 <= 0 || maxPoint2 <= 0)
				continue;
			
			pointIndex1 = rand.nextInt( maxPoint1 ); 
			pointIndex2 = rand.nextInt( maxPoint2 ); 
			
			pairDistance = 0;
			for(int k = 0; k < shapeletLength; k++)
			{
				diff = dataSet.timeseries[seriesIndex1][channel][pointIndex1 + k] 
						- dataSet.timeseries[seriesIndex2][channel][pointIndex2 + k]; 
				pairDistance += diff*diff;
			}  
			
			distances[i] = pairDistance/(double)shapeletLength;	
			
			stat.addValue(distances[i]); 
			
		} 
		
		return stat.getPercentile(percentile); 
	}
	
		
	
	// the main function
	public static void main(String [] args)
	{
		String sp = File.separator;
		
		if (args.length == 0) {
			args = new String[] { 
					"trainFile=E:\\Data\\classification\\multivariateTimeseries\\mHealth\\mHealth_TRAIN",
					"testFile=E:\\Data\\classification\\multivariateTimeseries\\mHealth\\mHealth_TEST",
					"paaRatio=0.125", 
					"percentile=35", 
					"numTrials=5" 
			};
		}
		
		// initialize variables
		String dir = "", ds = "";
		int percentile = 0, numTrials = 1;
		double paaRatio = 0.0;
		
		// set the paths of the train and test files
		String trainSetPath = "";
		String testSetPath = "";
		
		// parse command line arguments
		for (String arg : args) 
		{
			String[] argTokens = arg.split("=");

			if (argTokens[0].compareTo("dir") == 0) 
				dir = argTokens[1]; 
			else if (argTokens[0].compareTo("trainFile") == 0) 
				trainSetPath = argTokens[1];
			else if (argTokens[0].compareTo("testFile") == 0) 
				testSetPath = argTokens[1];
			else if (argTokens[0].compareTo("paaRatio") == 0) 
				paaRatio = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("percentile") == 0)
				percentile = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("numTrials") == 0)
				numTrials = Integer.parseInt(argTokens[1]);				
		}

		
		// run the algorithm a number of times times
		double [] errorRates = new double[numTrials]; 
        double [] trainTimes = new double[numTrials]; 
        double [] totalTimes = new double[numTrials]; 
        double [] numAccepted = new double[numTrials]; 
        
        for(int trial = 0; trial < numTrials; trial++)
        {
			long startMethodTime = System.currentTimeMillis(); 
        	
			ScalableShapeletDiscoveryMultivariate ssd = new ScalableShapeletDiscoveryMultivariate();
            ssd.trainSetPath = trainSetPath;
            ssd.testSetPath = testSetPath; 
            ssd.percentile = percentile;
            ssd.paaRatio = paaRatio;
            ssd.normalizeData = true;  
            ssd.LoadData(); 
            ssd.Search();     
            
			double elapsedMethodTime = System.currentTimeMillis() - startMethodTime;
			
			double errorRate = ssd.ComputeTestError(); 
			
			double testTime = System.currentTimeMillis() - startMethodTime;  
			
			errorRates[trial] = errorRate;
			trainTimes[trial] = elapsedMethodTime/1000; // in second
			totalTimes[trial] = testTime/1000; // in second
			numAccepted[trial] = ssd.numAcceptedShapelets;
			
			
			System.out.println( 
					"Trial="+trial+ ", " + 
					"Error="+errorRates[trial] +  
					", TrainTime=" + trainTimes[trial] + " " +
					", TotalTime=" + totalTimes[trial] + " " + 
					", nAccepted= " + numAccepted[trial] + " " +
							+ ssd.paaRatio ); 
						
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
