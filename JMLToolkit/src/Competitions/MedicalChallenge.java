package Competitions;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;
















import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.stat.descriptive.rank.Min;

import Classification.MFSVM;
import Classification.SVMInterface;
import DataStructures.DataSet;
import DataStructures.Matrix;
import DataStructures.Tripple;
import DataStructures.Tripples;
import MatrixFactorization.MatrixFactorization;
import Regression.TimeSeriesPolynomialApproximation;
import TimeSeries.CIDDistance;
import TimeSeries.DTW;
import TimeSeries.EuclideanDistance;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.StatisticalUtilities;

public class MedicalChallenge 
{
	double [][] trainSeries;
	double [] trainLabels;
	
	double [][] testSeries;
	double [] testLabels;
	
	
	// relevant length in the end of the series
	public int L;
	// number of nearest neighbors
	public int K;
	
	// load the segments file
	public void LoadSegments(String dataSetFile, boolean isTrainingSet)
	{
		double [][] series = null;
		double [] labels = null;
		
		try
		{
			BufferedReader br = new BufferedReader(new FileReader( dataSetFile ));
			String line = null;
			int numSeries = 0;
			String delimiters = "\t ,;";
			
			// ommitt header row
			br.readLine();
			
			while( (line=br.readLine()) != null)
			{
				StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
				numSeries++;
			} 
			
			br.close();
			
			//Logging.println("K=" + K + ", L=" + L + ", numSeries=" + numSeries);  
			
			// initialize the segments
			series = new double[numSeries][L];
			labels = new double[numSeries];
			
			// load the segments
			br = new BufferedReader(new FileReader( dataSetFile ));
			line = null; 
			
			int lineCount = 0; 
			
			// ommitt header row
			br.readLine();
			
			while( (line=br.readLine()) != null)
			{
				StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
				
				// create a series that holds all the values of a line
				int numPoints = tokenizer.countTokens();
				double [] fullSeries = new double[numPoints];
				
				for(int l=0; l < numPoints; l++)
					fullSeries[l] = Double.parseDouble( tokenizer.nextToken() ); 
				
				// copy the last L points of the series
				for(int l=0; l < L; l++) 
					series[lineCount][l] = fullSeries[numPoints - L + l]; 
				
				
				// copy the label
				labels[lineCount] = fullSeries[0];
				
				lineCount++;
			}
			
			br.close();
			
		}
		catch(Exception exc)
		{
			exc.getStackTrace();
		}
		
		// set the series as train or test
		if( isTrainingSet )
		{
			trainSeries = series;
			trainLabels = labels;
		}
		else
		{
			testSeries = series;
			testLabels = labels; 
		}
	}
	
	
	public double[][] ConvertSeries(double [][] series)
	{
		double [][] convertedSeries = new double[series.length][L]; 
		
		for(int i=0; i<series.length; i++)
		{	
			for(int l=2; l < L; l++)	
			{	
				convertedSeries[i][l] = series[i][l]-series[i][l-2];	
			}	
		}	
		
		return convertedSeries;
	}
	
	public void ClassifyTestNearestNeighbor()
	{
		// classify instances in random order
		List<Integer> testIndicesList = new ArrayList<Integer>();
		for(int i=0; i < testSeries.length; i++)
			testIndicesList.add(i);
		
		Collections.shuffle(testIndicesList);
		
		double numCorrectlyClassified = 0.0;
		double numTotalInstances = 0.0; 
		
		for(int i : testIndicesList)
		{
			double realLabel = testLabels[i];  
			double probability = ClassifyInstance(testSeries, i);
			
			double predictedLabel = probability > 0.5 ? 1 : 0; 
			double isCorrectlyClassified = (realLabel == predictedLabel ? 1 : 0);
			double uncertainity = Math.abs( predictedLabel - probability ); 
			
			numCorrectlyClassified += isCorrectlyClassified;
			numTotalInstances += 1.0;
			
			System.out.println(numTotalInstances + ", " + realLabel + ", " +  predictedLabel  + ", " + isCorrectlyClassified 
 								+ ", " + probability + ", " + uncertainity 
 								+ ", " + numCorrectlyClassified / numTotalInstances );  
		}
	}
	
	// classify a test instance 
	public double ClassifyInstance(double [][] series, int seriesIndex)
	{
		// get the distances of the test instance to all the training instances
		double [] distances = ComputeDistances(series[seriesIndex]);
		// get the K-nearest neighbors of the test instance, i.e. the indices of 
		// the smallest distance values
		int [] nearestTrainingNeighborIndices = GetIndicesSmallerValues(distances);
		// get the average label value as the prediction, since this is a (0,1) binary problem
		// then the average can reprent the label
		double probability = 0; 
		for(int k=0; k < K; k++) 
			probability += trainLabels[nearestTrainingNeighborIndices[k]];  
		probability /= K; 
		
		return probability; 
	}
	
	// get the indices of the K maximum values 
	public int[] GetIndicesSmallerValues(double[] array) 
	{
	    double[] min = new double[K];
	    int[] minIndices = new int[K];
	    Arrays.fill(min, Double.POSITIVE_INFINITY); 
	    Arrays.fill(minIndices, -1);

	    top: for(int i = 0; i < array.length; i++) 
	    {
	        for(int j = 0; j < K; j++) 
	        {
	            if(array[i] < min[j])  
	            {
	                for(int x = K - 1; x > j; x--) 
	                {
	                    minIndices[x] = minIndices[x-1]; min[x] = min[x-1];
	                }
        
	                minIndices[j] = i; min[j] = array[i];
	                continue top;
	            }
	        }
	    }
	    
	    return minIndices;
	}
	
	// comput the distances of a test instance to all the training instances
	public double[] ComputeDistances(double [] series) 
	{
		double [] distances = new double[ trainSeries.length ]; 
		
		for(int j=0; j < trainSeries.length; j++)
			//distances[j] = EuclideanDistance(series, trainSeries[j]); 
			distances[j] = DTWDistance(series, trainSeries[j]);
			//distances[j] = ChangeDistance(series, trainSeries[j]);
			//distances[j] = CIDDistance.getInstance().CalculateDistance(series, trainSeries[j]); 

		return distances; 
	} 
	
	// compute the euclidean distance of two series
	public double EuclideanDistance(double[] series1, double[] series2)
	{
		double dist = 0;
		
		for( int l=0; l < L; l++ )
			dist += (series1[l]-series2[l])*(series1[l]-series2[l]); 
		
		return Math.sqrt(dist); 
	}
	
	public double DTWDistance(double [] ts1, double [] ts2) 
    {
        int n = ts1.length, m = ts2.length;
        
        double [][] costMatrix = new double[n][m]; 
        // initialize first cost cell
        costMatrix[0][0] = (ts1[0]-ts2[0])*(ts1[0]-ts2[0]);
        
        // compute distances in first row
        for(int i = 1; i < n; i++)
            costMatrix[i][0] = costMatrix[i-1][0] + (ts1[i]-ts2[0])*(ts1[i]-ts2[0]);
        // pre compute distances in first column
        for(int j = 1; j < m; j++)
            costMatrix[0][j] = costMatrix[0][j-1] + (ts1[0]-ts2[j])*(ts1[0]-ts2[j]);
        
        // compute the cost matrix
        for(int i = 1; i < n; i++)
        {
            for(int j = 1; j < m; j++)
            {
                double min = Math.min(costMatrix[i-1][j], costMatrix[i][j-1]);
                min = Math.min(costMatrix[i-1][j-1], min);
                
                costMatrix[i][j] = min + (ts1[i]-ts2[j])*(ts1[i]-ts2[j] );
            }
        }
         
        return costMatrix[n-1][m-1];
    }
	
	
	// compute the euclidean distance of two series
	public double ChangeDistance(double[] series1, double[] series2)
	{
		double change1 = series1[L-1]-series1[0];
		double change2 = series2[L-1]-series2[0];
		
		return (change2-change1)*(change2-change1);
	}
	
	public void PrintSlopes()
	{
		for(int i=0; i<testSeries.length; i++) 
		{
			double pct = (testSeries[i][L-1]/testSeries[i][0])*100.0 - 100.0;   
			
			System.out.println(pct + ", " + testLabels[i]);
		} 
		
	}
	
	public static void main(String [] args)
	{
		MedicalChallenge mch = new MedicalChallenge();
		
		mch.K = 10;
		mch.L = 30;    
		
		// mch.LoadSegments("E:\\Data\\classification\\timeseries\\medicalChallenge\\LATEST_0.2-TRAIN_SAMPLES-NEW_32_1000.csv", true); 
		// System.out.println("Loaded " + mch.trainSeries.length + " train series."); 
		
		mch.LoadSegments("E:\\Data\\classification\\timeseries\\medicalChallenge\\LATEST_0.2-TEST_SAMPLES-NEW_32_1000.csv", false); 
		//System.out.println("Loaded " + mch.testSeries.length + " test series."); 
		
		mch.PrintSlopes(); 
		
		//mch.trainSeries = mch.ConvertSeries(mch.trainSeries);  
		//mch.testSeries = mch.ConvertSeries(mch.testSeries);  
		
		//mch.ClassifyTestNearestNeighbor();
	}
	
}
