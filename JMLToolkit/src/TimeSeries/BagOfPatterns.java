package TimeSeries;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import Regression.TimeSeriesPolynomialApproximation;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import DataStructures.DataSet;
import DataStructures.Matrix;

public class BagOfPatterns 
{
	// the size of the sliding window
	public int slidingWindowSize;
	// the size of the alphabet
	public int alphabetSize;
	
	// the maximum size of a collocations set 
	public int collocationSetSize; 
	
	// the sax representation of sliding window contents 
	public SAXRepresentation sr; 
	public int innerDimension; 
	
	public PolynomialRepresentation pr; 
	public int polyDegree; 
	
	public enum RepresentationType {SAX, Polynomial};
	public RepresentationType representationType;
	
	// dictionary created
	public List<String> dictionary;
	
	public BagOfPatterns()
	{
		sr = new SAXRepresentation();
		pr = new PolynomialRepresentation();
		
		representationType = RepresentationType.Polynomial;
		polyDegree = 4;
	}
	
	// create a dictionary from the bag of patterns of every time series
	public List<String> CreateWordsDictionary( List<List<String>> bagsOfPatterns )
	{
		Set<String> dictionarySet = new HashSet<String>();
		
		// add all bags to the dictionary, duplicates are discarded
		for( List<String> bag : bagsOfPatterns)
			for(String word : bag)
				dictionarySet.add(word);
	
		//Logging.println("Created dictionary of " + dictionarySet.size() + " subseries.", LogLevel.DEBUGGING_LOG);
		
		// return the dictionary set into a list structure
		return new ArrayList<String>(dictionarySet); 
	}
	
	public Matrix CreateWordFrequenciesMatrix(Matrix ds)
	{
		// get the bags of patterns, one bag per time series
		List<List<String>> bagsOfPatterns = null;
		
		if( representationType == RepresentationType.SAX)
			bagsOfPatterns = sr.ExtractBagOfPatterns(ds, slidingWindowSize, innerDimension, alphabetSize);
		else if( representationType == RepresentationType.Polynomial)
		{
			pr.polyRegression = new TimeSeriesPolynomialApproximation(slidingWindowSize, polyDegree);
			bagsOfPatterns = pr.ExtractBagOfPatterns(ds, slidingWindowSize, polyDegree, alphabetSize);
			
			Logging.println("slidingWindowSize=" + slidingWindowSize + ", alphabetSize="+alphabetSize +
					", polyDegree=" + polyDegree, LogLevel.DEBUGGING_LOG); 
		}
		
		// create a dictionary list where each time series is 
		dictionary = CreateWordsDictionary(bagsOfPatterns);
		
		// create a matrix for histogram
		Matrix H = new Matrix(ds.getDimRows(), dictionary.size());
		H.SetUniqueValue(0.0);
				
		// for every instance
		for(int instanceId = 0; instanceId < bagsOfPatterns.size(); instanceId++)
		{
			//Logging.print("Instance " + instanceId + ": [", LogLevel.DEBUGGING_LOG);
			
			// iterate through all the words in the bag
			for(int wordBagIndex = 0; wordBagIndex < bagsOfPatterns.get(instanceId).size(); wordBagIndex++)
			{
				// and increase the frequency in the respective histogram frequency column of the matrix
				String word = bagsOfPatterns.get(instanceId).get(wordBagIndex);
				int wordHistogramIndex = dictionary.indexOf(word); 

				// increment the frequency of the word in the histogram 
				H.set(instanceId, wordHistogramIndex, H.get(instanceId, wordHistogramIndex)+1);
			}			
		}
				
		Logging.println("Created histogram: Rows=" + H.getDimRows() + ", Columns=" + H.getDimColumns(), LogLevel.DEBUGGING_LOG);
		Logging.println("Created histogram: Sparsity=" + H.GetSparsityRatio() + " subseries.", LogLevel.DEBUGGING_LOG);
		
		return H;
	}
	
	public Matrix CreateCollocationFrequenciesMatrix(Matrix ds)
	{
		// get the bags of patterns, one bag per time series
		List<List<String>> bagsOfPatterns = null;
		
		if( representationType == RepresentationType.SAX)
			bagsOfPatterns = sr.ExtractBagOfPatterns(ds, slidingWindowSize, innerDimension, alphabetSize);
		else if( representationType == RepresentationType.Polynomial)
		{
			pr.polyRegression = new TimeSeriesPolynomialApproximation(slidingWindowSize, polyDegree);
			bagsOfPatterns = pr.ExtractBagOfPatterns(ds, slidingWindowSize, polyDegree, alphabetSize);
			
			Logging.println("slidingWindowSize=" + slidingWindowSize + ", alphabetSize="+alphabetSize +
					", polyDegree=" + polyDegree, LogLevel.DEBUGGING_LOG); 
		}
		
		// create a dictionary of words 
		dictionary = CreateWordsDictionary(bagsOfPatterns);
		
		// create a matrix for histogram
		Matrix H = new Matrix(ds.getDimRows(), dictionary.size());
		H.SetUniqueValue(0.0);
		
		// for every instance
		for(int instanceId = 0; instanceId < bagsOfPatterns.size(); instanceId++)
		{
			//Logging.print("Instance " + instanceId + ": [", LogLevel.DEBUGGING_LOG);
		
			// iterate through all the words in the bag
			for(int wordBagIndex = 0; wordBagIndex < bagsOfPatterns.get(instanceId).size(); wordBagIndex++)
			{
				// and increase the frequency in the respective histogram frequency column of the matrix
				String word = bagsOfPatterns.get(instanceId).get(wordBagIndex);
				int wordHistogramIndex = dictionary.indexOf(word); 

				// increment the frequency of the word in the histogram 
				H.set(instanceId, wordHistogramIndex, H.get(instanceId, wordHistogramIndex)+1);
			}
		
			//System.out.print( " ] \n" ); 
		}
				
		
		Logging.println("Created histogram: Rows=" + H.getDimRows() + ", Columns=" + H.getDimColumns(), LogLevel.DEBUGGING_LOG);
		Logging.println("Created histogram: Sparsity=" + H.GetSparsityRatio() + " subseries.", LogLevel.DEBUGGING_LOG);
		
		return H;
	}
}
