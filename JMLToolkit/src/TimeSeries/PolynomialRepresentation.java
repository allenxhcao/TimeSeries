package TimeSeries;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.xml.namespace.QName;

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.analysis.solvers.NewtonRaphsonSolver;
import org.apache.commons.math3.fitting.CurveFitter;
import org.apache.commons.math3.fitting.PolynomialFitter;
import org.apache.commons.math3.optim.nonlinear.vector.jacobian.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.optimization.general.AbstractLeastSquaresOptimizer;
import org.apache.commons.math3.optimization.general.GaussNewtonOptimizer;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.core.Utils;

import Regression.TimeSeriesPolynomialApproximation;
import Utilities.Logging;
import Utilities.StatisticalUtilities;
import Utilities.Logging.LogLevel;

import DataStructures.Matrix;

public class PolynomialRepresentation 
{
	String alphabet = "abcdefghijkl";
	
	TimeSeriesPolynomialApproximation polyRegression;
	
	// the percentiles
	double [][] percentiles;
	// the midpoint value of the coefficients
	double [][] midpoints;
	// the list of all polynomial coefficients, one list of coefficients per time series
	List<List<double []>> allPolyCoeffs;
	
	public PolynomialRepresentation()
	{
		
	}
	
	// fit a polynomial to the subsequence and return the coefficients
	public double [] FitPolynomial(double [] subSequence, double [] initialGuessCoeffss)
	{
		// create a curve fitter
		CurveFitter fitter = new CurveFitter(new LevenbergMarquardtOptimizer());
		
		for(int i = 0; i < subSequence.length; i++)  
			fitter.addObservedPoint(i, subSequence[i]); 
			
		// fit the coefficients
		return fitter.fit(new PolynomialFunction.Parametric(), initialGuessCoeffss);  
		
	}
	
	// iterate through all sliding window and collect polynomial coefficients
	// then generate quantiles from the distribution of values of each coefficient
	public void ComputeQuantiles(Matrix ds, int slidingWindowSize, int degree, int alphabetSize)
	{
		// initialize the descriptive statistics in order to 
		DescriptiveStatistics [] coeffStats = new DescriptiveStatistics[degree+1];
		for(int d = 0; d < degree+1; d++)
			coeffStats[d] = new DescriptiveStatistics();
		
		allPolyCoeffs = new ArrayList<List<double[]>>();
		
		// in the first step we gather all the polynomial coefficients
		// iterate through every time series
		for(int r = 0; r < ds.getDimRows(); r++)
		{
			List<double[]> seriesPolyCoeffs = new ArrayList<double[]>();
			
			// store the time series and its length
			double [] ins = ds.getRow(r);
			int numPoints = ins.length;
			
			double [] initialGuessCoeffs = new double[degree+1];
			
			// iterate through every sliding window
			for(int i = 0; numPoints - i >= slidingWindowSize ; i++)
	    	{
				// store the subseries
	    		double [] subSeries = new double[slidingWindowSize];
	    		for(int j = 0; j < slidingWindowSize; j++)
	    			subSeries[j] = ins[i+j];
	    		
	    		// normalize the subseries
	    		double [] subSeriesNorm = StatisticalUtilities.Normalize(subSeries);
	    		// retrieve the polynomial coefficients of the normalized sliding window subseries
	    		
	    		double [] coeffs = polyRegression.FitPolynomialToSubSeries(subSeriesNorm);
	    		
	    		Logging.print(coeffs, LogLevel.DEBUGGING_LOG);
	    		System.out.println();
	    		
	    		// add the coefficient values to the descriptive statistics
	    		for(int d = 0; d < degree+1; d++)
	    		{
	    			coeffStats[d].addValue(coeffs[d]);
	    			initialGuessCoeffs[d] = coeffs[d];
	    		}
				
	    		// add the coefficients to the list
	    		seriesPolyCoeffs.add(coeffs);
	    	}
			
			//Logging.println("Processed polynomials of series " + r, LogLevel.DEBUGGING_LOG);   
			
			allPolyCoeffs.add(seriesPolyCoeffs);
		}
		
		//System.out.println(allCoeffs.size());

		
			
		
		// compute the quantiles
		percentiles = new double[degree+1][alphabetSize];
		double quantileSize = 100.0/alphabetSize;
		
		for(int d = 0; d < degree+1; d++)
		{
			double mean = coeffStats[d].getMean();
			double std = coeffStats[d].getStandardDeviation();
			Logging.println("Statistics: Coeff=" + d + ", mean=" + mean + ", std=" + std + ", percentiles=[", LogLevel.DEBUGGING_LOG); 
			
			for( int percentileIndex = 0; percentileIndex < alphabetSize-1; percentileIndex++ )
			{
				System.out.print( (percentileIndex+1.0)*quantileSize + " ");
				
				percentiles[d][percentileIndex] = coeffStats[d].getPercentile( (percentileIndex+1.0)*quantileSize );
			}
			
			System.out.println(" ");
			percentiles[d][alphabetSize-1] = Double.MAX_VALUE;
						
			Logging.print(percentiles[d], LogLevel.DEBUGGING_LOG);
			Logging.print("]", LogLevel.DEBUGGING_LOG); 			
		}
		
		midpoints = new double[degree+1][alphabetSize];
		
		System.out.println("AlphabetMidpoints={");
		for(int d = 0; d < degree+1; d++)
		{
			System.out.print("beta-" + d + "=[ ");
			for( int k = 0; k < alphabetSize; k++ ) 
			{
				double alphabetMidPoint = coeffStats[d].getPercentile( ( ( 2*(k+1.0) -1 )/2.0)*quantileSize );
				midpoints[d][k] = alphabetMidPoint;
				System.out.print(alphabet.charAt(k) + "=" + alphabetMidPoint + " ");
			}
			System.out.println("]");
		}
		System.out.println("}");
		
		
	}
	
	// extract the bag of patterns for the whole dataset of time series
	public List<List<String>> ExtractBagOfPatterns(
			Matrix ds, 
			int slidingWindowSize, 
			int degree,
			int alphabetSize )
	{
		
		// set the degree of the polynomial regression
		polyRegression.degree = degree;
		
		// first of all generate the quantiles
		ComputeQuantiles(ds, slidingWindowSize, degree, alphabetSize);
		
		List<List<String>> bagOfPatterns = new ArrayList<List<String>>();
		
		for(int i = 0; i < ds.getDimRows(); i++)
			bagOfPatterns.add( ExtractBagOfPatterns(i, slidingWindowSize, degree, alphabetSize) );
		
		return bagOfPatterns;
	}
	
	public List<String> ExtractBagOfPatterns(
    		int seriesIndex, 
    		int slidingWindowSize, 
    		int degree,
    		int alphabetSize )
    {
    	List<String> bagOfPatterns = new ArrayList<String>();
    	
    	// initialize a word
    	String previousSubSeriesPolyWord = "zzzzzzzzzzzzzzzzzz";
    	
    	int numPolys = allPolyCoeffs.get(seriesIndex).size(); 
    	
    	for(int polyIdx = 0; polyIdx < numPolys; polyIdx++)
    	{
    		// convert the polynomial coefficients to a word
    		String subSeriesPolyWord = ConvertCoeffsToWord(
    										allPolyCoeffs.get(seriesIndex).get(polyIdx), 
    										alphabetSize);
    	
    		// apply numerosity reduction
    		if( previousSubSeriesPolyWord.compareTo(subSeriesPolyWord) != 0)
    		//if( true )
    		{
    			bagOfPatterns.add(subSeriesPolyWord);
    			previousSubSeriesPolyWord = subSeriesPolyWord;
    		}
    	}
    	
    	return bagOfPatterns;
    }
	
	// convert coefficients to a word
	 public String ConvertCoeffsToWord(double [] coefficients, int alphabetSize)
    {
    	String word = "";
    	
    	for(int i = 0; i < coefficients.length; i++)
    	{
    		int alphabetLetterIndex = -1;
    		
    		for(int percentileIndex = 0; percentileIndex < percentiles[i].length; percentileIndex++)
    		{
    			if( coefficients[i] < percentiles[i][percentileIndex] )
    			{
    				alphabetLetterIndex = percentileIndex;
    				break;
    			}
    		}
    		
    		String saxLetter = alphabet.substring(alphabetLetterIndex, alphabetLetterIndex+1);
    		
    		word += saxLetter;
    	}
    	
    	return word;
    }
	 
	 public double [] ConvertWordToCoeffs(String word, int alphabetSize)
    {
		 double [] coeffs = new double[word.length()];
		 
		 
		 for(int chIdx= 0; chIdx < word.length(); chIdx++)
		 {
			 coeffs[chIdx] =  midpoints[chIdx][alphabet.indexOf(word.charAt(chIdx)) ]; 
		 }
		
    	
    	return coeffs;
    }
	 
	 // test some functionalities
	 public static void main( String [] args )
	 {
		 
		 PolynomialRepresentation pr = new PolynomialRepresentation();
		 int degree = 5;
		 
		 double [] initCoeffs = new double[degree+1];
		 
		 Random rand = new Random();
		 
		 int seriesLength = 100;
		 double [] series = new double[seriesLength];
		 for(int i = 0; i < series.length; i++)
			 series[i] = 2*rand.nextDouble()-1;
		 
		 double [] coeffs = pr.FitPolynomial(series, initCoeffs); 		 
		 Logging.print(coeffs, LogLevel.DEBUGGING_LOG);
		 System.out.println("");

		 
		 TimeSeriesPolynomialApproximation myPr = new TimeSeriesPolynomialApproximation(seriesLength, degree);
		 coeffs = myPr.FitPolynomialToSubSeries(series);
		 Logging.print(coeffs, LogLevel.DEBUGGING_LOG);
		 System.out.println(""); 
		 
		 
	 }
	 
}
