package FeatureRich;

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;

import Regression.TimeSeriesPolynomialApproximation;
import TimeSeries.PolynomialRepresentation;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.StatisticalUtilities;

public class TimeSeriesTransformations 
{

	public double[][][] PolynomialTransformations(double [][] T, int segmentLength, int deltaPoints, int polyDegree)
	{
		int numSeries = T.length;
		int seriesLength = T[0].length;
		
		double[][][] S = new double[numSeries][][]; 
		
		TimeSeriesPolynomialApproximation tsPolyAppx = new TimeSeriesPolynomialApproximation(segmentLength, polyDegree);
		
		double [] segmentTemp = new double[segmentLength];
		
		for(int i = 0; i < numSeries; i++) 
        { 
        	// measure the number of segments
        	int numSegments = 0; 
        	for(int j = 0; j < seriesLength-segmentLength; j+=deltaPoints) 
        		numSegments++; 
        	
        	// initialize the i-th decomposed instance
        	S[i] = new double[numSegments][polyDegree+1]; 
        	
        	for(int j = 0; j < numSegments; j++) 
        	{ 
        		// create a copy of the segment
        		for(int l=0; l<segmentLength; l++)  
        			segmentTemp[l] = T[i][(j*deltaPoints)+l];  
        		
        		// normalize the segment
        		segmentTemp = StatisticalUtilities.Normalize(segmentTemp); 
        		
        		// convert the segment to polynomials 
        		S[i][j] = tsPolyAppx.FitPolynomialToSubSeries(segmentTemp);  
        	}
        }
		
		
		
		return S;
	}
	
	public double[][][] NormalizationTransformations(double [][] T, int segmentLength, int deltaPoints )
	{
		int numSeries = T.length;
		
		double[][][] S = new double[numSeries][][];  
		
		for(int i = 0; i < numSeries; i++) 
        { 
        	// measure the number of segments
        	int numSegments = 0; 
        	for(int j = 0; j < T[i].length-segmentLength; j+=deltaPoints) 
        		numSegments++; 
        	
        	// initialize the i-th decomposed instance 
        	S[i] = new double[numSegments][segmentLength];  
        	
        	for(int j = 0; j < numSegments; j++)  
        	{  
        		// create a copy of the segment 
        		for(int l=0; l<segmentLength; l++) 
        			S[i][j][l] = T[i][(j*deltaPoints)+l]; 
        		
        		// normalize the segment
        		S[i][j] = StatisticalUtilities.Normalize(S[i][j]); 
        	}
        }
		
		
		
		return S;
	}
	
	public double[][][] DerivativeTransformations(double [][] T, int segmentLength, int deltaPoints )
	{
		int numSeries = T.length;
		int seriesLength = T[0].length; 
		
		double[][][] S = new double[numSeries][][];  
		
		for(int i = 0; i < numSeries; i++) 
        { 
        	// measure the number of segments
        	int numSegments = 0; 
        	for(int j = 0; j < seriesLength-segmentLength; j+=deltaPoints) 
        		numSegments++; 
        	
        	// initialize the i-th decomposed instance 
        	S[i] = new double[numSegments][segmentLength];  
        	
        	double [] segmentContent = new double[segmentLength];
        	
        	for(int j = 0; j < numSegments; j++)  
        	{  
        		// create a copy of the segment 
        		for(int l=0; l<segmentLength; l++) 
        			segmentContent[l] = T[i][(j*deltaPoints)+l]; 
        		
        		// normalize the segment
        		segmentContent = StatisticalUtilities.Normalize(segmentContent); 
        		
        		// compute the derivative, set first point to zero
        		S[i][j][0] = 0;
        		for(int l=1; l<segmentLength; l++) 
        			S[i][j][l] = segmentContent[l]-segmentContent[l-1]; 
        	}
        }
		
		
		
		return S;
	}
	
	public double[][][] NormalizedOriginalAndDerivativeTransformations(double [][] T, int segmentLength, int deltaPoints )
	{
		int numSeries = T.length;
		int seriesLength = T[0].length; 
		
		double[][][] S = new double[numSeries][][];  
		
		for(int i = 0; i < numSeries; i++) 
        { 
        	// measure the number of segments
        	int numSegments = 0; 
        	for(int j = 0; j < seriesLength-segmentLength; j+=deltaPoints) 
        		numSegments++; 
        	
        	// initialize the i-th decomposed instance 
        	S[i] = new double[numSegments*2][segmentLength];  
        	
        	// first get the original segments
        	for(int j = 0; j < numSegments; j++)  
        	{  
        		// create a copy of the segment 
        		for(int l=0; l<segmentLength; l++) 
        			S[i][j][l] = T[i][(j*deltaPoints)+l]; 
        		
        		// normalize the segment
        		S[i][j] = StatisticalUtilities.Normalize(S[i][j]); 
        	}
        	
        	// then get the derivatives of the original segments
        	for(int j = numSegments; j < 2*numSegments; j++)  
        	{  
        		// compute the derivatives, set first point to zero
        		S[i][j][0] = 0;
        		for(int l=1; l<segmentLength; l++) 
        			S[i][j][l] = S[i][j-numSegments][l] - S[i][j-numSegments][l-1];
        		
        		// normalize the derivatives
        		S[i][j] = StatisticalUtilities.Normalize(S[i][j]); 
        	}
        }
		
		
		
		return S;
	}
	
	// the normalization segmentation and transformation of time-series
	public double[][][][] NormalizationTransformations(double [][] T, int R, int [] L, double delta )
	{
		int numSeries = T.length;
		
		double[][][][] S = new double[numSeries][R][][];   
		
		for(int i = 0; i < numSeries; i++) 
        { 
			for(int r = 0; r < R; r++) 
			{
				// the number of delta points at scale r
				int deltaPoints = (int) ( L[r]*delta );
				
				// measure the number of segments
	        	int numSegments = 0; 
	        	for(int j = 0; j < T[i].length-L[r]; j+=deltaPoints)  
	        		numSegments++; 
	        	
	        	// initialize the i-th decomposed instance 
	        	S[i][r] = new double[numSegments][L[r]]; 
	        	
	        	for(int j = 0; j < numSegments; j++)  
	        	{  
	        		// create a copy of the segment 
	        		for(int l=0; l<L[r]; l++)   
	        			S[i][r][j][l] = T[i][(j*deltaPoints)+l]; 
	        		
	        		// normalize the segment
	        		S[i][r][j] = StatisticalUtilities.Normalize(S[i][r][j]);  
	        	}
			}
        }
		
		return S;
	}
	
}
