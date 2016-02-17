package Regression;

import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import libsvm.svm_model;

import DataStructures.Matrix;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import weka.classifiers.functions.SMOreg;
import weka.core.Utils;

public class TimeSeriesPolynomialApproximation 
{
	public int maxEpochs;
	
	public int seriesLength, degree;
	
	RealMatrix projection;
	RealMatrix target;
	
	double [] coeffs;
	
	public TimeSeriesPolynomialApproximation(int seriesLength, int polyDegree) 
	{
		// set the degree
		this.degree = polyDegree; 
		this.seriesLength = seriesLength;
		
		// initialize the predictors
		RealMatrix predictors = new Array2DRowRealMatrix(seriesLength, degree+1);
		
		int midPoint = seriesLength/2;
		
		for(int t = 0; t < seriesLength; t++) 
			for(int d = 0; d < degree+1; d++) 
				predictors.setEntry(t, d, Pow(t, d)); 
			
		RealMatrix predictorsTranspose =  predictors.transpose();
		
		RealMatrix mpPseudoInv = new QRDecomposition( predictorsTranspose.multiply(predictors) ).getSolver().getInverse();
		
		projection = mpPseudoInv.multiply(predictorsTranspose);
		
	}
	
	// compute the power val^pow
	public double Pow(int t, int pow)
    {
    	int result = 1; 
    	
    	for(int i = 0; i < pow; i++)
    		result *= t;
    	
    	return (double) result;
    }
	
	// learn using linear system of equations approach
	public double [] FitPolynomialToSubSeries(double [] subSeries)
	{
		//System.out.println(seriesLength);
		
		target = new Array2DRowRealMatrix(seriesLength, 1);
		for(int t = 0; t < seriesLength; t++)
			target.setEntry(t, 0, subSeries[t]);
		
		RealMatrix solution = projection.multiply(target);
		
		return solution.getColumn(0);
	}
	
	
}
