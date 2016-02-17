package Classification;

import Utilities.GlobalValues;
import DataStructures.Matrix;

public class Kernel 
{
	public enum KernelType{Linear, Polynomial, Gaussian, Euclidean};
	
	public KernelType type;
	
	// various possible kernel parameters
	public int degree;
	public double sig2;
	
	public Kernel()
	{
		type = KernelType.Polynomial;
	}
	
	public Kernel( Kernel k)
	{
		type = k.type;
		degree = k.degree;
		sig2 = k.sig2;
	}
	
	
	public Kernel(KernelType kernelType)
	{
		type = kernelType;
	}
	
	// the derivative
	public double K(Matrix m, int row1, int row2) 
	{
		double kernel = 0;
		
		if( type == KernelType.Linear )
			kernel = m.RowDotProduct(row1, row2); 
		else if( type == KernelType.Polynomial )
		{
			double dp = m.RowDotProduct(row1, row2);
			
			kernel = 1;
			for(int d=0; d < degree; d++)
				kernel *= (dp+1);
		}
		else if( type == KernelType.Gaussian )
			kernel = Math.exp(-m.RowEuclideanDistance(row1, row2)/sig2); 
		else if( type == KernelType.Euclidean )
			kernel = m.RowEuclideanDistance(row1, row2);
		
		return kernel;
	}
	
	public double K(double [] instance1, double [] instance2) 
	{
		double kernel = 0;
		
		if( type == KernelType.Linear )
			kernel = DotProduct(instance1, instance2); 
		else if( type == KernelType.Polynomial )
		{
			double dp = DotProduct(instance1, instance2);
			
			kernel = 1;
			for(int d=0; d < degree; d++)
				kernel *= (dp+1);
		}
		else if( type == KernelType.Gaussian )
		{
			
			kernel = Math.exp(-EuclideanDistance(instance1, instance2)/sig2);
		}
		
		return kernel;
	}
	
	
	public double DotProduct(double [] instance1, double [] instance2)
	{
		double dp = 0;
		
		for(int i = 0; i < instance1.length; i++)
			if( instance1[i] != GlobalValues.MISSING_VALUE && 
					instance2[i]  != GlobalValues.MISSING_VALUE )
						dp += instance1[i]*instance2[i];
		
		return dp;
	}
	
	public double EuclideanDistance(double [] instance1, double [] instance2)
	{
		double euclideanDistance = 0;
		
		for(int i = 0; i < instance1.length; i++)
		{
			if( instance1[i] != GlobalValues.MISSING_VALUE && 
					instance2[i]  != GlobalValues.MISSING_VALUE )
						{
							double val = instance1[i]-instance2[i];
							euclideanDistance += val*val;
						}
		}
				
		return euclideanDistance; 
	}
}
