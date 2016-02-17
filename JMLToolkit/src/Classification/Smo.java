package Classification;

import java.util.*;
import java.lang.*; 
import java.io.*;
import java.rmi.*;
import java.math.*;

import Utilities.Logging;
import Utilities.Logging.LogLevel;

import DataStructures.DataSet;
import DataStructures.Matrix;


public class Smo
{
	// box constraint parameter
	public double C = 1.0;
	
	// tolerance for margin and epsilon for alpha change
	public double tolerance = 1.0E-4;
	public double eps = 1.0E-14;
	
	int maxIterations = 10000;
 
  
	public List<Double> alphas; /* Lagrange multipliers */
	double b = 0;                        /* threshold */

	// cache of precomputed errors
	List<Double> error_cache = new ArrayList<Double>();
	
	Matrix features, labels;
	int n_train;
	
	double delta_b=0;
	
	
	
	Kernel kernel;
	
	/*
	 * Constructor initializes the data matrix
	 * parameters are the features matrix, the labels matrix
	 * and the number of training instances in the features matrix
	 * i.e [0-numTrain-1] are train and the others [numTrain...] test 
	 * (semi-supervised)
	 */
	public Smo(Matrix feats, Matrix labs, int numTrain, 
					double boxConstraint, Kernel k)
	{
		features = feats;
		labels = labs;
		n_train = numTrain;
		C = boxConstraint;
		kernel = k;
		
		Initialize();
	}
	
	/*
	 * Initialize alphas, beta and error cache
	 */
	public void Initialize()
	{
		alphas = new ArrayList<Double>();
		error_cache = new ArrayList<Double>();
		
		for(int i = 0; i < n_train; i++)
		{
			alphas.add(0.0);
			error_cache.add(0.0);
		}
		
		b = 0;
		
	}

	int examineExample(int i1)
	{
		//Logging.println("Examine: " + i1, LogLevel.DEBUGGING_LOG);
		
		double y1=0, alph1=0, E1=0, r1=0; 
		
		y1 = labels.get(i1);
		alph1 = alphas.get(i1);
		
		if (alph1 > 0 && alph1 < C)
			E1 = error_cache.get(i1);
		else 
			E1 = FunctionalMargin(i1) - y1;;
		
 
		r1 = y1 * E1;
		
		// pick an i2 that violates KKT
		if ((r1 < -tolerance && alph1 < C) || (r1 > tolerance && alph1 > 0))
		{       
			// first heuristics, chose i2 that maximizes error difference
			{
				int k=0, i2=0;
				double tmax=0;
      
				for (i2 = (-1), tmax = 0, k = 0; k < n_train; k++)
					if (alphas.get(k) > 0 && alphas.get(k) < C) 
					{
						double Ek=0, temp=0;

						Ek = error_cache.get(k);
						temp = Math.abs(E1 - Ek);
						if (temp >= tmax)
						{
							tmax = temp;
							i2 = k;      
						}
					}
   
				if (i2 >= 0) 
				{
					if (BiOptimize (i1, i2)==1)
					{   
						return 1;
					}
				}
			}
			// second heuristics iterate over the non-bound examples
			// start at random instances
			double rands = 0;
			{
				int k=0, k0=0;
				int i2=0; 
   
				for (rands=Math.random(), k0 = (int)(rands*n_train), k = k0; k < n_train + k0; k++) 
				{
    
					i2 = k % n_train;
        
					if ( alphas.get(i2) > 0 && alphas.get(i2) < C) 
					{
						if (BiOptimize(i1, i2)==1)
						{
							return 1;
						}
					}
				}
			}

			// iterate through all remaining instances, 
			// start at random instances
			{      
				int k0=0, k=0, i2=0;
				rands = 0;
     
				for (rands=Math.random(), k0 = (int)(rands*n_train), k = k0; k < n_train + k0; k++) 
				{
					i2 = k % n_train;
					
					if (BiOptimize(i1, i2)== 1)
					{ 	
						return 1;
					}
				}
			}
		}
		
		return 0;
	}
	
	
	// bi-optimize for two alphas alpha_i1 and alpha_i2
	int BiOptimize(int i1, int i2) 
	{ 
		double y1=0, y2=0, s=0;
		double alpha1=0, alpha2=0; /* old_values of alpha_1, alpha_2 */
		double a1=0, a2=0;       /* new values of alpha_1, alpha_2 */
		double E1=0, E2=0, L=0, H=0, k11=0, k22=0, k12=0, eta=0, Lobj=0, Hobj=0;

		//Logging.println("TakeStep(" + i1 + "," + i2 + ")", LogLevel.DEBUGGING_LOG);
		
		if (i1 == i2) 
			return 0;
    
		alpha1 = alphas.get(i1);
		y1 = labels.get(i1);
		
		if (alpha1 > 0 && alpha1 < C)
			E1 = error_cache.get(i1);
		else 
			E1 = FunctionalMargin(i1) - y1;
  
		alpha2 = alphas.get(i2);
		y2 = labels.get(i2);

		if (alpha2 > 0 && alpha2 < C)
			E2 = error_cache.get(i2);
		else 
			E2 = FunctionalMargin(i2) - y2; 
  
		s = y1 * y2;

  
		if (y1 == y2) 
		{
			double gamma = alpha1 + alpha2;
			if (gamma > C) 
			{
				L = gamma-C;
				H = C;
			}
			else 
			{
				L = 0;
				H = gamma;
			}
		}
		else 
		{
			double gamma = alpha1 - alpha2;
			if (gamma > 0) 
			{
				L = 0;
				H = C - gamma;
			}
			else 
			{
				L = -gamma;
				H = C;
			}
		}
		
		if (L == H)
		{
			return 0;
		}

		k11 = K(i1, i1);
		k12 = K(i1, i2);
		k22 = K(i2, i2);
		eta = 2 * k12 - k11 - k22;

		if (eta < 0) 
		{
			a2 = alpha2 + y2 * (E2 - E1) / eta;
			if (a2 < L)
				a2 = L;
			else if (a2 > H)
				a2 = H;
		}
		else 
		{
    		double c1 = eta/2;
			double c2 = y2 * (E1-E2)- eta * alpha2;
			Lobj = c1 * L * L + c2 * L;
			Hobj = c1 * H * H + c2 * H;

			if (Lobj > Hobj+eps)
				a2 = L;
			else if (Lobj < Hobj-eps)
				a2 = H;
			else
				a2 = alpha2;
		}

		if (Math.abs(a2-alpha2) < eps*(a2+alpha2+eps))
			return 0;

		a1 = alpha1 - s * (a2 - alpha2);
		if (a1 < 0) 
		{
			a2 += s * a1;
			a1 = 0;
		}
		else if (a1 > C) 
		{
			double t = a1-C;
			a2 += s * t;
			a1 = C;
		}

		/*
		 * Compute the beta
		 */
		double b1=0, b2=0, bnew=0;
  
		if (a1 > 0 && a1 < C)
			bnew = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
		else 
		{
			if (a2 > 0 && a2 < C)
				bnew = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
			else 
			{
				b1 = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
				b2 = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
				bnew = (b1 + b2) / 2;
			}
		}
  
		delta_b = bnew - b;
		b = bnew;
  
		/*
		 * Update the cache
		 */
		
		{
			double t1 = y1 * (a1-alpha1);
			double t2 = y2 * (a2-alpha2);
  
			for (int i=0; i<n_train; i++)
				if (0 < alphas.get(i) && alphas.get(i) < C)
				{  
					double tmp = error_cache.get(i);
					tmp +=  t1*K(i1,i) + t2*K(i2,i) - delta_b;
					error_cache.set(i, tmp);
				}
			error_cache.set(i1, 0.0);
			error_cache.set(i2, 0.0);

		}
		
		/*
		 * Finally set the new alphas
		 */

		alphas.set(i1,a1);
		alphas.set(i2,a2);

		return 1;
	}

	// the missclassification rate of the test instances
	double MCRTest()
	{
		double n_total = 0;
		double n_error = 0;
		
		for (int i=n_train; i < features.getDimRows(); i++) 
		{
			if ( labels.get(i) * FunctionalMargin(i) < 0 )
				n_error++;
			
			n_total++;
		}
		return n_error/n_total;
	}


	/*
	 * The functional margin of instance i
	 */
	public double FunctionalMargin(int i)
	{
		double s = 0;
		
		for (int j = 0; j < n_train; j++)
			if ( alphas.get(j) > 0)
			{   
				s += alphas.get(j) * labels.get(j) * K(i,j); 			
			}
		
		s -= b;
		return s;
	}		

	/*
	 * The kernel of two instances
	 */
	double K(int i, int j)
	{
		double val = kernel.K(features, i, j);
		
		//System.out.println("K("+i+","+j+")=" + val );
		
		return val; 
	}
	
	/*
	 * solve the svm, if the parameter is true, the optimization
	 * runs only for one iteration
	 */
	public void Optimize(  )
	{
		Optimize(maxIterations);
	}
	
	public void Optimize( int numIterations )
	{
		//Initialize();
		
		int numChanged = 0;
		int examineAll = 1;
		
		int iterationCount = 0;
		
		while (numChanged > 0 || examineAll > 0) 
		{
			numChanged = 0;
				
			if (examineAll>0) 
			{     
				// iterate over all instances
				for (int k = 0; k < n_train; k++)
					numChanged += examineExample (k);
   
			}
			else 
			{ 
				// iterate over all unbound instances
				for (int k = 0; k < n_train; k++)
				{
					if (alphas.get(k) != 0 && alphas.get(k) != C)
						numChanged += examineExample(k);
				}
			}
			
			double mcr = MCRTest();
			Logging.println("SMO, Iteration=" + iterationCount + ", MCR=" + mcr + ", NumberAlphasChanged=" + numChanged, LogLevel.DEBUGGING_LOG);
			
			if (examineAll == 1)
				examineAll = 0;
			else if (numChanged == 0)
				examineAll = 1; 
			
			
			// check whether we are running only one iteration
			if( ++iterationCount > numIterations )
				break;
			
			
		}
		
		System.out.println("Error_rate="+ MCRTest()); 
	}

}