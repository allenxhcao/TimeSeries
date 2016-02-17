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


public class NaiveLinearSmo
{
	// box constraint parameter
	public double C = 1.0;
	
	// tolerance for margin and epsilon for alpha change
	public double tolerance = 1.0E-4;
	public double eps = 1.0E-14;
	
	int maxIterations = 1000;
 
  
	public List<Double> alphas; /* Lagrange multipliers */
	double b = 0;                        /* threshold */

	// the features and the labels
	Matrix features;
	List<Double> labels;
	
	int n_train;
	
	double delta_b=0;
	
	/*
	 * Constructor initializes the data matrix
	 * parameters are the features matrix, the labels matrix
	 * and the number of training instances in the features matrix
	 * i.e [0-numTrain-1] are train and the others [numTrain...] test 
	 * (semi-supervised)
	 */
	public NaiveLinearSmo(Matrix feats, List<Double> labs, int numTrain, 
					double boxConstraint)
	{
		features = feats;
		labels = labs;
		n_train = numTrain;
		C = boxConstraint;
		
		Initialize();
	}
	
	/*
	 * constructor with matrix labels
	 */
	public NaiveLinearSmo(Matrix feats, Matrix labsMat, int numTrain, 
			double boxConstraint)
	{
		List<Double> labs = new ArrayList<Double>();
		
		for(int i = 0; i < labsMat.getDimRows(); i++)
			labs.add( labsMat.get(i) ); 
		
		features = feats;
		labels = labs;
		n_train = numTrain;
		C = boxConstraint;
		
		Initialize();
	} 
	
	
	
	/*
	 * Initialize alphas, beta and error cache
	 */
	public void Initialize()
	{
		alphas = new ArrayList<Double>();
		
		for(int i = 0; i < n_train; i++)
		{
			alphas.add(0.0);
		}
		
		b = 0;
		
	}

	int examineExample(int i1)
	{
		//Logging.println("Examine: " + i1, LogLevel.DEBUGGING_LOG);
		
		double y1=0, alph1=0, E1=0, r1=0; 
		
		y1 = labels.get(i1);
		alph1 = alphas.get(i1);
		
		E1 = FunctionalMargin(i1) - y1;
		
 
		r1 = y1 * E1;
		
		// pick an i2 that violates KKT
		if ((r1 < -tolerance && alph1 < C) || (r1 > tolerance && alph1 > 0))
		{       
			double rands;
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
		
		E1 = FunctionalMargin(i1) - y1;
  
		alpha2 = alphas.get(i2);
		y2 = labels.get(i2);

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
		return features.RowDotProduct(i, j); 
	}
	
	/*
	 * solve the svm, if the parameter is true, the optimization
	 * runs only for one iteration
	 */
	public void Optimize(  )
	{
		Optimize(maxIterations);
	}
	
	// optimize and return the number of changed alphas 
	public int Optimize( int numIterations )
	{
		//Initialize();
		
		System.out.println("Entered SMO Optimize: " + numIterations);
		
		int numChanged = 1;
		
		int iterationCount = 0;
		
		while (numChanged > 0) 
		{
			numChanged = 0;
			
			// iterate over all instances
			for (int k = 0; k < n_train; k++)
				numChanged += examineExample (k);
			
			// check whether we are running only one iteration
			if( ++iterationCount >= numIterations )
				break;
			
			//double loss = GetLoss();
			//Logging.println("Loss="+loss + ", changed="+numChanged+ ".", LogLevel.DEBUGGING_LOG);
			
		}
		
		//double mcr = MCRTest();
		//Logging.println("SMO, Iteration=" + iterationCount + ", MCR=" + mcr + ", NumberAlphasChanged=" + numChanged, LogLevel.DEBUGGING_LOG);
		 
		return numChanged;
	}
	
	public double GetLoss()
	{
		double loss = 0;
		
		double sum = 0;
		for(int i = 0; i < n_train; i++)
			for(int l = 0; l < n_train; l++)
			{
				// skip if any of labels is zero
				if( labels.get(i) == 0 || labels.get(l) == 0 )
					continue;
				
				sum += labels.get(i)*alphas.get(i)*labels.get(l)*alphas.get(l)* K(i, l);
			}
		
		loss += 0.5*sum;

		sum = 0;
		for(int i = 0; i < n_train; i++)
			sum += alphas.get(i);
		
		loss -= sum;

		
		return loss;
	}

}