package MatrixFactorization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import weka.experiment.LearningRateResultProducer;


import TimeSeries.DTW;
import TimeSeries.TransformationFieldsGenerator;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import Classification.InvariantSVM;
import Classification.Kernel;
import Classification.NaiveLinearSmo;
import Classification.Kernel.KernelType;
import Classification.NaiveSmo;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;

public class SimilaritySupervisedMF 
{
	// the matrix X, Y, U, Psi_i of the method
	Matrix X, S, U, V, Y;
	// number of instances of train and test sets
	private int n_train, n_test;
	// number of features
	private int m;
	// the latent dimensionality parameter
	public int d;
	// the box constraint parameter of SVM
	public double C;
	// the gamma of the gaussian kernel
	public double gamma;
	// the regularization paramters of U
	public double lambdaU, lambdaV;
	// the learning rate 
	public double eta1, eta2, eta3;
	// the maximum number of epocs
	public int maxEpocs;
	
	// a smo solver
	NaiveLinearSmo smo;
	Kernel kernel;
	
	// reciprocal of instances count and features count
	private double rec_n, rec_m;
	
	// constructor
	public SimilaritySupervisedMF(DataSet trainSet, DataSet testSet, int k)
	{
		// initialize matrices X,Y
		X = new Matrix();
        X.LoadDatasetFeatures(trainSet, false);
        X.LoadDatasetFeatures(testSet, true);
        Y = new Matrix();
        Y.LoadDatasetLabels(trainSet, false);
        Y.LoadDatasetLabels(testSet, true);  
       
        // initialize the dimensions 
        n_train = trainSet.instances.size();
        n_test = testSet.instances.size();
        m = X.getDimColumns();
        // initialize the reciprocal used for the gradients
        rec_n = 1.0/(double) (n_train+n_test);
        rec_m = 1.0/(double) m;
        d= k;
        
        // set the values of the labels as 1 and -1
        for(int i = 0; i < n_train+n_test; i++)
        {
        	if( Y.get(i, 0) != 1 )
        		Y.set(i, 0, -1);
        }
        
        // initialize latent matrices
        double eps = 0.01;
        
		U = new Matrix(n_train+n_test, d);
		U.RandomlyInitializeCells(-eps, eps);
	
		V = new Matrix(d, m);
		V.RandomlyInitializeCells(-eps, eps);
		
		kernel = new Kernel();
		kernel.type = KernelType.Linear;
		kernel.sig2 = gamma;
		
		System.out.println(V.getDimColumns() + " " + V.getDimRows());
		
        // initialize a smo solver
        smo = new NaiveLinearSmo(U, Y, n_train, C); 
        
        S = new Matrix(n_train+n_test, n_train+n_test);
        
        // compute the similarity matrix for the training instances
        for(int i = 0; i < n_train+n_test; i++)
        	for(int l = i; l < n_train+n_test; l++)
        	{
        		double sim = DTW.getInstance().CalculateDistance(X.getRow(i), X.getRow(l));
        		S.set(i, l, sim);
        		S.set(l, i, sim);
        	}
	}

	// reconstruct the value of the original matrix
	public double Reconstruct(int i, int j)
	{
		double dp_ij = 0;
		for(int k = 0; k < d; k++)
			dp_ij += U.get(i, k)*V.get(k,j);
		return dp_ij;
	}
	
	// update all the U and Psi_i cells once
	public void UpdateUV()
	{
		// update the reconstruction error
		for(int i = 0; i < n_train+n_test; i++)
		{
			for(int j = 0; j < m; j++)
			{
				double err_ij = X.get(i, j) - Reconstruct(i, j); 
				
				for(int k = 0; k < d; k++)
				{
					double grad_u_ik = -2*err_ij*V.get(k,j) + 2*lambdaU*U.get(i,k);
					U.set(i, k, U.get(i,k) - eta1*grad_u_ik);
					
					double grad_v_kj = -2*err_ij*U.get(i,k) + 2*lambdaV*V.get(k,j);
					V.set(k, j, V.get(k,j) - eta1*grad_v_kj);
					
					//System.out.println(grad_u_ik + " " + U.get(i, k)); 
				}
			}
		}
		
		
		// update the similarity and the classification loss
		for(int i = 0; i < n_train+n_test; i++)
		{
			for(int l = 0; l < n_train+n_test; l++)
			{ 
				double err_il = S.get(i, l) - U.RowDotProduct(i, l); 
				
				for(int k = 0; k < d; k++)
				{
					U.set(i, k, U.get(i,k) - eta2*(-2*err_il*U.get(l,k)) );					
				}
				
				if( i < n_train && l < n_train)
				{
					if( smo.alphas.get(i)*smo.alphas.get(l) == 0 )
						continue;
				
					double tmp = 0.5*smo.alphas.get(i)*smo.alphas.get(l)*Y.get(i)*Y.get(l);
				
					// avoid updating bias weights U(:,d-1)
					for(int k = 0; k < d; k++)
					{
						double grad = tmp * U.get(l, k);
						
						U.set(i, k, U.get(i,k) - eta3*grad);
					}
				}
			}
		}		
	}

	// the main optimization routing
	public double Optimize()
	{
		int iterationCount = 0; 
		double lastFR = Double.MAX_VALUE, 
				lastFCA = Double.MAX_VALUE;
		
		int numAlphasChanged = 0;
		
		// exit if the maximum epocs are reached, but nevertheless 
		// continue until all alphas dont violate KKT and need no more change
		while(true)
		{
			// optimize the U cells
			UpdateUV();
			 
			// iterate through smo and update alphas
			numAlphasChanged = smo.Optimize(1);
			
			// get the losses of the reconstruction, similarity and 
			// classification accuracy terms
			double fr = GetFRLoss();
			double fsim = GetFSimLoss();
			double fca = GetFCALoss();
			// classify the test instances
			double mcrTrain = GetMCR(false);
			// classify the test instances
			double mcrTest = GetMCR(true);
			// print the progress of the iteration
			PrintProgress(iterationCount, fr, fsim, fca, mcrTrain, mcrTest, numAlphasChanged);
			
			// check if the loss has overflown as a result
			// of divergences, then return 100% error
			if(Double.isNaN(fr) || Double.isNaN(fca))
				return 1.0;
			
			// stop if no improvement on the loss
			// after max iteration
		
			// stop after max iterations
			if( iterationCount > maxEpocs )
			{
				break; 
			} 
		
		} 
		
		smo.Optimize();
		
		
		
		return GetMCR(true);
	}

	
	
	// the loss due to reconstruction
	public double GetFRLoss()
	{
		// the loss from reconstruction 
		// SUM_i SUM_j (X_ij - SUM_k U_ik*V_kj )^2
		double recErrorsSum = 0;
		for(int i = 0; i < n_train+n_test; i++)
		{
			for(int j = 0; j < m; j++)
			{
				double err_ij = X.get(i, j) - Reconstruct(i, j);
				
				recErrorsSum += err_ij*err_ij;  
			}
		}
		
		// the loss from regularization of U 
		// SUM_i SUM_k U_ik^2
		double regUSum=0;
		for(int i = 0; i < n_train+n_test; i++)
			for(int k = 0; k < d; k++)
				regUSum += U.get(i, k)*U.get(i, k);
				
				
		double loss = recErrorsSum + lambdaU*regUSum;
		
		return loss;
	}
	
	// the loss due to reconstruction
	public double GetFSimLoss()
	{
		// the loss from reconstruction 
		// SUM_i SUM_j (X_ij - SUM_k U_ik*V_kj )^2
		double simErrorsSum = 0;
		for(int i = 0; i < n_train+n_test; i++)
		{
			for(int l = 0; l < n_train+n_test; l++)
			{
				double err_il = S.get(i, l) - kernel.K(U, i, l);
				
				simErrorsSum += err_il*err_il; 
			}
		}
		
		return simErrorsSum;
	}
	
	// the loss due to classification accuracy
	public double GetFCALoss()
	{
		double loss = 0;
		
		double sum = 0;
		for(int i = 0; i < n_train; i++)
			for(int l = 0; l < n_train; l++)
				sum += Y.get(i)*smo.alphas.get(i)*Y.get(l)*smo.alphas.get(l)*kernel.K(U, i, l); 
		
		loss += 0.5*sum;

		sum = 0;
		for(int i = 0; i < n_train; i++)
			sum += smo.alphas.get(i);
		
		loss -= sum;

		
		return loss;
	}
	
	/* get the miss classification rate on the training set
	 * parameter true -> mcr of test instances, false -> mcr on train
	 */
	public double GetMCR(boolean test) 
	{
		int errors = 0;
		
		int startIndex = 0;
		int endIndex = 0;
		
		if(test)
		{
			startIndex = n_train;
			endIndex = n_train + n_test;
		}
		else
		{
			startIndex = 0;
			endIndex = n_train;
		}
			
		
		for(int i = startIndex; i < endIndex; i++)
		{
			double f_i = smo.FunctionalMargin(i);
			
			double y_i = Y.get(i);
			
			if( y_i * f_i < 0)
				errors++;			
		}
		
		return (double)errors/(double)(endIndex-startIndex);
	}
	
	
	// print the progress made during the iteration
	public void PrintProgress(int numIter, double fr, double fsim, double fca, 
			double mcrTrain, double mcrTest, int numAlphasChanged)
	{
		Logging.println("Iter=" + numIter + ", FR=" + fr + ", FSim=" + fsim + ", FCA=" + fca 
						+ ", AlphasChanged="+ numAlphasChanged +", MCRTrain=" + mcrTrain + ", MCRTest=" + mcrTest, LogLevel.DEBUGGING_LOG);
	}
	
	
	
	
	
}

