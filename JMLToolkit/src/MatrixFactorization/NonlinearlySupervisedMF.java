package MatrixFactorization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import weka.experiment.LearningRateResultProducer;


import TimeSeries.TransformationFieldsGenerator;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import Classification.InvariantSVM;
import Classification.Kernel;
import Classification.Kernel.KernelType;
import Classification.NaiveSmo;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;

public class NonlinearlySupervisedMF 
{
	// the matrix X, Y, U, Psi_i of the method
	Matrix X, Y, U, V;
	// number of instances of train and test sets
	private int n_train, n_test;
	// number of features
	private int m;
	// the latent dimensionality parameter
	public int d;
	// the box constraint parameter of SVM
	public double C;
	// the impact switch parameter
	public double beta;
	// the regularization paramters of U and Psi_i
	public double lambdaU, lambdaV;
	// the learning rate 
	public double etaR, etaCA ;
	// the degree of the polynomial kernel
	// kernel is of type K(Ui,Uj) = (<Ui,Uj>+b)^p
	public double p;
	// the 1/variance of gaussian kernel denoted gamma
	public double gamma;
	// the maximum number of epocs
	public int maxEpocs;
	
	// the last loss
	public double lastLoss;
	
	Random rand = new Random();
	
	boolean preDecomposeUnsupervised = false;
	
	
	// kernel class
	Kernel kernel;
	
	// a smo solver
	NaiveSmo smo;
	
	// reciprocal of instances count and features count
	private double rec_n, rec_m;
	
	// constructor
	public NonlinearlySupervisedMF(
			DataSet trainSet, DataSet testSet, int dimensions, 
			String svmKernel, double boxConstraing, double polynomialDegree, double radialGamma, 
			double betaSwitch, double lU, double lV, 
			double learnRateR, double learnRateCA, int maximumEpocs)
	{
		// first of all normalize the datasets
		
		//trainSet.NormalizeDatasetFeatures();
		//testSet.NormalizeDatasetFeatures();
		
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
        
        rec_n = 1.0/(double)(n_train+n_test);
        rec_m = 1.0/(double)m;
        
        
        // initialize hyper parameters
        d = dimensions;
        C = boxConstraing;
        p = polynomialDegree;
        gamma = radialGamma;
        beta = betaSwitch;
        lambdaU = lU;
        lambdaV = lV;
        etaR = learnRateR;
        etaCA = learnRateCA;
        maxEpocs = maximumEpocs;
        
        
        // set the values of the labels as 1 and -1
        for(int i = 0; i < n_train+n_test; i++)
        {
        	if( Y.get(i, 0) != 1 )
        		Y.set(i, 0, -1);
        }
        
        if(!preDecomposeUnsupervised)
        {
	        // initialize latent matrices
	        double eps = 1;  
			U = new Matrix(n_train+n_test, d);
			U.RandomlyInitializeCells(-eps, eps);
				
			// initialize Psi_i to average per column
			V = new Matrix(d, m); 
			V.RandomlyInitializeCells(-eps, eps);
			        
			// set bias weights to 1
			// U <-- [U bu 1] , Psi_i <-- [ Psi_i ; 1 ; bv]
			for(int i=0; i < n_train+n_test; i++)
				U.set(i, d-1, 1.0);
			for(int j=0; j < m; j++)
				V.set(d-2, j, 1.0);
        }
        else 
        {
        	PreDecomposeUnsupervised();
        }
        
        kernel = new Kernel();
        
        if(svmKernel.compareTo("polynomial") == 0)
        {
        	kernel.type = Kernel.KernelType.Polynomial;
        	kernel.degree = (int)p; 
        }
        else if(svmKernel.compareTo("gaussian") == 0)
        {
        	kernel.type = Kernel.KernelType.Gaussian;
        	kernel.sig2 = gamma;  
        }
        
        // initialize a smo solver
        smo = new NaiveSmo(U, Y, n_train, C, kernel); 
        
        
	}
	
	public void PreDecomposeUnsupervised()
	{
		MatrixFactorization mf = new MatrixFactorization( d );
	    mf.lambdaU = lambdaU; 
		mf.lambdaV = lambdaV;
	    mf.learningRate = etaR;
		mf.alpha = beta;
		mf.maxEpocs = maxEpocs/2;
		
		// just some initializations
		mf.numTotalInstances = n_train+n_test; 
		mf.numFeatures =m;
		mf.rec_features = 1.0 / (double) m;
		mf.rec_totalInstances = 1.0 / (double) mf.numTotalInstances;
		
		mf.Decompose(X);
		
		U = new Matrix( mf.getU() );
		V = new Matrix( mf.getV() );
		
	}
	
	
	
	// update all the U and Psi_i cells once
	public void UpdateUV()
	{
		// update the cells of U using reconstruction loss terms
		// and the regularization loss term
		for(int i = 0; i < n_train+n_test; i++)
		//for(int index1 = 0; index1 < n_train+n_test; index1++)
		//for(int counter = 0; counter < (n_train+n_test)*m; counter++)
		{
			//int i = randomRowIndices.get(index1);
			
			//int i = rand.nextInt(n_train+n_test); 
			//int j = rand.nextInt(m); 
			
			for(int j = 0; j < m; j++)
			//for(int index2 = 0; index2 < m; index2++)
			{ 
				//int j = randomRowIndices.get(index2);
				
				// skip the cell if it is empty
				if(X.get(i,j) == GlobalValues.MISSING_VALUE)
					continue;
				
				// the dot product \SUM_k U_ik V_kj
				double dp = 0;
				for(int k = 0; k < d; k++)
					dp += U.get(i,k)*V.get(k,j);
				
				// the error (X_ij - \SUM U_ik V_kj)^2
				double e_ij = X.get(i, j) - dp;
				
				for(int k = 0; k < d; k++)
				{
					// skip updating the bv bias weight column of U
					if( k != d-1 )
					{
						double grad_u_ik = -2*beta*e_ij*V.get(k,j);
						
						// dont regularize bu
						if(k != d-2) 
							grad_u_ik += 2*lambdaU*rec_m*U.get(i,k);
						
						
						U.set(i, k, U.get(i,k) - etaR * grad_u_ik);
					}

					if( k != d-2)
					{
						double grad_v_kj = -2*beta*e_ij*U.get(i,k);
						
						// dont regularize bv
						if( k != d-1) 
							grad_v_kj += 2*lambdaV*rec_n*V.get(k,j);
						
						V.set(k, j, V.get(k,j) - etaR * grad_v_kj);
					}
					//System.out.println("grad_u="+grad_u_ik+", grad_v="+grad_v_kj);
					
				}
			}
		}
		
		// check if classification is supervised, i.e. beta not 1
		if(beta < 1)
		{
			// optimize U_ik, per each classification loss subterm
			// FCA_il, in neg direct of gradient dFCA_il / dU_ik
			//for(int index1 = 0; index1 < n_train; index1++)
			for(int i = 0; i < n_train; i++)
			{
				//int i = rand.nextInt(n_train);
				
				//for(int index2 = 0; index2 < n_train; index2++)
				for(int l = 0; l < n_train; l++)
				{
					//int l = rand.nextInt(n_train);
					
					// both alphas need to be non zero otherwise 
					// no need to proceed as the product will be zero
					if( smo.alphas.get(i)*smo.alphas.get(l) == 0 )
						continue;
				
					// prepare the section of the multiplication 
					// that doesn't depend on k, so we dont have to
					// compute per each k
					
					// For polynomial the gradient (p*(<Ui,Ul>+1)^(p-1))*U_lk 
					// can be split to tmp=p*(<Ui,Ul>+1)^(p-1) before
					// and then tmp*U_lk in a loop per k
					if( kernel.type == Kernel.KernelType.Polynomial )
					{
						
						double tmp =(1-beta)*0.5* Y.get(i)*smo.alphas.get(i)*Y.get(l)*smo.alphas.get(l) *
								p*Math.pow(U.RowDotProduct(i, l) + 1, p-1);
						
						tmp *= -1;
						
						
						// avoid updating bias weights U(:,d-1)
						for(int k = 0; k < d-1; k++) 
						{
							// regularize except for the case of biases
						/*	if( k != d-2)
							{
								U.set(i, k, U.get(i,k) - etaCA*(tmp * U.get(l, k) + lambdaU*rec_n*U.get(i, k) ));
								U.set(l, k, U.get(l,k) - etaCA*(tmp * U.get(i, k) + lambdaU*rec_n*U.get(l, k)));
							}
							else
							{
							*/
								U.set(i, k, U.get(i,k) - etaCA*tmp * U.get(l, k));
								U.set(l, k, U.get(l,k) - etaCA*tmp * U.get(i, k));
							//}
						}
						
						
					}
					// For gaussian the gradient -gamma*2*(U_ik-U_lk)*e^(-gamma*||Ui,Ul||^2) 
					// can be split to tmp=-gamma*2*e^(-gamma*||Ui,Ul||^2) before
					// and then tmp*(U_ik-U_lk) in a loop per k
					else if( kernel.type == Kernel.KernelType.Gaussian )
					{
						double tmp = (1-beta)*0.5*Y.get(i)*smo.alphas.get(i)*Y.get(l)*smo.alphas.get(l) *
								-1*gamma * 2 * Math.exp(- gamma * U.RowEuclideanDistance(i, l));
						
						for(int k = 0; k < d-1; k++)
						{					
							double grad = tmp * (U.get(i,k)-U.get(l, k));
							U.set(i, k, U.get(i,k) - etaCA*grad);
							grad = tmp * (U.get(i,k)-U.get(l, k)) * -1;
							U.set(l, k, U.get(l,k) - etaCA*grad);
						}
					}
					
				} // end for instance l
			} // end for instance i
		
		}
		
	
		
	}
	

	// the main optimization routing
	public double Optimize()
	{
		int iterationCount = 0; 
		int debugIterFrequency = 1;
		double lastFR = Double.MAX_VALUE, 
				lastFCA = Double.MAX_VALUE;
		
		int numAlphasChanged = 0;
		
		// exit if the maximum epocs are reached, but nevertheless 
		// continue until all alphas dont violate KKT and need no more change
		while(true)
		{
			
			// optimize the FR w.r.t U and Psi_i
			UpdateUV();
			 
			if( iterationCount % debugIterFrequency == 0 )
			{
				// iterate through smo and update alphas
				// for just one iteration
				if( beta < 1)
					numAlphasChanged = smo.Optimize(2); 
				
				// get the losses of the reconstruction and 
				// classification accuracy terms
				double fr = GetFRLoss(), fca = GetFCALoss();
				// classify the test instances
				double mcrTrain = GetMCR(false);
				// classify the test instances
				double mcrTest = GetMCR(true);
				
				// print the progress of the iteration
				//PrintProgress(iterationCount, fr, fca, mcrTrain, mcrTest, numAlphasChanged);
				
				// check if the loss has overflown as a result
				// of divergences, then return 100% error
				if(Double.isNaN(fr) || Double.isNaN(fca))
					return 1.0;
				
				// stop if no improvement on the loss
				// after max iteration
				
				if( fr+fca >= lastFR+lastFCA)
				{
					// stop after max iterations
					if( iterationCount > maxEpocs )
					{
						break; 
					} 
				}
				// hard stop at max epocs
				//else if( iterationCount > maxEpocs )
				//{
				//	break; 
				//} 
				else
				{
					lastFR = fr;
					lastFCA = fca;
					
					lastLoss = lastFR+lastFCA;
				}
				
				
			}
			
			// increase the iteration count			
			iterationCount++;
			// update the stopping criterion
			//System.out.println("...");
		} 
		
		smo.Optimize();
		
		// debugging 
		//PrintLatent();
		
		
		
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
				if( X.get(i, j) != GlobalValues.MISSING_VALUE )
				{
					double dp = 0;
					for(int k = 0; k < d; k++)
						dp += U.get(i,k)*V.get(k,j);
					
					recErrorsSum += Math.pow(X.get(i, j) - dp, 2);
				}
			}
		}
		
		// the loss from regularization of U 
		// SUM_i SUM_k U_ik^2
		double regUSum=0;
		for(int i = 0; i < n_train+n_test; i++)
			for(int k = 0; k < d; k++)
				regUSum += U.get(i, k)*U.get(i, k);
				
		// the loss from regularization of Psi_i 
		// SUM_k SUM_j V_kj^2
		double regVSum=0;
		for(int k = 0; k < d; k++)
			for(int j = 0; j < m; j++)
				regVSum += V.get(k, j)*V.get(k, j);
				
		double loss = beta*recErrorsSum + lambdaU*regUSum 
				+ lambdaV*regVSum;
		
		return loss;
	}
	
	// the loss due to classification accuracy
	public double GetFCALoss()
	{
		double loss = 0;
		
		double sum = 0;
		for(int i = 0; i < n_train; i++)
			for(int l = 0; l < n_train; l++)
				sum += Y.get(i)*smo.alphas.get(i)*Y.get(l)*smo.alphas.get(l)* kernel.K(U, i, l); 
		
		loss += 0.5*sum;

		sum = 0;
		for(int i = 0; i < n_train; i++)
			sum += smo.alphas.get(i);
		
		loss -= sum;

		
		return -(1-beta)*loss;
	}
	
	// the loss due to classification accuracy
	public double GetFCALossTest()
	{
		double loss = 0;
		
		double sum = 0;
		for(int i = n_train; i < n_train+n_test; i++)
			for(int l = 0; l < n_train; l++)
				sum += Y.get(i)*smo.alphas.get(i)*Y.get(l)*smo.alphas.get(l)* kernel.K(U, i, l); 
		
		loss += 0.5*sum;

		sum = 0;
		for(int i = 0; i < n_train; i++)
			sum += smo.alphas.get(i);
		
		loss -= sum;

		
		return -(1-beta)*loss;
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
	public void PrintProgress(int numIter, double fr, double fca,
			double mcrTrain, double mcrTest, int numAlphasChanged)
	{
		Logging.println("Iter=" + numIter + ", FR=" + fr + ", FCA=" + fca
						+ ", AlphasChanged="+ numAlphasChanged +", MCRTrain=" + mcrTrain + ", MCRTest=" + mcrTest, LogLevel.DEBUGGING_LOG);
	}
	
	
	public void PrintLatent()
	{
		for(int i = 0; i < n_train; i++)
		{
			System.out.println(U.get(i,0)+" "+U.get(i,1)+";");
		}
	}
	
	
	
}

