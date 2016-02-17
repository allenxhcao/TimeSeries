package MatrixFactorization;

import java.util.ArrayList;
import java.util.List;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

import TimeSeries.TransformationFieldsGenerator;
import Utilities.BlindPrint;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import Classification.InvariantSVM;
import Classification.Kernel;
import Classification.Kernel.KernelType;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;

public class MaxMarginSupervisedDecompositionLibSVM 
{
	// the matrix X, Y, U, Psi_i of the method
	Matrix X, Y, U, V;
	// the coefficients are coeff_i = alpha_i*Y_i
	List<Double> coeff;
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
	
	
	// kernel class
	Kernel kernel;
	
	// the parameters of the svm
	svm_parameter svmParameters;
	svm_model svmModel;
	
	// reciprocal of instances count and features count
	private double rec_n, rec_m;
	
	// constructor
	public MaxMarginSupervisedDecompositionLibSVM(
			DataSet trainSet, DataSet testSet, int dimensions, 
			String svmKernel, double boxConstraing, double polynomialDegree, double radialGamma, 
			double betaSwitch, double lU, double lV, 
			double learnRateR, double learnRateCA, int maximumEpocs)
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
        	
        	//Logging.println(Y.get(i,0)+"", LogLevel.INFORMATIVE_LOG);
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
        
        
        
        // initialize the SVM parameters
        svmParameters = new svm_parameter();
        
        svmParameters.svm_type = svm_parameter.C_SVC;
        svmParameters.C = C;
        
        if( kernel.type == Kernel.KernelType.Polynomial )
        {
        	svmParameters.kernel_type = svm_parameter.POLY;
            svmParameters.degree = (int) p;
            svmParameters.gamma = 1.0;
            svmParameters.coef0 = 1;
        }
        else if( kernel.type == Kernel.KernelType.Gaussian )
        {
        	svmParameters.gamma = gamma;
        	svmParameters.kernel_type = svm_parameter.RBF;
        }

        svmParameters.eps = 0.000000000001;
        svmParameters.nr_weight = 0;
        
        svm.svm_set_print_string_function(new BlindPrint());
       
        // initialize the latent matrices
        double eps = 0.0001;
		U = new Matrix(n_train+n_test, d);
		U.RandomlyInitializeCells(-eps, eps);
		V = new Matrix(d, m); 
		V.RandomlyInitializeCells(-eps, eps);
		
		// set the coefficients to 0
		coeff = new ArrayList<Double>();
		for(int i=0; i<n_train; i++)
			coeff.add(0.0);
	}
	
	// solve the SVM and return the alphas
	public void SolveSVM()
	{
		// create svm problem data
        svm_problem data = new svm_problem();
        
        data.l = n_train;
        data.x = new svm_node[n_train][d];
        data.y = new double[n_train];
                
        // iterate through all the instances
        for(int i = 0; i < n_train; i++)
        {
            for(int k = 0; k < d; k++)
            {
                data.x[i][k] = new svm_node(); 
                data.x[i][k].index = k;
                data.x[i][k].value = U.get(i, k); 
            } 
            
            data.y[i] = (int)Math.rint(Y.get(i, 0));
        }
        
        svmModel = svm.svm_train(data, svmParameters);
   
        /*
        System.out.println("model.nr_class="+ svmModel.nr_class);
        System.out.println("model.model.label="+ svmModel.label.length);
        System.out.println("model.nSV.length="+ svmModel.nSV.length);
        System.out.println("numSv"+ svmModel.nSV[0]+svmModel.nSV[1]);
        */
        /*
        int numSv = svmModel.nSV[0]+svmModel.nSV[1];
        System.out.println("SV% ="+ (double)numSv/(double)n_train);
        
        System.out.println("model.sv_coef.length="+ svmModel.sv_coef.length);
        System.out.println("model.sv_coef[0].length="+ svmModel.sv_coef[0].length);
        System.out.println("svmModel.rho.length: "+ svmModel.rho.length);
        System.out.println("p: "+ p);
        */
        
        // let us read the coefficients of the learned model
        
		// reset the coefficients to 0
		for(int i=0; i<n_train; i++)
			coeff.set(i, 0.0); 
		
        int svIndexOffset = 0;
         
        for( int c = 0; c < svmModel.nr_class; c++)
        {
        	double label = (double)svmModel.label[c];
        	
        	//System.out.println("Label: " + label);
        	
        	int noSVPerClass = svmModel.nSV[c];
        	
        	for(int svIndex = svIndexOffset; svIndex < svIndexOffset+noSVPerClass; svIndex++)
        	{
        		//System.out.println("svIndex:" + svIndex);
        		// create a double vector row format from the 
        		// support vector to use it for searching the index
        		List<Double> sv = new ArrayList<Double>();
                for(int k = 0; k < d; k++)
                    sv.add(svmModel.SV[svIndex][k].value);
                                
                // get the index of the support vector in the dataset
                int index = U.SearchRow(sv);
                
                		
                if(index >= 0)
                {
                	//Logging.println("SV label:" + svLabel + ", Instance label:" + Y.get(index,0), LogLevel.ERROR_LOG);
                	
                	if( label == Y.get(index,0))
                		coeff.set(index, Math.abs(svmModel.sv_coef[0][svIndex])*label);
                	else
                		Logging.println("SV label:" + label + "!= Instance label:" + Y.get(index,0), LogLevel.ERROR_LOG);
                }
                
                //System.out.println( "sv:" + index + ", coeff: " + coeff.get(index) + ", y: " + Y.get(index,0));
                
        	}
        	
        	svIndexOffset += noSVPerClass;
        }
        
        /*System.out.println("----------------");
		
        for(int i=0; i<n_train; i++)
        {
    		//System.out.println("i=" + i + ", alpha_i*y_i=" + coeff.get(i) + ", y_i= " + Y.get(i,0) ) ;
        }
        */
		
	}
	
	// update the U cells
	public void SolveUV()
	{
		boolean converged=false;
		int iterationCount = 0;
		int maxIterationCount = maxEpocs/5;
		
		// iterate until loss is minimized w.r.t U,Psi_i
		while(!converged)
		{
			// update the cells of U using reconstruction loss terms
			// and the regularization loss term
			for(int i = 0; i < n_train+n_test; i++)
			{
				for(int j = 0; j < m; j++)
				{ 
					// the dot product \SUM_k U_ik V_kj
					double dp = 0;
					for(int k = 0; k < d; k++)
						dp += U.get(i,k)*V.get(k,j);
					
					// the error (X_ij - \SUM U_ik V_kj)^2
					double e_ij = X.get(i, j) - dp;
					
					for(int k = 0; k < d; k++)
					{
						double grad_ui = -2*beta*e_ij*V.get(k,j) + 2*lambdaU*rec_m*U.get(i,k);
						U.set(i, k, U.get(i,k) - etaR * grad_ui);
						
						double grad_v_kj = -2*beta*e_ij*U.get(i,k) + 2*lambdaV*rec_n*V.get(k,j);
						V.set(k, j, V.get(k,j) - etaR * grad_v_kj);
					}
				}
			}
			
			// optimize U_ik, per each classification loss subterm
			// FCA_il, in neg direct of gradient dFCA_il / dU_ik
			for(int i = 0; i < n_train; i++)
			{
				for(int l = 0; l < n_train; l++)
				{
			
					// prepare the section of the multiplication 
					// that doesn't depend on k, so we dont have to
					// compute per each k
					
					// For polynomial the gradient (p*(<Ui,Ul>+1)^(p-1))*U_lk 
					// can be split to tmp=p*(<Ui,Ul>+1)^(p-1) before
					// and then tmp*U_lk in a loop per k
					if( kernel.type == Kernel.KernelType.Polynomial )
					{
						double tmp =(1-beta)*0.5*coeff.get(i)*coeff.get(l) *
								p*Math.pow(U.RowDotProduct(i, l) + 1, p-1);
						
						for(int k = 0; k < d; k++)
						{
							double grad = tmp * U.get(l, k);
							U.set(i, k, U.get(i,k) - etaCA*grad);
						}
					}
					// For gaussian the gradient -gamma*2*(U_ik-U_lk)*e^(-gamma*||Ui,Ul||^2) 
					// can be split to tmp=-gamma*2*e^(-gamma*||Ui,Ul||^2) before
					// and then tmp*(U_ik-U_lk) in a loop per k
					else if( kernel.type == Kernel.KernelType.Gaussian )
					{
						double tmp = (1-beta)*0.5*coeff.get(i)*coeff.get(l) *
								-1*gamma * 2 * Math.exp(- gamma * U.RowEuclideanDistance(i, l));
						
						for(int k = 0; k < d; k++)
						{
							double grad = tmp * (U.get(i,k)-U.get(l, k));
							U.set(i, k, U.get(i,k) - etaCA*grad);
						}
					}
					
				} // end for instance l
			} // end for instance i
			
			
			
			// update the iteration count
			iterationCount++;
			
			if(iterationCount < maxIterationCount)
				converged = true;
		}
		
		
	}
	

	// the main optimization routing
	public double Optimize()
	{
		int iterationCount = 0;
		int debugIterFrequency = 10;
		double lastFR = Double.MAX_VALUE, lastCA = Double.MAX_VALUE;
		
		while(iterationCount < maxEpocs)
		{
			// optimize the FR w.r.t U and Psi_i
			SolveUV();
			
			// solve the svm and update the alphas
			SolveSVM();
			
			if( iterationCount % debugIterFrequency == 0 )
			{
				// get the losses of the reconstruction and 
				// classification accuracy terms
				double fr = GetFRLoss(), fca = GetFCALoss();
				// classify the test instances
				double mcrTrain = GetMCRTrainSet();
				// classify the test instances
				double mcrTest = GetMCRTestSet();
				// print the progress of the iteration
				PrintProgress(iterationCount, fr, fca, mcrTrain, mcrTest);
				
				// check if the loss has overflown as a result
				// of divergences, then return 100% error
				if(Double.isNaN(fr) || Double.isNaN(fca))
					return 1.0;
			}
			
			// increase the iteration count			
			iterationCount++;
			// update the stopping criterion
		}
		
		return GetMCRTestSet();
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
				double dp = 0;
				for(int k = 0; k < d; k++)
					dp += U.get(i,k)*V.get(k,j); 
				
				recErrorsSum += Math.pow(X.get(i, j) - dp, 2);
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
			{
				sum += coeff.get(i)*coeff.get(l)* kernel.K(U, i, l); 
			}
		
		loss += 0.5*sum;

		sum = 0;
		for(int i = 0; i < n_train; i++)
			sum += coeff.get(i)/Y.get(i,0);
		
		loss -= sum;

		
		return (1-beta)*loss;
	}
	
	// get the miss classification rate on the training set
	public double GetMCRTrainSet() 
	{
		int errors = 0;
		
		for(int i = 0; i < n_train; i++)
		{
			double sum = 0;
			for(int l = 0; l < n_train; l++)
				sum += coeff.get(l)*Math.pow(U.RowDotProduct(i, l) + 1, p);
			sum -= svmModel.rho[0];
			
			double y_i = Y.get(i, 0);
			
			if( y_i * sum < 0)
				errors++;			
		}
		
		return (double)errors/(double)n_train;
	}
	
	// get the miss classification rate on the test set
	public double GetMCRTestSet() 
	{
		int errors = 0;
		
		for(int i = n_train; i < n_train + n_test; i++)
		{
			double sum = 0;
			for(int l = 0; l < n_train; l++)
				sum += coeff.get(l)*Math.pow(U.RowDotProduct(i, l) + 1, p);
			sum -= svmModel.rho[0];

			double y_i = Y.get(i, 0);
			
			//System.out.println("sum="+sum + ", label: " + y_i);
			
			if( y_i * sum < 0)
				errors++;
		}
		
		return (double)errors/(double)n_test; 
	}
	
	// print the progress made during the iteration
	public void PrintProgress(int numIter, double fr, double fca, 
			double mcrTrain, double mcrTest)
	{
		Logging.println("Iter=" + numIter + ", FR=" + fr + ", FCA=" + fca 
						+ ", MCRTrain=" + mcrTrain + ", MCRTest=" + mcrTest, LogLevel.DEBUGGING_LOG);
	}
	
	
	
	
	
}


