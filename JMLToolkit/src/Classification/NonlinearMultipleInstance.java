package Classification;

import DataStructures.Tripple;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;
import MatrixFactorization.MatrixFactorizationModel;
import MatrixFactorization.MatrixUtilities;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.rmi.CORBA.Util;

public class NonlinearMultipleInstance 
{
	// the learn rate coefficient for each loss term
	public double etaX, etaY, etaL;

	// the coefficients of each loss term
	public double coeffX, coeffY, coeffL;

	// the regularization parameters
	public double lambdaU, lambdaV, lambdaAlpha;
	
	// training+testing predictors
	public Matrix X;
	// training labels
	public Matrix Y;

	// the number of latent dimensions
	public int maxEpochs;
	
	// the weights of the reconstruction linear regression models
	public Matrix U;
	
	// the weights of the reconstruction linear regression models of predictors
	public Matrix V;
	double [] biasV;
	
	// the weights of the kernel logistic regression
	double [] alphas;
	double biasAlpha;
	
	
	// the gamma parameter of the RBF kernel of the Kernel Logistic Regression
	public double gamma;
	
	// the block size, i.e. the number of instances in each bag, (each bag has same number of instances in this model)
	// i.e. if block size is 4, then instances 0,1,2,3 will have same label, instances 4,5,6,7 also same label and so on ...
	public int blockSize;
	
	// number of latent dimensions
	public int D;
	
	/*
     * A random generator
     */
	Random rand = new Random();

    // The list of observed items in the XTraining and XTesting
    protected List<Tripple> XObserved;

	protected List<Integer> YObserved;
    
    // number of training instances 
    public int numTrainInstances, 
    			numTotalInstances, 
    			numFeatures;
        
	/*
	 * Constructor for the regularized SVD 
	 */
	public NonlinearMultipleInstance(int factorsDim) 
	{
            etaX = 0.001;
            etaY = 0.001;
            etaL = 0.001;

            lambdaU = 0.001;  
            lambdaV = 0.001;  
            lambdaAlpha = 0.001;  

            XObserved = null;  
            YObserved = null;
            
            D = factorsDim;
            
            coeffX = 1.0;
            coeffY = 1.0;
            coeffL = 1.0;
            
            blockSize = 4;
  }
	
	// get the total X loss of the reconstruction, the loss will be called
        // for getting the loss of a cell, so it must be initialized first
	public double GetPredictorsMSE()
    {
        double XTrainingLoss = 0;
        int numObservedCells = 0;
        
        for(int i = 0; i < numTotalInstances; i++)
            for(int j = 0; j < numFeatures; j++)
                if( X.get(i, j) != GlobalValues.MISSING_VALUE )
                {
                	double err =  X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasV[j];
                    XTrainingLoss += err*err;
                    numObservedCells++;
                }
                    
        return XTrainingLoss/(double)numObservedCells; 
    } 
        
    // initialize the matrixes before training
    public void Initialize(Matrix x, Matrix y)
    {
        // store original matrix
        
    	// numFeatures = numFeatures * blockSize;
    	// numTotalInstances = numTotalInstances / blockSize;
    	
    	X = new Matrix(numTotalInstances/blockSize, numFeatures*blockSize);
    	Y = new Matrix(numTotalInstances/blockSize, 1); 
    	
    	for(int i = 0; i < numTotalInstances/blockSize; i++)
    	{
    		for(int blockIndex = 0; blockIndex < blockSize; blockIndex++)
    		{
    			for(int j = 0; j < numFeatures; j++)
    			{
    				int oldInsId = i*blockSize + blockIndex;
    				int newInsId = i;
    				int oldFeatId = j;
    				int newFeatId = numFeatures*blockIndex + j;
    						
    				//System.out.println(oldInsId+","+oldFeatId + " -> " + newInsId +","+ newFeatId );
    				
    				X.set(newInsId, newFeatId,  x.get( oldInsId, oldFeatId)); 
    			}
    		}
    		
    		Y.set(i, 0, y.get(i*blockSize));
    	}
    	
    	numTrainInstances /= blockSize; 
    	numTotalInstances /= blockSize; 
    	numFeatures *= blockSize; 
        
        // initialize latent representation of X into latent space represented by U
        U = new Matrix(numTotalInstances, D);
        U.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
        
        // initialize a transposed Psi_i
        V = new Matrix(D, numFeatures);
        V.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);

        biasV = new double[numFeatures];
        for(int j = 0; j < numFeatures; j++)
        	biasV[j] = X.GetColumnMean(j);
        
        // initialize the alphas
		// initialize the alphas
		alphas = new double[numTrainInstances];
		
		// initialize the weights between -epsilon +epsilon
		for(int i = 0; i < numTrainInstances; i++)
			alphas[i] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON;
		
		for(int i = 0; i < numTrainInstances; i++)
			biasAlpha += Y.get(i);
		biasAlpha /= numTrainInstances;
				
		//biasAlpha = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON;
        
        
        // setup the prediction and loss 
        XObserved = new ArrayList<Tripple>();
        // record the observed values
        for(int i=0; i < X.getDimRows(); i++)
            for(int j=0; j < X.getDimColumns(); j++)
                if( X.get(i, j) != GlobalValues.MISSING_VALUE )
                    XObserved.add(new Tripple(i, j)); 
        
        Collections.shuffle(XObserved);
        
        YObserved = new ArrayList<Integer>();
        // record the observed values
        for(int i = 0; i < numTrainInstances; i++)
                if( Y.get(i) != GlobalValues.MISSING_VALUE )
                    YObserved.add(i); 
        
        Collections.shuffle(YObserved);
        
        Logging.println("numTrainInstances="+numTrainInstances + 
        				", numTotalInstances="+numTotalInstances +
        				", numFeatures="+numFeatures + 
        				", latentDim=" + D + 
        				", biasAlpha=" + biasAlpha, LogLevel.DEBUGGING_LOG);  
        
    }

	/*
	 * Train and generate the decomposition 
	 * */

    public double Decompose(Matrix X, Matrix Y) 
    {
    	// initialize the model
    	Initialize(X, Y);
    	
    	for(int epoch = 0; epoch < maxEpochs; epoch++)
		{
    		// set a frequency tick to update the target loss multiple times
    		// compared to the predictors loss
    		int frequencyTick = XObserved.size() / 100;
    		
		    for( int predictorCellIndex = 0; predictorCellIndex < XObserved.size(); predictorCellIndex++ )
            {
		    	UpdatePredictorsLoss(XObserved.get(predictorCellIndex).row, XObserved.get(predictorCellIndex).col);
		    	
		    	if( predictorCellIndex % frequencyTick == 0 )
		    	{
		    		for( int targetCellIndex = 0; targetCellIndex < YObserved.size(); targetCellIndex++)
					{
		        		UpdateTargetLossAlphas( YObserved.get(targetCellIndex) );
		        		UpdateTargetLossU( YObserved.get(targetCellIndex) ); 
					}
		    	}
            }
         
		    double predictorsMSE = GetPredictorsMSE();
        	double trainLogLoss = GetTrainLogLoss();
        	double testLogLoss = GetTestLogLoss();
        	double testErrorRate = GetTestErrorRate();
        	
            
            Logging.println("Epoch=" + epoch + 
            					", predictorsMSE="+predictorsMSE + 
            					", trainLogLoss=" + trainLogLoss + 
            					", testLogLoss=" + testLogLoss + 
            					", testErrorRate=" + testErrorRate, LogLevel.DEBUGGING_LOG);
            
		}

    	// in the end learn the target loss over the complete latent representation 
    	for(int epoch = 0; epoch < maxEpochs; epoch++)
    		for( int targetCellIndex = 0; targetCellIndex < YObserved.size(); targetCellIndex++)
    			UpdateTargetLossAlphas( YObserved.get(targetCellIndex) ); 
	
	    double predictorsMSE = GetPredictorsMSE();
    	double trainLogLoss = GetTrainLogLoss();
    	double testLogLoss = GetTestLogLoss();
    	double testErrorRate = GetTestErrorRate();
    	
        
        Logging.println("Final " + 
        					", predictorsMSE="+predictorsMSE + 
        					", trainLogLoss=" + trainLogLoss + 
        					", testLogLoss=" + testLogLoss + 
        					", testErrorRate=" + testErrorRate, LogLevel.DEBUGGING_LOG);
    	
    	
    	// print the probabilities of the test instances 
    	for(int i = numTrainInstances; i < numTotalInstances; i++)
    		System.out.println( (i+1) + "," + Probability(U.getRow(i)) );
    	
		return 0;
	}

    // update the latent representation to approximate X_{i,j}
    public void UpdatePredictorsLoss(int i, int j)
    {
    	double X_ij, error_ij, u_ik, v_kj, grad_u_ik, grad_v_kj;
    	
    	X_ij = X.get(i, j);
    	
    	if( X_ij == GlobalValues.MISSING_VALUE ) return;
    	
        error_ij = X_ij - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasV[j]; 
        
        for(int k = 0; k < D; k++)
        {
        	u_ik =  U.get(i, k); 
        	v_kj = V.get(k, j);
            
    		grad_u_ik = -2*coeffX*error_ij*v_kj + 2*lambdaU*u_ik;
        	U.set(i, k, u_ik - etaX * grad_u_ik);
        	
        	grad_v_kj = -2*coeffX*error_ij*u_ik + 2*lambdaV* v_kj;
            V.set(k, j, v_kj - etaX * grad_v_kj);
    	}
        
        biasV[j] = biasV[j] + etaX *coeffX * error_ij;
    }
    
    // update the latent nonlinear weights alpha to approximate target label Y_i
    public void UpdateTargetLossAlphas(int i)
    {
    	double err_i, Y_i, k_il, grad_l;
    	
		Y_i = Y.get(i);
		
		if( Y_i == GlobalValues.MISSING_VALUE ) return;
		
		err_i = Y_i - Probability(U.getRow(i)); 
		
		for(int l = 0; l < numTrainInstances; l++)
		{
			// compute the gradient
			k_il = ComputeRBFKernel(U.getRow(i), U.getRow(l));
			
			grad_l = -coeffY*err_i*k_il + 2*lambdaAlpha*alphas[i]*k_il; 
			
			// update the alpha in a gradient descent fashion 
			alphas[l] = alphas[l] - etaY*grad_l;
		}
		
		// update the bias parameter
		biasAlpha = biasAlpha + etaY*coeffY*err_i; 
		
    }
    
    // update the latent nonlinear weights alpha to approximate target label Y_i
    public void UpdateTargetLossU(int i)
    {
    	double err_i, Y_i, k_il, grad_lk, grad_k_il;
    	double euclideanDistance_il = 0, dist_il = 0;
    	
    	Y_i = Y.get(i);
		
		if( Y_i == GlobalValues.MISSING_VALUE ) return;
		
		err_i = Y_i - Probability(U.getRow(i)); 
		
		for(int l = 0; l < numTrainInstances; l++)
		{
			euclideanDistance_il = 0;
			for(int k = 0; k < D; k++)
			{
				dist_il = U.get(i, k) - U.get(l, k);
				euclideanDistance_il += dist_il*dist_il; 
			}
			
			k_il = Math.exp( - gamma * euclideanDistance_il );
			
			// update every U_lk
			for(int k = 0; k < D; k++)
			{
				grad_k_il = k_il * gamma * 2 *(U.get(i, k) - U.get(l, k));
				
				grad_lk = -coeffY*err_i*alphas[l]*grad_k_il + 2*lambdaAlpha*alphas[i]*grad_k_il*alphas[l];// + 2*lambdaU*U.get(l, k);  
				
				U.set(l, k, U.get(l, k) - etaY*grad_lk );
			}
		}
		
    }
    
    // update for the laplacian loss of the bag items i&p, i.e: set vartheta_i = Y_hat_p
    
    public void UpdateLaplacianLossAlphas(int i, int p)
    {
    	double k_il, k_pl, grad_ipl;
		
		double Y_hat_i = Probability(U.getRow(i));
		double Y_hat_p = Probability(U.getRow(p));
		
		for(int l = 0; l < numTrainInstances; l++)
		{
			// compute the gradient
			k_il = ComputeRBFKernel(U.getRow(i), U.getRow(l));
			k_pl = ComputeRBFKernel(U.getRow(p), U.getRow(l));
			
			grad_ipl = coeffL*2*(Y_hat_i-Y_hat_p)*( Y_hat_i*(1-Y_hat_i)*k_il - Y_hat_p*(1-Y_hat_p)*k_pl ); 
			
			// update the alpha in a gradient descent fashion 
			alphas[l] = alphas[l] - etaL*grad_ipl;
		}
		
		// update the bias parameter
		biasAlpha = biasAlpha - etaL*coeffL*2*(Y_hat_i-Y_hat_p)*( Y_hat_i*(1-Y_hat_i) - Y_hat_p*(1-Y_hat_p) ); 
		
    }
    
    
	// compute the probability of an arbitrary predictors feature vector
	public double Probability(double [] predictorFeatures)
	{
		double val = biasAlpha;
		
		for(int i = 0; i < numTrainInstances; i++)
			val += ComputeRBFKernel(U.getRow(i), predictorFeatures) * alphas[i];
		
		return Utilities.Sigmoid.Calculate(val);
	}
	
	// compute rbf kernel of U_i and U_l 
	public double ComputeRBFKernel(double [] U_i, double [] U_l)
	{
		double k_il = 0;
		
		double euclideanDistance = 0, dist_il = 0;
		for(int k = 0; k < U_i.length; k++)
		{
			dist_il = U_i[k]-U_l[k];
			euclideanDistance += dist_il*dist_il; 
		}
		
		k_il = Math.exp( - gamma * euclideanDistance );
		
		return k_il;
	}
	
	public double GetTrainLogLoss()
	{
		double logLoss = 0.0;
		
		double Y_hat_i, Y_i;
		
		int numObservedTrainInstances = 0;
		
		// go through the test instances
		for(int i = 0; i < numTrainInstances; i++)
		{
			Y_i = Y.get(i);
			
			if( Y_i != GlobalValues.MISSING_VALUE )
			{
				Y_hat_i = Probability(U.getRow(i));
				logLoss += - Y_i * Math.log( Y_hat_i ) - ( 1 - Y_i )*Math.log( 1- Y_hat_i);
				
				numObservedTrainInstances++;
			}
		}
		
		logLoss /= (double) numObservedTrainInstances; 
		
		return logLoss;
	}
	
	public double GetTestLogLoss()
	{
		double logLoss = 0.0;
		
		double Y_hat_i, Y_i;
		
		int numObservedTestInstances = 0;
		
		// go through the test instances
		for(int i = numTrainInstances; i < numTotalInstances; i++) 
		{
			Y_i = Y.get(i);
			
			if( Y_i != GlobalValues.MISSING_VALUE )
			{
				Y_hat_i = Probability(U.getRow(i));
				logLoss += - Y_i * Math.log( Y_hat_i ) - ( 1 - Y_i )*Math.log( 1- Y_hat_i);
				
				numObservedTestInstances++;
			}
		}
		
		logLoss /= (double) numObservedTestInstances; 
		
		return logLoss;
	}
	
	public double GetTestErrorRate()
	{
		double errorRate = 0.0;
		
		double Y_hat_i, Y_i, label_i;
		
		double numObservedTestInstances = 0.0;
		
		// go through the test instances
		for(int i = numTrainInstances; i < numTotalInstances; i++) 
		{
			Y_i = Y.get(i);
			
			if( Y_i != GlobalValues.MISSING_VALUE )
			{
				Y_hat_i = Probability(U.getRow(i));
				
				label_i = Y_hat_i > 0.5 ? 1.0 : 0.0; 
				
				if( Y_i != label_i )
					errorRate += 1.0;
				
				numObservedTestInstances += 1.0;
			}
		}
		
		errorRate /= numObservedTestInstances; 
		
		return errorRate; 
	}
}
