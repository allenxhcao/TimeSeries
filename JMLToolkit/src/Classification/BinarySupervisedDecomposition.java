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

public class BinarySupervisedDecomposition 
{
	// the learn rate coefficient for each loss term
	public double eta;

	// the coefficients of each loss term
	public double alpha;

	// the regularization parameters
	public double lambdaU, lambdaV, lambdaW;
	
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
	

	// the weights of the logistic regression
	double [] W;
	double biasW;	
	
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
	public BinarySupervisedDecomposition(int factorsDim) 
	{
            eta = 0.001;
            
            lambdaU = 0.001;  
            lambdaV = 0.001;  
            lambdaW = 0.001;  

            XObserved = null;  
            YObserved = null;
            
            D = factorsDim;
            
            alpha = 1.0;
            
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
        
    	X = x;
    	Y = y;
    	
        // initialize latent representation of X into latent space represented by U
        U = new Matrix(numTotalInstances, D);
        U.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
        
        // initialize a transposed Psi_i
        V = new Matrix(D, numFeatures);
        //Psi_i.SetUniqueValue(0.0);
        V.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON); 

        biasV = new double[numFeatures];
        for(int j = 0; j < numFeatures; j++)
        	biasV[j] = X.GetColumnMean(j);
        
		// initialize the alphas
		W = new double[D];
		
		// initialize the weights between -epsilon +epsilon
		for(int k = 0; k < D; k++)
			W[k] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON;
		
		for(int i = 0; i < numTrainInstances; i++)
			biasW += Y.get(i);
		biasW /= numTrainInstances;
				
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
        				", biasW=" + biasW, LogLevel.DEBUGGING_LOG);  
        
    }

	/*
	 * Train and generate the decomposition 
	 * */

    public double Decompose(Matrix X, Matrix Y) 
    {
    	// initialize the model
    	Initialize(X, Y);
    	
    	// supervised learning
    	for(int epoch = 0; epoch < maxEpochs; epoch++) 
		{
    		
		    for( int predictorCellIndex = 0; predictorCellIndex < XObserved.size(); predictorCellIndex++ )
            	UpdatePredictorsLoss(XObserved.get(predictorCellIndex).row, XObserved.get(predictorCellIndex).col);
		            
    		for( int targetCellIndex = 0; targetCellIndex < YObserved.size(); targetCellIndex++) 
				UpdateTargetLoss( YObserved.get(targetCellIndex) ); 
				
         
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

 		return GetTestErrorRate();
	}

    // update the latent representation to approximate X_{i,j}
    public void UpdatePredictorsLoss(int i, int j)
    {
    	double X_ij, error_ij;
    	
    	X_ij = X.get(i, j);
    	
    	if( X_ij == GlobalValues.MISSING_VALUE ) return;
    	
        error_ij = X_ij - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasV[j]; 
        
        for(int k = 0; k < D; k++)
        {
        	U.set(i, k, U.get(i, k) - eta * ( -2*alpha*error_ij*V.get(k,j) + 2*lambdaU*U.get(i, k) ) );
            V.set(k, j, V.get(k,j) - eta * ( -2*alpha*error_ij*U.get(i, k) + 2*lambdaV*V.get(k,j) ) ); 
    	}
        
        biasV[j] = biasV[j] - eta*-2*alpha * error_ij;
    }
    
    // update the latent nonlinear weights alpha to approximate target label Y_i
    public void UpdateTargetLoss(int i)
    {
    	double err_i, Y_i;
    	
		Y_i = Y.get(i);
		
		if( Y_i == GlobalValues.MISSING_VALUE ) return;
		
		err_i = Y_i - Probability(U.getRow(i)); 
		
		for(int k = 0; k < D; k++)
		{
			U.set(i, k,  U.get(i,k) - eta*( (1-alpha)*-err_i*W[k] + 2*lambdaU*U.get(i, k) ) );
			W[k] = W[k] - eta*( (1-alpha)*-err_i*U.get(i,k) + 2*lambdaW*W[k] ); 	
		}
		
		// update the bias parameter
		biasW = biasW - eta*(1-alpha)*-err_i; 
		
    }
    
    
	// compute the probability of an arbitrary predictors feature vector
	public double Probability(double [] predictorFeatures)
	{
		double val = biasW;
		
		for(int k = 0; k < D; k++)
			val += predictorFeatures[k]*W[k];
		
		return Utilities.Sigmoid.Calculate(val);
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
