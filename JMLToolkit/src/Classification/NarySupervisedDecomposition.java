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

import java.nio.channels.ShutdownChannelGroupException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.rmi.CORBA.Util;

/*
 * The standard matrix factorization is best described by the loss function
 *  argmin_UV { u* loss_x(X, pl_x(U,Psi_i')) + (1-u)*loss_y(Y, pl_y(U,W')) + l_u*||U||+l_v*||Psi_i||+l_w*||W||) }, 
 *  where ||.|| is frobenius second norm
 *  l_u, l_v, l_w are lambdas, coefficients of the regularization 
 *  
 */

public class NarySupervisedDecomposition
{
	// the predictors X and the extended target Y
	Matrix X,YExtended, Y;
	
	// the learning rate 
	public double eta;
	// the maximum epochs
	public int maxEpochs; 
        
    // the impact switch between predictors loss and target loss
	public double alpha;
	
	// the latent data dimension D_i
	public int D;
	
	// tha latent matrices and the biases
	public Matrix U;
	public double [] biasU;
	public Matrix V;
	public double [] biasV;
	public Matrix W;
	public double [] biasW;
	
	public double lambdaU, lambdaV, lambdaW;
	
        
	Random rand = new Random();
        
    protected List<Tripple> XObserved;
	protected List<Tripple> YObserved;
        
	public int numTrainInstances, numTotalInstances, numLabels, numFeatures;
	public double recNumTrainInstances, recNumTotalInstances, recNumLabels, recNumFeatures;
    
	/*
	 * Constructor for the regularized SVD 
	 */
	public NarySupervisedDecomposition(int factorsDim) 
	{
		D = factorsDim;
	    
		XObserved = null;  
	    YObserved = null;

	}

	/*
	 * Gradient descent training implementation matching X = UV'
	 */
	public void TrainReconstructionLoss(int i, int j)
	{
        double error_ij = X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasU[i] - biasV[j]; 
        
        double u_ik, v_kj, grad_u_ik, grad_v_kj;
        
        for(int k = 0; k < D; k++)
        {
        	u_ik =  U.get(i, k);
            v_kj = V.get(k, j);
            
            grad_u_ik = -2*alpha*error_ij*v_kj + 2*lambdaU*u_ik;
            grad_v_kj = -2*alpha*error_ij*u_ik + 2*lambdaV* v_kj;
    	 	
            U.set(i, k, u_ik - eta * grad_u_ik);
            V.set(k, j, v_kj - eta * grad_v_kj);
        }
        
        biasU[i] -= eta*-2*alpha*error_ij;
        biasV[j] -= eta*-2*alpha*error_ij;
	}
	
	/*
	 * The gradient descent training concerning matching Y = UW
	 */
        
	public void TrainClassificationAccuracy(int i, int l) 
	{
		// get weights by input product
	    double val = MatrixUtilities.getRowByColumnProduct(U, i, W, l) + biasU[i] + biasW[l];
	    
	    double y_hat = Sigmoid.Calculate(val);
        double y = YExtended.get(i, l);
    	
        double u_ik, w_kl, grad_u_ik, grad_w_kl;
            
        for(int k = 0; k < D; k++)
        {
            u_ik =  U.get(i, k);
            w_kl = W.get(k, l);
                
            grad_u_ik =  (1-alpha) * -(y-y_hat)*w_kl + 2*lambdaU*u_ik;
            grad_w_kl = (1-alpha) * -(y-y_hat)*u_ik + 2*lambdaW*w_kl;              
 
            U.set(i, k, u_ik - eta * grad_u_ik);
        	W.set(k, l, w_kl - eta * grad_w_kl);
        }
         
        biasU[i] -= eta * (1-alpha) * -(y-y_hat); 
        biasW[l] -= eta * (1-alpha) * -(y-y_hat);
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
                	double err =  X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasU[i] - biasV[j];
                    XTrainingLoss += err*err;
                    
                    numObservedCells++;
                }
                    
        return XTrainingLoss / (double) numObservedCells;
    }
	
	//train log loss
	public double GetTrainTargetLoss()
    {
        double YTrainLoss = 0;
        int numObservedCells = 0;
        
        for(int i = 0; i < numTrainInstances; i++)
            for(int l = 0; l < numLabels; l++) 
                if( YExtended.get(i, l) != GlobalValues.MISSING_VALUE )
                {
                	// get weights by input product
        		    double val = MatrixUtilities.getRowByColumnProduct(U, i, W, l) + biasU[i] + biasW[l];
        		    
					double sig = Sigmoid.Calculate(val);            
					double y = YExtended.get(i, l);
					    
					YTrainLoss += -y*Math.log( sig ) - (1-y)*Math.log(1-sig);
					
					numObservedCells++;
        		   
                }
                    
        return YTrainLoss / (double) numObservedCells;
    }
	
	// the accuracy of the train set
	public double GetTrainAccuracy()
    {
        int numCorrectClassifications = 0;
        int numInstances = 0;
        
        for(int i = 0; i < numTrainInstances; i++)
        {
        	if( Y.get(i) != GlobalValues.MISSING_VALUE )
        	{
				double y = Y.get(i);
				double y_predicted = PredictLabel(i);
				    
				if(y == y_predicted)
					numCorrectClassifications++;
				
				numInstances++;
        	}
        }
                    
        return (double) numCorrectClassifications / (double) numInstances; 
    }

	// the accuracy of the test set
	public double GetTestAccuracy()
    {
        int numCorrectClassifications = 0;
        int numInstances = 0;
        
        for(int i = numTrainInstances; i < numTotalInstances; i++)
        {
        	if( Y.get(i) != GlobalValues.MISSING_VALUE )
        	{
				double y = Y.get(i);
				double y_predicted = PredictLabel(i);

				if(y == y_predicted)
					numCorrectClassifications++;

				numInstances++;
        	}
        }

        return (double) numCorrectClassifications / (double) numInstances; 
    }
       
           
    // initialize the matrixes before training
    public void Initialize(Matrix x, Matrix y)
    {
        // store original matrix
        X = x;
        Y = y;
        
        recNumTotalInstances = 1.0 / (double) numTotalInstances;
        recNumTrainInstances = 1.0 / (double) numTrainInstances; 
        recNumFeatures = 1.0 / (double) numFeatures;
        recNumLabels = 1.0 / (double) numLabels;
        
        // create the extended Y
        YExtended = new Matrix(numTrainInstances, numLabels);
        
        // set all the cells to zero initially
        for(int i = 0; i < numTrainInstances; i++)
        	for(int l = 0; l < numLabels; l++)
        		YExtended.set(i, l, 0.0);
        
        // set to 1 only the column corresponding to the label
        for(int i = 0; i < numTrainInstances; i++)
            YExtended.set(i, (int)Y.get(i), 1.0);  
        
        
        // initialize latent representation of X into latent space represented by U
        U = new Matrix(numTotalInstances, D);
        U.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
       
        biasU = new double[numTotalInstances];
        for(int i = 0; i < numTotalInstances; i++)
        	biasU[i] = X.GetRowMean(i);
       
    	// initialize a transposed Psi_i
        V = new Matrix(D, numFeatures);
        V.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);

        biasV = new double[numFeatures];
        for(int j = 0; j < numFeatures; j++)
        	biasV[j] = X.GetColumnMean(j);

        W = new Matrix( D, numLabels);
        W.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);

        biasW = new double[numLabels]; 
        for(int l = 0; l < numLabels; l++)
        	biasW[l] = YExtended.GetColumnMean(l); 

        // setup the prediction and loss 
        XObserved = new ArrayList<Tripple>();
        for(int i=0; i < numTotalInstances; i++)
            for(int j=0; j < numFeatures; j++)
                if( X.get(i, j) != GlobalValues.MISSING_VALUE )
                    XObserved.add(new Tripple(i, j)); 
       
        Collections.shuffle(XObserved); 

        YObserved = new ArrayList<Tripple>();
        // record the observed values
        for(int i = 0; i < numTrainInstances; i++)
            for(int l = 0; l < numLabels; l++) 
                if( YExtended.get(i, l) != GlobalValues.MISSING_VALUE )
                    YObserved.add(new Tripple(i, l)); 
        
        Collections.shuffle(YObserved);  
        
    }
         
       
	/*
	 * Train and generate the decomposition 
	 * */

    public double Decompose(Matrix X, Matrix Y) 
    {
    	// initialize the model
    	Initialize(X, Y);
        
		int numObservedPredictorCells = XObserved.size();
		int numObservedLabelCell = YObserved.size();

		for(int epoc = 0; epoc < maxEpochs; epoc++)
		{
			for( int predictorCellIndex = 0; predictorCellIndex < numObservedPredictorCells; predictorCellIndex++ )
                TrainReconstructionLoss(XObserved.get(predictorCellIndex).row, XObserved.get(predictorCellIndex).col); 

        	for( int labelCellIndex = 0; labelCellIndex < numObservedLabelCell; labelCellIndex++ )
        		TrainClassificationAccuracy(YObserved.get(labelCellIndex).row, YObserved.get(labelCellIndex).col);

            double predictorsLoss = GetPredictorsMSE(),
		            trainTargetLoss = GetTrainTargetLoss(),
    				trainAccuracy = GetTrainAccuracy(),
					testAccuracy = GetTestAccuracy();
            
           DecimalFormat twoDForm = new DecimalFormat("#.######");
           
           Logging.println(
                   epoc + ": " +
                   "LX: " + twoDForm.format(predictorsLoss) +
                   ", LossTrain: " + twoDForm.format(trainTargetLoss) +
                   ", AccTrain: " + twoDForm.format(trainAccuracy) + 
                   ", AccTest: " + twoDForm.format(testAccuracy) 
                   , Logging.LogLevel.DEBUGGING_LOG);     
               
		}
 
		return GetTestAccuracy(); 
                
	}

    public double PredictLabel(int i)
    {
    	double label = 0;
    	double maxConfidence = 0;
	    		
    	for(int l = 0; l < numLabels; l++)
    	{
    		// get weights by input product
		    double val = MatrixUtilities.getRowByColumnProduct(U, i, W, l) + biasU[i] + biasW[l];
		    
    		double confidence = Sigmoid.Calculate(val);
    		
    		if(confidence > maxConfidence)
    		{
    			maxConfidence = confidence;
    			label = (double) l;
    		}
    	}
    	
    	return label;
    }

}
