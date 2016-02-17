package MatrixFactorization;

import DataStructures.Tripple;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;
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

/*
 * The standard matrix factorization is best described by the loss function
 *  argmin_UV { u* loss_x(X, pl_x(U,Psi_i')) + (1-u)*loss_y(Y, pl_y(U,W')) + l_u*||U||+l_v*||Psi_i||+l_w*||W||) }, 
 *  where ||.|| is frobenius second norm
 *  l_u, l_v, l_w are lambdas, coefficients of the regularization 
 *  
 */

public class MatrixFactorization extends MatrixFactorizationModel 
{
	/*
	 * The alpha coefficient is used to denote the speed of convergence of the 
	 * gradient descent
	 */ 
	public double learningRate;

	/*
	 * The coefficient is used to denote the weight of the regularization
	 * terms in the loss function
	 */
	public double lambdaU, lambdaV;
        
        
    /*
     * Value ranges for the initialization of UV,W
     */
    public double minInitialValue, maxInitialValue;
        
	/*
	 * The u hyperparameter
	 */
	public double alpha;
	
	/*
     * Max number of epocs and the number of unimproved epocs allowed
     */
	public int maxEpocs, unimprovedEpocsTolerance;
	protected int currentEpochIndex;

    /*
     * The min allowed Value an epoc loss can have, per observed cell
     */
    public double minAllowedEpocLossPerCell;
    
    /*
     * Stop if in the last threshold steps we have less then a threshold
     * improvement
     */
    public int thresholdSteps;
    public double thresholdImprovement;
        
	/* 
     * A random generator
     */
	Random rand = new Random();
	
	/*
	 * we might want to reshuffle the random list of reconstruction cells
	 * each X epocs
	 */
	public int tickFrequency; 

    /*
     * The list of observed items in the XTraining and XTesting
     */ 
    protected List<Tripple> XObserved;
    
    /*
     * History of 
     */
    public int numTotalInstances, numFeatures; 
    // the reciprocal of the figures, i.e. 1/val
    double rec_features, rec_totalInstances;
        

	/*
	 * Constructor for the regularized SVD 
	 */
	public MatrixFactorization(int factorsDim) 
	{
            super(factorsDim); 

            learningRate = 0.0001; 
            double lambdaValue = 0.001; 
               
            tickFrequency = 1;
            
            alpha = 1;  
            maxEpocs  = 24000;
            currentEpochIndex=0;
            unimprovedEpocsTolerance = 400; 
            minAllowedEpocLossPerCell = 0.0001; 
            
            minInitialValue = -0.0100;
            maxInitialValue = 0.0100; 

            thresholdSteps = 20;
            thresholdImprovement = 0;
            
            lambdaU = lambdaValue;  
            lambdaV = lambdaValue;  

            XObserved = null;
	}

	/*
	 * Gradient descent training implementation matching X = UV'
	 */
	public void TrainReconstructionLoss(int i, int j)
	{
        double error_ij = X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, j); 
        
        double u_ik, v_kj, grad_u_ik, grad_v_kj;
        
        for(int k = 0; k < latentDim; k++)
        {
        	v_kj = V.get(k, j);
        	u_ik =  U.get(i, k);
        	
        	// skip updating the bv bias weight column of U
			if( k != latentDim-1 )
			{
	        	grad_u_ik = -2*error_ij*v_kj; 
	        	// dont add regularization to the biases
	        	if(k != latentDim-2) grad_u_ik += rec_features*2*lambdaU*u_ik;
	        	
	        	U.set(i, k, u_ik - learningRate * grad_u_ik);
			}
			
			// dont update the weights of bu
			if( k != latentDim-2) 
			{
	            grad_v_kj = -2*error_ij*u_ik;
	            
	            if(k != latentDim-1) grad_v_kj += rec_totalInstances*2*lambdaV* v_kj;
	    	 	
	            V.set(k, j, v_kj - learningRate * grad_v_kj);
			}
        }
	}
	
	
	// get the total X loss of the reconstruction, the loss will be called
        // for getting the loss of a cell, so it must be initialized first
	public double GetTotalReconstructionLoss()
    {
        double XTrainingLoss = 0;
        
        for(int i = 0; i < numTotalInstances; i++)
            for(int j = 0; j < numFeatures; j++)
                if( X.get(i, j) != GlobalValues.MISSING_VALUE )
                {
                	double err =  X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, j);
                    XTrainingLoss += err*err;
                }
                    
        return XTrainingLoss;
    }
	
    // initialize the matrixes before training
    public void Initialize(Matrix x)
    {
        X = x;
        
        // initialize latent representation of X into latent space represented by U
        U = new Matrix(numTotalInstances, latentDim);
        U.RandomlyInitializeCells(minInitialValue, maxInitialValue);
        //U.SetUniqueValue(0);
       
    	// initialize a transposed Psi_i
        V = new Matrix(latentDim, numFeatures);
        V.RandomlyInitializeCells(minInitialValue, maxInitialValue);
        //U.SetUniqueValue(0);

		for(int i=0; i < numTotalInstances; i++)
			U.set(i, latentDim-1, 1.0);
		for(int j=0; j < numFeatures; j++)
			V.set(latentDim-2, j, 1.0);
        
        // apply a different initialization if neccessary
        PostInitializationRoutine();
        
        // setup the prediction and loss 
        
        XObserved = new ArrayList<Tripple>();
        // record the observed values
        for(int i=0; i < X.getDimRows(); i++)
            for(int j=0; j < X.getDimColumns(); j++)
                if( X.get(i, j) != GlobalValues.MISSING_VALUE )
                    XObserved.add(new Tripple(i, j));     
    }
         

    /*
     * A method used for pre-epoch routines of the algorithm
     */
    public void PreEpochRoutine()
    {
    	// subclasses can override and implement their pre epoch routines
    }
    
    /*
     * A method used for pre-epoch routines of the algorithm
     */
    public void PostInitializationRoutine()
    {
    	// subclasses can override and implement their pre epoch routines
    }
       
	/*
	 * Train and generate the decomposition 
	 * */

    public double Decompose(Matrix X) 
    {
    	// initialize the model
    	Initialize(X);
        
		double minLoss = Double.MAX_VALUE;
        int lastImprovementEpoch = 0;
		
        int numObservedPoints = XObserved.size();
		int numSelectedPoints = numObservedPoints;
        
		Logging.println("Observed: " + numObservedPoints + ", Selected: " + numSelectedPoints, 
                        Logging.LogLevel.DEBUGGING_LOG);

		for(int epoc = 0; epoc < maxEpocs; epoc++)
		{
			// call pre-epoch routines
			PreEpochRoutine();
			
            for( int i = 0; i < numSelectedPoints; i++ )
            {
                TrainReconstructionLoss(XObserved.get(i).row, XObserved.get(i).col); 
            } 

            // check the loss only after some epocs
            if(currentEpochIndex % 5 != 0) 
            {
            	// randomly shuffle the observed points
                // every some epoc
                Collections.shuffle(XObserved); 
            	
                currentEpochIndex++;
            	continue;
            }
            
            double XLoss = GetTotalReconstructionLoss(),
		            regLossU = U.getSquaresSum(),
		            regLossV = V.getSquaresSum();
            
            double epocLoss = XLoss + lambdaU*regLossU +lambdaV*regLossV;  
                
            // in case there is no improvement on minimizing the loss function
            if( epocLoss > minLoss )
            {
                if(epoc - lastImprovementEpoch > unimprovedEpocsTolerance)
                    break; 

                Logging.println("No Improvement:: Epoc: " + epoc + ". Loss: " + epocLoss,
                                    Logging.LogLevel.DEBUGGING_LOG);
            }
            else
            {
               minLoss = epocLoss;	
               lastImprovementEpoch = epoc; 

               DecimalFormat twoDForm = new DecimalFormat("#.##");
               
               Logging.println(
                       epoc + ": " +
                       "L: " + twoDForm.format(minLoss) + 
                       ", LX: " + twoDForm.format(XLoss) +
                       ", LR: " + twoDForm.format(lambdaU*regLossU +lambdaV*regLossV) 
                       , Logging.LogLevel.DEBUGGING_LOG);     
               
            }   
            
            currentEpochIndex++;
		}

        return minLoss;
	}

        /*
         * Factorize the train set and the test set into latent representations
         * Please note that the latent objects have to be initialized before calling
         * this function as the references will be updated inside the method.
         */
    @Override
    public double Factorize(DataSet trainingSet, DataSet testingSet, DataSet latentTrainSet, DataSet latentTestSet) 
    {
    	labelsLossType = LossType.SmoothHinge;
    	
    	trainSet = trainingSet; 
    	testSet = testingSet;
        trainSet.ReadNominalTargets(); 
        int numTrainInstances = trainSet.instances.size();
        numTotalInstances = numTrainInstances + testSet.instances.size(); 
        numFeatures = trainSet.numFeatures;
        rec_features = 1.0 / (double) numFeatures;
        rec_totalInstances = 1.0 / (double) numTotalInstances;
        
        Logging.println("Train: " + numTrainInstances + ", Test:" + (numTotalInstances-numTrainInstances) + ", NumFeat: " + 
        numFeatures, LogLevel.DEBUGGING_LOG);
        
        Matrix x = new Matrix();
        x.LoadDatasetFeatures(trainSet, false);
        x.LoadDatasetFeatures(testSet, true);
        Matrix y = new Matrix();
        y.LoadDatasetLabels(trainSet, false);
        y.LoadDatasetLabels(testSet, true); 
        
    	Decompose(x);
       
        if( latentTrainSet != null )
        {
            latentTrainSet.LoadMatrixes(U, y, 0, numTrainInstances);
            Logging.println("Latent Train(" + latentTrainSet.instances.size() + "," + latentTrainSet.instances.get(0).features.size() + ")" , Logging.LogLevel.DEBUGGING_LOG); 
        }
        
        if( latentTestSet != null ) 
        {
            latentTestSet.LoadMatrixes(U, y, numTrainInstances, numTotalInstances);
            Logging.println("Latent Test(" + latentTestSet.instances.size() + "," + latentTestSet.instances.get(0).features.size() + ")" , Logging.LogLevel.DEBUGGING_LOG);
        } 
        
        return 1.0;
    }
 
    public DataSet Factorize(DataSet dataset) {
        
        // Split the decomposed matrix into the factorized datasets
        
        Matrix x = new Matrix();
        x.LoadDatasetFeatures(dataset, false);
        Matrix y = new Matrix();
        y.LoadDatasetLabels(dataset, false);
        
        numTotalInstances = dataset.instances.size();
        dataset.ReadNominalTargets();
        numFeatures = dataset.numFeatures;
        
        Decompose(x);
        
        DataSet latentSet = new DataSet();
    	latentSet.LoadMatrixes(U, Y);
         
    	return latentSet;
    }
	
    /*
     * Fold in a test dataset by projecting it into the latent space without
     * updating the 
     */
    public DataSet FoldIn(DataSet testSet)
    {
    	int testDim = testSet.instances.size();
    	
    	// create a nested factorization class to do the fold in
    	// however disable updating 
    	DataSet latentTestSet = new DataSet();
    	latentTestSet.name = testSet.name;
    	latentTestSet.numFeatures = latentDim;
    	
    	for(int i = 0; i < testSet.instances.size(); i++)
    	{
    		System.out.println("Instance: " + i);
    		
    		DataInstance ins = testSet.instances.get(i);
    		
    		DataInstance foldedIns = new DataInstance();
    		
    		for(int j = 0; j < latentDim; j++)
    			foldedIns.features.add(new FeaturePoint(0));
    		
    		double previousLoss = Double.MAX_VALUE, currentLoss = Double.MAX_VALUE;
        		
    		while( currentLoss <= previousLoss )
    		{
    			for(int j = 0; j < latentDim; j++)
    			{
    				double error_j = 0;
        			double predicted_j = 0;
        			double real_j = ins.features.get(j).value;
        			
        			// compute the predicted j index of the latent test instance
        			for(int k = 0; k < latentDim; k++)
        			{
        				double f_k = foldedIns.features.get(k).value;
        				predicted_j = f_k*V.get(j, k);
        			}
        			
        			error_j = (real_j-predicted_j)*(real_j-predicted_j);
        			
	    			for(int k = 0; k < latentDim; k++)
	    			{
	    				double f_k = foldedIns.features.get(k).value;
	    				double grad_f_k = -2*error_j*V.get(j, k) + 2*lambdaU*f_k;
	    				foldedIns.features.get(k).value = f_k - learningRate * grad_f_k;
	    			}
    			}
    			
    			
    			// compute total loss
    			double totalLoss = 0;
    			
    			for(int j = 0; j < latentTestSet.numFeatures; j++)
    			{
    				double error_j = 0;
        			double predicted_j = 0;
        			double real_j = ins.features.get(j).value;
        			
        			// compute the predicted j index of the latent test instance
        			for(int k = 0; k < latentDim; k++)
        			{
        				double f_k = foldedIns.features.get(k).value;
        				predicted_j = f_k*V.get(j, k);
        			}
        			
        			error_j = (real_j-predicted_j)*(real_j-predicted_j);
        			
        			totalLoss += error_j;
    			}
    			
    			previousLoss = currentLoss;
    			currentLoss = totalLoss;
    			
    			
    			
    		}
    		
    		System.out.println("Fold ins loss: " + currentLoss);
    		
    		foldedIns.target = ins.target;
    		latentTestSet.instances.add(foldedIns);
    	}
    	
    	return latentTestSet;
    }
}
