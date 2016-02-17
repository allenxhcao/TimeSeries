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

public class SupervisedMatrixFactorization extends MatrixFactorizationModel 
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
	public double lambdaU, lambdaV, lambdaW;
        
        
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
        
        public double [] intercepts;
        
        public double maxMargin;
        
	/*
         * A random generator
         */
	Random rand = new Random();
        
        /*
         * percentage of observed points to update in a single epoc 
         */
	public double stochasticUpdatesPercentage;
	
	/*
	 * we might want to reshuffle the random list of reconstruction cells
	 * each X epocs
	 */
	public int tickFrequency; 

        /*
         * The list of observed items in the XTraining and XTesting
         */
        protected List<Tripple> XObserved;

		protected List<Tripple> YObserved;
        
        /*
         * History of 
         */
        public int numTrainInstances, numTotalInstances,
        numLabels, numFeatures;
        // the reciprocal of the figures, i.e. 1/val
        double rec_features, rec_totalInstances, rec_labels, rec_trainInstances;
        

	/*
	 * Constructor for the regularized SVD 
	 */
	public SupervisedMatrixFactorization(int factorsDim) 
	{
            super(factorsDim); 

            learningRate = 0.0001; 
            double lambdaValue = 0.001; 
            
            stochasticUpdatesPercentage = 1;   
            tickFrequency = 1;
            
            alpha = 1;  
            maxEpocs  = 24000;
            currentEpochIndex=0;
            unimprovedEpocsTolerance = 400; 
            minAllowedEpocLossPerCell = 0.0001; 
            
            minInitialValue = -1;
            maxInitialValue = 1; 

            thresholdSteps = 20;
            thresholdImprovement = 0;
            
            lambdaU = lambdaValue;  
            lambdaV = lambdaValue;  
            lambdaW = lambdaValue;  

            XObserved = null;  
            YObserved = null;
            
            
            maxMargin = 1.0;
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
        	u_ik =  U.get(i, k);
            v_kj = V.get(k, j);
            
            grad_u_ik = -2*alpha*error_ij*v_kj + rec_features*lambdaU*u_ik;
            grad_v_kj = -2*alpha*error_ij*u_ik + rec_totalInstances*2*lambdaV* v_kj;
    	 	
            U.set(i, k, u_ik - learningRate * grad_u_ik);
            V.set(k, j, v_kj - learningRate * grad_v_kj);
        }
	}
	
	/*
	 * The gradient descent training concerning matching Y = UW
	 */
        
	public void TrainClassificationAccuracy(int i, int l) 
	{
		// get weights by input product
	    double val = MatrixUtilities.getRowByColumnProduct(U, i, W, l);
	    // add the intercept
    	val += (numLabels > 2 ? intercepts[l] : intercepts[0]);
	    
    	if( labelsLossType == LossType.Logistic  )
    	{
	        double sig = Sigmoid.Calculate(val);
            double y = Y.get(i, l);
    	
            double u_ik, w_kl, grad_u_ik, grad_w_kl;
            
            for(int k = 0; k < latentDim; k++)
            {
                u_ik =  U.get(i, k);
                w_kl = W.get(k, l);
                
                grad_u_ik =  (1-alpha) * -(y-sig)*w_kl + rec_labels*lambdaU*u_ik;
                grad_w_kl = (1-alpha) * -(y-sig)*u_ik + rec_trainInstances*2*lambdaW*w_kl;              
 
                U.set(i, k, u_ik - learningRate * grad_u_ik);
            	W.set(k, l, w_kl - learningRate * grad_w_kl);
            }
            
            // update the intercept
	        if(numLabels > 2)
	        	intercepts[l] -= learningRate * (1-alpha) * -(y-sig);
		    else
		    	intercepts[0] -= learningRate * (1-alpha) * -(y-sig);
    	}
    	else if(labelsLossType == LossType.SmoothHinge)
    	{
    		  double y = Y.get(i, l);
    		  double z = val * y;	    				  
              
              double gradHinge = 0;              
              if(z <= 0)					gradHinge = -1; 
              else if ( z > 0 && z < 1)		gradHinge = z-1;
              else if( z >= 1)				gradHinge = 0;
              
          	double u_ik, w_kl, grad_u_ik, grad_w_kl;
          	
            for(int k = 0; k < latentDim; k++)
            {
                u_ik =  U.get(i, k);
                w_kl = W.get(k, l);
                
                grad_u_ik =  (1-alpha)*y*gradHinge*w_kl + rec_labels*lambdaU*u_ik;
                grad_w_kl = (1-alpha)*y*gradHinge*u_ik + rec_trainInstances*2*lambdaW*w_kl;              
 
                U.set(i, k, u_ik - learningRate * grad_u_ik);
            	W.set(k, l, w_kl - learningRate * grad_w_kl);
            }
            
            // update the intercept
	        if(numLabels > 2)
	        	intercepts[l] -= learningRate*(1-alpha)*y*gradHinge; 
		    else
		    	intercepts[0] -= learningRate*(1-alpha)*y*gradHinge;
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
	
	// get the total X loss of the reconstruction, the loss will be called
    // for getting the loss of a cell, so it must be initialized first
	public double GetTotalClassificationAccuracyLoss()
    {
        double YTrainingLoss = 0;
        
        for(int i = 0; i < numTotalInstances; i++)
            for(int l = 0; l < Y.getDimColumns(); l++) 
                if( Y.get(i, l) != GlobalValues.MISSING_VALUE )
                {
                	// get weights by input product
        		    double val = MatrixUtilities.getRowByColumnProduct(U, i, W, l);
        		    val += (numLabels > 2 ? intercepts[l] : intercepts[0]);
        		    
        		    if(labelsLossType == LossType.Logistic)
        		    {
	        		    double sig = Sigmoid.Calculate(val);            
	                    double y = Y.get(i, l);
	                    
	                    YTrainingLoss += -y*Math.log( sig ) - (1-y)*Math.log(1-sig);
        		    }
        		    else if(labelsLossType == LossType.SmoothHinge)
        		    {
    		    	    double z = val*Y.get(i, l); 
    		    	    
    		            if(z <= 0) 
    		            	YTrainingLoss += 0.5 - z;
    		            else if ( z > 0 && z < 1) 
    		            	YTrainingLoss += 0.5 * (1-z)*(1-z);
    		            else if( z >= 1)
    		            	YTrainingLoss += 0;   
        		    }
                }
                    
        return YTrainingLoss;
    }
       
        
        // initialize the matrixes before training
        public void Initialize(Matrix x, Matrix y)
        {
            // store original matrix
            //XTrain = new Matrix(X);
            X = x;
            //Y = new Matrix(Y);
            Y = y; 
            
            // initialize latent representation of X into latent space represented by U
            U = new Matrix(numTotalInstances, latentDim);
            U.RandomlyInitializeCells(minInitialValue, maxInitialValue);
           
        	// initialize a transposed Psi_i
            V = new Matrix(latentDim, numFeatures);
            V.RandomlyInitializeCells(minInitialValue, maxInitialValue);

            // initialize a transposed W 
            W = new Matrix( latentDim, y.getDimColumns());
            W.RandomlyInitializeCells(minInitialValue, maxInitialValue);
        
            // apply a different initialization if neccessary
            PostInitializationRoutine();
            
            // setup the prediction and loss 
            
            XObserved = new ArrayList<Tripple>();
            // record the observed values
            for(int i=0; i < X.getDimRows(); i++)
                for(int j=0; j < X.getDimColumns(); j++)
                    if( X.get(i, j) != GlobalValues.MISSING_VALUE )
                        XObserved.add(new Tripple(i, j)); 
            
            Collections.shuffle(XObserved);
            
            YObserved = new ArrayList<Tripple>();
            // record the observed values
            for(int i = 0; i < Y.getDimRows(); i++)
                for(int l = 0; l < Y.getDimColumns(); l++)
                    if( Y.get(i, l) != GlobalValues.MISSING_VALUE )
                        YObserved.add(new Tripple(i, l)); 
            
            Collections.shuffle(YObserved);
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

    public double Decompose(Matrix X, Matrix Y) 
    {
    	// initialize the model
    	Initialize(X, Y);
        
		double minLoss = Double.MAX_VALUE;
        int lastImprovementEpoch = 0;
		
        int numObservedPoints = XObserved.size();
		int numSelectedPoints = numObservedPoints;
		int numObservedLabels = YObserved.size();
        
		Logging.println("Observed: " + numObservedPoints + ", Selected: " + numSelectedPoints, 
                        Logging.LogLevel.DEBUGGING_LOG);

		for(int epoc = 0; epoc < maxEpocs; epoc++)
		{
			// call pre-epoch routines
			PreEpochRoutine();
			
            // randomly shuffle the observed points
            // every some epoc
            Collections.shuffle(XObserved); 
            Collections.shuffle(YObserved);
        
            for( int i = 0; i < numSelectedPoints; i++ )
            {
                TrainReconstructionLoss(XObserved.get(i).row, XObserved.get(i).col); 
            }

            for( int i = 0; i < numObservedLabels; i++ )
            {      
            	TrainClassificationAccuracy(YObserved.get(i).row, YObserved.get(i).col);
        	}
            
            // check the loss only after some epocs
            if(currentEpochIndex % 1 != 0)
            {
            	currentEpochIndex++;
            	continue;
            }
            
            double XLoss = GetTotalReconstructionLoss(),
		            YLoss = GetTotalClassificationAccuracyLoss(),
		            regLossU = U.getSquaresSum(),
		            regLossV = V.getSquaresSum(),
		            regLossW = W.getSquaresSum();
            
            double epocLoss = alpha * XLoss + (1-alpha) * YLoss + lambdaU*regLossU +lambdaV*regLossV +lambdaW*regLossW;  
                
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
                       ", LX: " + twoDForm.format(alpha*XLoss) +
                       ", LY: " + twoDForm.format((1-alpha)*YLoss) +
                       ", LR: " + twoDForm.format(lambdaU*regLossU +lambdaV*regLossV +lambdaW*regLossW) 
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
        numTrainInstances = trainSet.instances.size();
        numTotalInstances = numTrainInstances + testSet.instances.size();
        numLabels = trainSet.nominalLabels.size(); 
        numFeatures = trainSet.numFeatures;
        rec_features = 1.0 / (double) numFeatures;
        rec_labels = 1.0 / (double) numLabels;
        rec_totalInstances = 1.0 / (double) numTotalInstances;
        rec_totalInstances = 1.0 / (double) numTotalInstances;
        
        
        Logging.println("NumInst: " + numTrainInstances + "+" + numTotalInstances + ", NumFeat: " + 
        numFeatures + ", NumLabels: " + numLabels, LogLevel.DEBUGGING_LOG);
        
        Matrix x = new Matrix();
        x.LoadDatasetFeatures(trainSet, false);
        x.LoadDatasetFeatures(testSet, true);
        Matrix y = new Matrix();
        y.LoadDatasetLabels(trainSet, false);
        y.LoadDatasetLabels(testSet, true);
        
        // hide test set labels
        for(int i = numTrainInstances; i < numTotalInstances; i++)
        {
            y.set(i, 0, GlobalValues.MISSING_VALUE);
        }
        
        // in case the number of labels is more than two we have to expand the 
        // label matrix into a 1 vs all configuration
        if(numLabels > 2)
        {
        	// initialize the intercepts array
    	    intercepts = new double[numLabels];
                
	        // expand the labels vector matrix into L number of one vs all classification vectors
	        Matrix yExpanded = new Matrix(numTotalInstances, numLabels);
	        
	        // initialize the extended representation  
	        for(int i = 0; i < numTrainInstances; i++) 
	        {
	        	// firts set everything to either 0 or -1, i.e. the other class
	            for(int j = 0; j < numLabels; j++) 
	            {
	                if(labelsLossType == LossType.Logistic)
	                    yExpanded.set(i, j, 0);
	                else if(labelsLossType == LossType.SmoothHinge)
	                    yExpanded.set(i, j, -1);
	                else
	                	yExpanded.set(i, j, -1);
	            }
	            
	            // then set the real label index to 1
	            int indexLabel = trainSet.nominalLabels.indexOf( y.get(i, 0) );
	            yExpanded.set(i, indexLabel, 1.0);
	        }
	        
	        // hide test labels 
	        for(int i = numTrainInstances; i < numTotalInstances; i++) 
	        {
	            for(int j = 0; j < numLabels; j++) 
	            {
	            	yExpanded.set(i, j, GlobalValues.MISSING_VALUE);
	            }
	        }
        
	        Logging.println("Expanded Labels:\n ", LogLevel.DEBUGGING_LOG);
	        Logging.print(yExpanded, LogLevel.DEBUGGING_LOG);
	        
            Decompose(x, yExpanded);
        }
        else
        {
        	// make sure that the binary labels are set correctly
        	// some binary datasets comes as 0-1, some others as -1,1
        	// and some as 1-2, so keep the 1 and fix the other
        	for(int i = 0; i < numTrainInstances; i++) 
	        {
        		if( y.get(i, 0) != 1 )
        		{
		    		if(labelsLossType == LossType.Logistic)
		                y.set(i, 0, 0);
		            else if(labelsLossType == LossType.SmoothHinge)
		                y.set(i, 0, -1);
		            else
		            	y.set(i, 0, -1);
        		}
	        }
        	
        	// initialize the intercept
        	intercepts = new double[1];
        	// decompose the matrixes
        	
        	Logging.println("Expanded Labels:\n ", LogLevel.DEBUGGING_LOG);
 	        Logging.print(y, LogLevel.DEBUGGING_LOG);
 	        
        	Decompose(x, y);
        }
        
    	double errorRate = 0;
    	
        // classify the test set
        for(int i = numTrainInstances; i < numTotalInstances; i++)
        {
        	double realLabel = testSet.instances.get(i-numTrainInstances).target;
            double predictedLabel = PredictLabel(i);
            Logging.println("Real " + realLabel + ", Predicted: " + predictedLabel, LogLevel.DEBUGGING_LOG);
            
            if( realLabel != predictedLabel )
            	errorRate+=1.0;
        } 
        
        errorRate /= (double)(numTotalInstances-numTrainInstances);
        
        Logging.println("Error Rate: " + errorRate, LogLevel.DEBUGGING_LOG);

        // set the real labels of the test set back visible,
        // we have hidden them during learning
        for(int i = 0; i < numTrainInstances; i++)
            y.set(i, 0, trainSet.instances.get(i).target);
        
        for(int i = numTrainInstances; i < numTotalInstances; i++)
            y.set(i, 0, testSet.instances.get(i-numTrainInstances).target); 
        
        
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
        
        return errorRate;
    }
    
    public double PredictLabel(int i)
    {
    	double label = 0;
    	
    	// if the loss used is the logistic loss then the label prediction
    	// is done using the sigmoid confidence value
    	if( labelsLossType == LossType.Logistic )
    	{
	    	// if more than two labels exists then compute the 
	    	// confidence for all labels and pick the label for 
	    	// wich the highest confidence is produced
	    	if( numLabels > 2)
	    	{
	    		double maxConfidence = 0;
	    		
		    	for(int l = 0; l < numLabels; l++)
		    	{
		    		// get weights by input product
        		    double val = MatrixUtilities.getRowByColumnProduct(U, i, W, l) + intercepts[l];
        		    
		    		double confidence = Sigmoid.Calculate(val);
		    		
		    		Logging.println("Label " + l + ", Confidence: " + confidence, LogLevel.DEBUGGING_LOG);
		    		
		    		if(confidence > maxConfidence)
		    		{
		    			maxConfidence = confidence;
		    			label = (double) l;
		    		}
		    		
		    	}
	    	}
	    	else
	    	{
    			double val =  MatrixUtilities.getRowByColumnProduct(U, i, W, 0) + intercepts[0];
	    		double confidence = Sigmoid.Calculate(val);
	    		
	    		Logging.println("Confidence: " + confidence, LogLevel.DEBUGGING_LOG);
	    		
	    		if( confidence > 0.5)
	    			label = 1.0;
	    		else
	    			label = 0.0;
	    	}
    	}
    	else if( labelsLossType == LossType.SmoothHinge )
    	{
    		
    		// for binary classification check whether product is 
    		// on correct side of the boundary
    		if(numLabels == 2)
    		{
    			double val =  MatrixUtilities.getRowByColumnProduct(U, i, W, 0) + intercepts[0];
	    		
	    		if( val > 0 )
	    			label = 1.0;
	    		else
	    			label = 0.0;
    		}
    		else
    		{
    			for(int l = 0; l < numLabels; l++)
		    	{
    				double maxConfidence = 0;
    				
		    		// get weights by input product
    				double val =  MatrixUtilities.getRowByColumnProduct(U, i, W, l) + intercepts[l];
    	    		
		    		Logging.println("Label " + l + ", Confidence: " + val, LogLevel.DEBUGGING_LOG);
		    		
		    		if(val > maxConfidence)
		    		{
		    			maxConfidence = val;
		    			label = (double) l; 
		    		}
		    		
		    	}
    		}
    		
    	}
    	
    	return label;
    }

    public DataSet Factorize(DataSet dataset) {
        
        // Split the decomposed matrix into the factorized datasets
        
        Matrix x = new Matrix();
        x.LoadDatasetFeatures(dataset, false);
        Matrix y = new Matrix();
        y.LoadDatasetLabels(dataset, false);
        
        numTrainInstances = dataset.instances.size();
        numTotalInstances = dataset.instances.size();
        dataset.ReadNominalTargets();
        numLabels = dataset.nominalLabels.size();
        numFeatures = dataset.numFeatures;
        
        Decompose(x, y);
        
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
