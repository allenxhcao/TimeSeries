package Classification;

import DataStructures.Tripple;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;
import MatrixFactorization.MatrixFactorizationModel;
import MatrixFactorization.MatrixUtilities;
import TimeSeries.TotalVariation;
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

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.gui.beans.TrainingSetEvent;

public class StructurePreservingDecomposition 
{
	// the learn rate coefficient for each loss term
	public double eta;

	// the coefficients of each loss term
	public double alphaR, alphaA, alphaT, alphaD; 

	// the regularization parameters
	public double lambdaU, lambdaV, lambdaW;
	
	// training+testing predictors
	public Matrix X;
	// training labels
	public Matrix Y;
	// the extanded one-vs-all labels
	Matrix YExtended;
	
	// the number of latent dimensions
	public int maxEpochs;
	
	// the weights of the reconstruction linear regression models
	public Matrix U;
	double [] biasU;
	
	// the weights of the reconstruction linear regression models of predictors
	public Matrix V;
	double [] biasV;
	
	// the weights of the logistic regression
	public Matrix W;
	public double [] biasW;
	
	// number of latent dimensions
	public int D;
	
	
	// complexities 
	double [] C;
	
	/*
     * A random generator
     */
	Random rand = new Random();

    // The list of observed items in the XTraining and XTesting
    protected List<Tripple> XObserved;
    protected List<Tripple> YObserved;
    
    // number of training instances 
    public int numTrainInstances, 
    			numTotalInstances, 
    			numPoints,
    			numLabels;;
    
    double recNumTrainInstances,
    		recNumTotalInstances, 
    		recNumPoints, 
    		recNumLabels;
    
    // a decimal format used for printing    
    DecimalFormat df = new DecimalFormat("#.######");
    
    // the error complexities 
    double [] C_err;
    
    
    
	/*
	 * Constructor for the regularized SVD 
	 */
	public StructurePreservingDecomposition(int factorsDim) 
	{
            eta = 0.001;
            
            lambdaU = 0.001;  
            lambdaV = 0.001;  
            lambdaW = 0.001;  

            XObserved = null;  
            YObserved = null;
            
            D = factorsDim;
            
            alphaA = 1.0; 
            alphaR = 1.0; 
    		alphaT = 1.0; 
    		
            
  }
	
	// get the total X loss of the reconstruction, the loss will be called
        // for getting the loss of a cell, so it must be initialized first
	public double GetPredictorsLoss()
    {
        double XTrainingLoss = 0;
        int numObservedCells = 0;
        
        for(int i = 0; i < numTotalInstances; i++)
            for(int j = 0; j < numPoints; j++)
                if( X.get(i, j) != GlobalValues.MISSING_VALUE )
                {
                	double err =  X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasV[j];
                    XTrainingLoss += err*err;
                    numObservedCells++;
                }
                    
        return XTrainingLoss/(double)numObservedCells; 
    } 
	
	// get the total X loss of the reconstruction, the loss will be called
    // for getting the loss of a cell, so it must be initialized first
	public double GetTotalVariationLoss()
	{
	    double XTotalVariationLoss = 0;
	    
	    for(int i = 0; i < numTotalInstances; i++)
	    {
	    	double C_appx_i = 0;
	    	
	    	for(int p = 1; p < numPoints-1; p++)
	    	{
	    		double val_j_prev = RegressPredictor(i, p-1);
	    		double val_j = RegressPredictor(i, p);
	    		
	    		C_appx_i += (val_j_prev - val_j)*(val_j_prev - val_j);
	    	}
	    	
	    	double err =  C[i] - C_appx_i; 
	    	XTotalVariationLoss += err*err;
	    }
	                
	    return XTotalVariationLoss/(double)numTotalInstances;  
	} 
    
	// get the derivative loss
	public double GetDerivativeLoss()
	{
	    double XDerivativeLoss = 0; 
	    int numObservedCells = 0;
	    
	    for(int i = 0; i < numTotalInstances; i++)
	    {
	    	for(int j = 0; j < numPoints-1; j++)
	    	{
	        	double d_err_ij = X.get(i, j+1) - X.get(i, j) - RegressPredictor(i, j+1) + RegressPredictor(i, j);
	        	
	    		XDerivativeLoss += d_err_ij*d_err_ij;
	    		
	    		numObservedCells++;
	    	
	    	}
	    }
	                
	    return XDerivativeLoss/(double)numObservedCells;  
	} 
    
    // initialize the matrixes before training
    public void Initialize(Matrix x, Matrix y)
    {
        // store original matrix
        
    	X = x;
    	Y = y;
    	
    	 // create the extended Y
        YExtended = new Matrix(numTotalInstances, numLabels);
        
        // set all the cells to zero initially
        for(int i = 0; i < numTrainInstances; i++)
        	for(int l = 0; l < numLabels; l++)
        		YExtended.set(i, l, 0.0);
        
        // set to 1 only the column corresponding to the label
        for(int i = 0; i < numTotalInstances; i++)
            YExtended.set(i, (int)Y.get(i), 1.0);  
        
    	
        // initialize latent representation of X into latent space represented by U
        U = new Matrix(numTotalInstances, D);
        U.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
        
        biasU = new double[numTotalInstances];
        for(int i = 0; i < numTotalInstances; i++)
        	biasU[i] = X.GetRowMean(i);
        
        // initialize a transposed Psi_i
        V = new Matrix(D, numPoints);
        //Psi_i.SetUniqueValue(0.0);
        V.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON); 

        biasV = new double[numPoints];
        for(int j = 0; j < numPoints; j++)
        	biasV[j] = X.GetColumnMean(j);
        
		// initialize the alphas
        W = new Matrix( D, numLabels);
        W.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);

        biasW = new double[numLabels]; 
        for(int l = 0; l < numLabels; l++)
        	biasW[l] = YExtended.GetColumnMean(l); 
        
        
		// initialize the complexities
		C = new double[numTotalInstances];
		for(int i = 0; i < numTotalInstances; i++) 
			C[i] = TotalVariation.getInstance().GetTotalVariation(X.getRow(i)); 
	
		// initialize the errors C(X(i)) - C(X_hat(i))
		C_err = new double[numTotalInstances];
		
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
        for(int i = 0; i < numTrainInstances; i++)
        	for(int l = 0; l < numLabels; l++)
                if( YExtended.get(i,l) != GlobalValues.MISSING_VALUE )
                    	YObserved.add(new Tripple(i, l)); 
        
        Collections.shuffle(YObserved);
        
        recNumPoints = 1.0 / (double) numPoints;
        recNumTotalInstances = 1.0 / (double) numTotalInstances;
        recNumTrainInstances = 1.0 / (double) numTrainInstances;
        recNumLabels = 1.0 / (double) numLabels;
        
        Logging.println("numTrainInstances="+numTrainInstances + 
        				", numTotalInstances="+numTotalInstances +
        				", numFeatures="+numPoints +
        				", numLabels="+numLabels + 
        				", latentDim=" + D, LogLevel.DEBUGGING_LOG);  
        
    }

	/*
	 * Train and generate the decomposition 
	 * */

    public double Decompose(Matrix X, Matrix Y) 
    {
    	// initialize the model
    	Initialize(X, Y);
    	int numPredictorsCells = XObserved.size(),
				numTargetCells = YObserved.size();
		
		int predictorCellIdx = 0, derivativeCellIdx = numPredictorsCells/2;
		
		
    	// supervised learning
    	for(int epoch = 0; epoch < maxEpochs; epoch++) 
		{
    		for(int c = 0; c < numPredictorsCells; c++)
    		{
	    		if( alphaR > 0)
	    			UpdatePredictorsLoss(XObserved.get(predictorCellIdx).row, XObserved.get(predictorCellIdx).col);
	            
				if( alphaD > 0)
	    			UpdateDerivativeLoss(XObserved.get(derivativeCellIdx).row, XObserved.get(derivativeCellIdx).col);
			    
			    predictorCellIdx = (predictorCellIdx+1) % numPredictorsCells; 
			    derivativeCellIdx = (derivativeCellIdx+1) % numPredictorsCells; 
		    }
		    
    		if( alphaA > 0)
    			for(int targetCellIdx = 0; targetCellIdx < numTargetCells; targetCellIdx++)
    	    		UpdateTargetLoss( YObserved.get(targetCellIdx).row, YObserved.get(targetCellIdx).col); 
			
    		if( alphaT > 0)
		    	for(int i = 0; i < numTotalInstances; i++)
		    	{
		    		C_err[i] = C[i] - GetComplexity(i);
		    		
		    		for(int j = 0; j < numPoints; j++)
		    			UpdateTotalVariationLoss(i, j);
		    	}
		   
   	        
		    if( Logging.currentLogLevel != LogLevel.PRODUCTION_LOG )
		    	if( epoch % 10 == 0 )
			    {
				    
				    double predictorsMSE = GetPredictorsLoss();
				    double totalVariationLoss = GetTotalVariationLoss();
				    double derivativeLoss = GetDerivativeLoss(); 
				    double trainLogLoss = GetAccuracyLoss(0, numTrainInstances);
				    double testLogLoss = GetAccuracyLoss(numTrainInstances, numTotalInstances);
				    double trainErrorRate = GetErrorRate(0, numTrainInstances);
				    double testErrorRate = GetErrorRate(numTrainInstances, numTotalInstances); 
				    //double testErrorSVM = GetTestErrorSVM();
				    double testErrorNN = GetTestErrorNN();
				    
		            Logging.println("Epoch=" + epoch + 
		            					", LR=" + df.format(predictorsMSE) +
		            					", LT="+ df.format(totalVariationLoss) +
		            					", LD="+ df.format(derivativeLoss) + 
		            					", LA=[" + df.format(trainLogLoss) + " ; " + df.format(testLogLoss) + "]" +
		            					", MCR=[" + df.format(trainErrorRate)  + " ; " + df.format(testErrorRate) + "]" // , LogLevel.DEBUGGING_LOG);
		            					+", MCR=" + df.format(testErrorNN) 
		            					, LogLevel.DEBUGGING_LOG);
		            
			    }
		}

 		// return GetTestErrorNN();
    	return GetErrorRate(numTrainInstances, numTotalInstances);
	}

    // run a support vector machine for classifying the target variable Y
    public double GetTestErrorSVM() 
    {
	    double C = 1;
	    
	    DataSet trainSet = new DataSet();
	    trainSet.LoadMatrixes(U, Y, 0, numTrainInstances);
	    Instances trainSetWeka = trainSet.ToWekaInstances();
	    
	    DataSet testSet = new DataSet();
	    testSet.LoadMatrixes(U, Y, numTrainInstances, numTotalInstances);
	    Instances testSetWeka = testSet.ToWekaInstances();
		 
		SMO svm = WekaClassifierInterface.getRbfSvmClassifier(C, lambdaW);
		try{
			svm.buildClassifier(trainSetWeka);
		}catch(Exception exc){ System.out.println(exc.getMessage());}
		
		Evaluation eval = null;
		try{
			eval = new Evaluation(trainSetWeka);
			eval.evaluateModel(svm, testSetWeka);
		}catch(Exception exc){ System.out.println(exc.getMessage());}

		double errorRate = eval.errorRate();
		
		return errorRate;
		
    }
    
    public double GetTestErrorNN()
    {
	    DataSet trainSet = new DataSet();
	    trainSet.LoadMatrixes(U, Y, 0, numTrainInstances);

	    DataSet testSet = new DataSet();
	    testSet.LoadMatrixes(U, Y, numTrainInstances, numTotalInstances);
	    
	    NearestNeighbour nn = new NearestNeighbour("euclidean");

		return nn.Classify(trainSet, testSet);
		
    }
    
    // update the latent representation to approximate X_{i,j}
    public void UpdatePredictorsLoss(int i, int j)
    {
    	double X_ij, error_ij;
    	
    	X_ij = X.get(i, j);
    	
    	if( X_ij == GlobalValues.MISSING_VALUE ) return;
    	
        error_ij = X_ij - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasU[i] - biasV[j]; 
        
        for(int k = 0; k < D; k++)
        {
        	U.set(i, k, U.get(i, k) - eta * ( -2*alphaR*error_ij*V.get(k,j) + lambdaU*U.get(i, k) ) );
            V.set(k, j, V.get(k,j) - eta * ( -2*alphaR*error_ij*U.get(i, k) + lambdaV*V.get(k,j) ) ); 
    	}
        
        biasU[i] -= eta*-2*alphaR* error_ij;
        biasV[j] -= eta*-2*alphaR* error_ij;
    }
    
    // update the latent nonlinear weights alpha to approximate target label Y_i
    public void UpdateTargetLoss(int i, int l) 
	{
		// get weights by input product
	    double val = MatrixUtilities.getRowByColumnProduct(U, i, W, l) + biasU[i] + biasW[l];
	    
	    double y_hat = Sigmoid.Calculate(val);
        double y = YExtended.get(i, l);
    	
        double u_ik, w_kl, grad_u_ik, grad_w_kl;
        double cte = 1;
            
        for(int k = 0; k < D; k++)
        {
            u_ik =  U.get(i, k);
            w_kl = W.get(k, l);
                
            grad_u_ik =  alphaA*-(y-y_hat)*w_kl + lambdaU*u_ik;
            grad_w_kl = alphaA*-(y-y_hat)*u_ik + lambdaW*w_kl;              
 
            U.set(i, k, u_ik - eta * cte * grad_u_ik);
        	W.set(k, l, w_kl - eta * cte * grad_w_kl);
        }
        
        biasU[i] -= eta * cte * alphaA * -(y-y_hat);
        biasW[l] -= eta * cte * alphaA * -(y-y_hat);
	}
    


    // update the complexity loss corresponding to the i,j-th cell 
    public void UpdateTotalVariationLoss(int i, int j)
    {
    	//double c_err_i = C[i] - GetComplexity(i);
    	double c_err_i = C_err[i];
    	double prefix = -4*alphaT*c_err_i;
    	
    	double grad_v_term = 0;
    	
    	if( j == 0)
    		grad_v_term = RegressPredictor(i, j) - RegressPredictor(i, j+1);
		else if(j == numPoints-1)
			grad_v_term = -RegressPredictor(i, j-1) + RegressPredictor(i, j);
		else
			grad_v_term = -RegressPredictor(i, j-1) + 2*RegressPredictor(i, j) - RegressPredictor(i, j+1);
    	
    	for(int k = 0; k < D; k++)
    	{
    		if( j < numPoints-1 )
    		{
	    		double grad_u_ik = (RegressPredictor(i, j) - RegressPredictor(i, j+1))
	    							*(V.get(k,j) - V.get(k,j+1)); 
	    			
	    		U.set(i, k, U.get(i, k) - eta*(prefix*grad_u_ik + 2*lambdaU*U.get(i, k))); 
    		}
    		
			V.set(k,j, V.get(k,j) - eta*(prefix*grad_v_term*U.get(i, k) + 2*lambdaV*V.get(k, j)));
    	}
    	
    	biasU[i] -= eta*prefix*grad_v_term;
		biasV[j] -= eta*prefix*grad_v_term;
    	
    }
    
    // update the loss arising from 
    public void UpdateDerivativeLoss(int i, int j)
    {
    	// avoid updating the last and first points
    	if(j == 0 || j == numPoints - 1) return;
    	
    	double d_X_ij = X.get(i, j+1) - X.get(i, j);
		double d_X_ijprev = X.get(i, j) - X.get(i, j-1); 
		
		double d_X_ij_hat = RegressPredictor(i, j+1) - RegressPredictor(i, j);
		double d_X_ijprev_hat = RegressPredictor(i, j) - RegressPredictor(i, j-1);
    	
    	double d_err_ij = d_X_ij - d_X_ij_hat;
    	double d_err_ijprev = d_X_ijprev - d_X_ijprev_hat; 
    		
    	double grad_u_ik = 0, grad_v_kj = 0;
    	
    	for(int k = 0; k < D; k++)
    	{
    		grad_u_ik = 2*alphaD*(d_err_ijprev*(-V.get(k, j)+V.get(k, j-1)) + d_err_ij*(-V.get(k, j+1)+V.get(k, j)));
    		
    		grad_v_kj = 2*alphaD*(-d_err_ijprev + d_err_ij)*U.get(i,k);
    		
    		U.set(i, k, U.get(i, k) - eta*(grad_u_ik + lambdaU*U.get(i, k)));
    		V.set(k, j, V.get(k, j) - eta*(grad_v_kj + lambdaV*V.get(k, j))); 
    	}
    	
    	biasU[i] -= eta*alphaD*2*(d_err_ij - d_err_ijprev);
    	biasV[j] -= eta*alphaD*2*(d_err_ij - d_err_ijprev);
    	
    }
        	
	// compute the loss between a ground truth instance target value
	// and an approximation
	
	public double GetAccuracyLoss(int startIndex, int endIndex)
    {
        double YTrainLoss = 0;
        int numObservedCells = 0;
        
        for(int i = startIndex; i < endIndex; i++)
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
	
	// get the error rate of instances in the interval [start,end)
	public double GetErrorRate(int startIndex, int endIndex)
	{
        int numIncorrectClassifications = 0;
        int numInstances = 0;
        
        for(int i = startIndex; i < endIndex; i++)
        {
        	if( Y.get(i) != GlobalValues.MISSING_VALUE )
        	{
				double y = Y.get(i);
				double y_predicted = PredictLabel(i);
				    
				if(y != y_predicted)
					numIncorrectClassifications++;
				
				numInstances++;
        	}
        }
                    
        return (double) numIncorrectClassifications / (double) numInstances; 
	}
	
	// X_hat(i,j)
	public double RegressPredictor(int i, int j)
	{
		return MatrixUtilities.getRowByColumnProduct(U, i, V, j) + biasU[i] + biasV[j];
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
		
		
	// get complexity of X_hat_i
	public double GetComplexity(int i)
	{
		double C_i = 0;
			
		for(int p = 0; p < numPoints-1; p++)
		{
			double val_j = RegressPredictor(i, p);
			double val_j_next = RegressPredictor(i, p+1);
			
			C_i += (val_j - val_j_next)*(val_j - val_j_next);
		}
		
		return C_i;
	}
}
