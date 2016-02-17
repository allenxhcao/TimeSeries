package Classification;

import Classification.Kernel.KernelType;
import DataStructures.Tripple;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;
import MatrixFactorization.MatrixFactorizationModel;
import MatrixFactorization.MatrixUtilities;
import Regression.LSSVM;
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

public class NonlinearStructurePreservingDecomposition 
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
	
	// the number of latent dimensions
	public int maxEpochs;
	
	// the weights of the reconstruction linear regression models
	public Matrix U;
	
	// the weights of the reconstruction linear regression models of predictors
	public Matrix V;
	double [] biasV;
	
	// the weights of the logistic regression
	LSSVM targetModel;
	
	double [] observedLabels;
	int [] observedLabelIndices;
	
	// a kernel
	public Kernel kernel;
	// the degree of the polynomial kernel
	public int degree;
	
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
	public NonlinearStructurePreservingDecomposition(int factorsDim) 
	{
            eta = 0.001;
            
            lambdaU = 0.001;  
            lambdaV = 0.001;  
            lambdaW = 0.001;  

            XObserved = null;
            
            D = factorsDim;
            
            alphaA = 1.0; 
            alphaR = 1.0; 
    		alphaT = 1.0; 
    		
    		degree = 3;
    		
    		targetModel = null;
    		kernel = null;
            
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
    	
    	// set to the binary labels to +1 and -1
        for(int i = 0; i < numTotalInstances; i++)
        	if(Y.get(i) != 1) Y.set(i, 0, -1.0);
        
        // initialize latent representation of X into latent space represented by U
        U = new Matrix(numTotalInstances, D);
        U.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON);
        
        // initialize a transposed Psi_i
        V = new Matrix(D, numPoints);
        //Psi_i.SetUniqueValue(0.0);
        V.RandomlyInitializeCells(-GlobalValues.SMALL_EPSILON, GlobalValues.SMALL_EPSILON); 

        biasV = new double[numPoints];
        for(int j = 0; j < numPoints; j++)
        	biasV[j] = X.GetColumnMean(j);
        
		// initialize the alphas
        
        kernel = new Kernel(KernelType.Polynomial);
		kernel.type = KernelType.Polynomial;
		kernel.degree = degree;
		
        targetModel = new LSSVM();
        targetModel.kernel = kernel;
		targetModel.lambda = lambdaW;
		
		// store the observed labels, training set only
		observedLabels = new double[numTrainInstances];
		observedLabelIndices = new int[numTrainInstances];
		
		for(int i = 0; i < numTrainInstances; i++)
		{
			observedLabels[i] = Y.get(i);
			observedLabelIndices[i] = i;
		}
		
		
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
    	
    	// supervised learning
    	for(int epoch = 0; epoch < maxEpochs; epoch++) 
		{
    		int numPredictorsCells = XObserved.size();
    		
    		int predictorCellIdx = 0, derivativeCellIdx = numPredictorsCells/2;
    		
    		for(int c = 0; c < numPredictorsCells; c++)
    		{
	    		if( alphaR > 0)
	    			UpdatePredictorsLoss(XObserved.get(predictorCellIdx).row, XObserved.get(predictorCellIdx).col);
	            
			    if( alphaD > 0)
	    			UpdateDerivativeLoss(XObserved.get(derivativeCellIdx).row, XObserved.get(derivativeCellIdx).col);
			    
			    predictorCellIdx = (predictorCellIdx+1) % numPredictorsCells; 
			    derivativeCellIdx = (derivativeCellIdx+1) % numPredictorsCells; 
		    }
    		
    		 
    		if( alphaA > 0 )
    		{
    			// solve the least square svm 
        		targetModel.Train(U, observedLabels, observedLabelIndices);
				//UpdateUTarget(); 
    		}
    		
    	
		   if( alphaT > 0)
		   {
		    	for(int i = 0; i < numTotalInstances; i++)
		    	{
		    		C_err[i] = C[i] - GetComplexity(i);
		    		
		    		for(int j = 0; j < numPoints; j++)
		    			UpdateTotalVariationLoss(i, j);
		    	}
		   }
		    
		   
		    if( Logging.currentLogLevel != LogLevel.PRODUCTION_LOG )
		    	if( epoch % 10 == 0 )
			    {
				    
				    double predictorsMSE = GetPredictorsLoss();
				    double totalVariationLoss = GetTotalVariationLoss();
				    double derivativeLoss = GetDerivativeLoss(); 
				    double trainLogLoss = GetTargetLoss(0, numTrainInstances);
				    double testLogLoss = GetTargetLoss(numTrainInstances, numTotalInstances);
				    double trainErrorRate = GetErrorRate(0, numTrainInstances);
				    double testErrorRate = GetErrorRate(numTrainInstances, numTotalInstances); 
				    
		            Logging.println("Epoch=" + epoch + 
		            					", LR=" + df.format(predictorsMSE) +
		            					", LT="+ df.format(totalVariationLoss) +
		            					", LD="+ df.format(derivativeLoss) + 
		            					", LA=[" + df.format(trainLogLoss) + " ; " + df.format(testLogLoss) + "]" +
		            					", MCR=[" + df.format(trainErrorRate)  + " ; " + df.format(testErrorRate) + "]", LogLevel.DEBUGGING_LOG);

		            					//LogLevel.DEBUGGING_LOG);
		            
			    }
		}

 		return GetErrorRate(numTrainInstances, numTotalInstances);
	}

    // run a support vector machine for classifying the target variable Y
 
    // update the latent representation to approximate X_{i,j}
    public void UpdatePredictorsLoss(int i, int j)
    {
    	double X_ij, error_ij;
    	
    	X_ij = X.get(i, j);
    	
    	if( X_ij == GlobalValues.MISSING_VALUE ) return;
    	
        error_ij = X_ij - MatrixUtilities.getRowByColumnProduct(U, i, V, j) - biasV[j]; 
        
        for(int k = 0; k < D; k++)
        {
        	U.set(i, k, U.get(i, k) - eta * ( -2*alphaR*error_ij*V.get(k,j) + lambdaU*U.get(i, k) ) );
            V.set(k, j, V.get(k,j) - eta * ( -2*alphaR*error_ij*U.get(i, k) + lambdaV*V.get(k,j) ) ); 
    	}
        
        biasV[j] = biasV[j] - eta*-2*alphaR* error_ij;
    }
    
    // update the latent nonlinear weights alpha to approximate target label Y_i
    public void UpdateUTarget() 
	{

		int i=0,l=0;
		double grad = 0, kernelGrad = 0;
		
		for(int iIndex = 0; iIndex < observedLabelIndices.length; iIndex++)
		{
			
			for(int lIndex = 0; lIndex < observedLabelIndices.length; lIndex++)
			{
				i = observedLabelIndices[iIndex];
				l = observedLabelIndices[lIndex];
				
				for(int k = 0; k < D; k++)
				{
					// update U(i,k)
					kernelGrad = ComputeKernelGradient(i, l, i, k); 
					
					
					if( kernelGrad != 0 && !Double.isInfinite(kernelGrad)
							&& !Double.isNaN(kernelGrad))
					{
						//System.out.println(kernelGrad);
						
						grad = alphaA * (1.0/(2*lambdaW)) * targetModel.alphas[iIndex] 
							* targetModel.alphas[lIndex]
								* kernelGrad;
								//-lambdaU*U.get(i,k);
						
						U.set(i, k,  U.get(i,k) + eta*grad);
					}
					
					// update U(l,k)
					kernelGrad = ComputeKernelGradient(i, l, l, k);
					
					if( kernelGrad != 0 && !Double.isInfinite(kernelGrad)
							&& !Double.isNaN(kernelGrad))
					{
						grad = alphaA * (1.0/(2*lambdaW))* targetModel.alphas[iIndex] 
								* targetModel.alphas[lIndex]
									* kernelGrad;
									//- lambdaU*U.get(l,k);
						
						U.set(l, k,  U.get(l,k) + eta*grad);
					}
					
				}
			}
		}		
	}
    
 // compute the kernel gradient { d K(i,l) / d U(r,k) } 
 	public double ComputeKernelGradient(int i, int l, int r, int k)
 	{
 		double grad = 0;
 		
 		if( kernel.type == KernelType.Linear)
 		{
 			if( r == i )
 				grad = U.get(l, k);
 			else if(r == l)
 				grad = U.get(i, k);	
 		}
 		else if( kernel.type == KernelType.Polynomial)
 		{
 			if( r == i || r == l )
 			{
 				grad = kernel.degree * 
 						Utilities.StatisticalUtilities.Power(U.RowDotProduct(i, l)+1, kernel.degree-1); 
 				
 				if( r == i )
 					grad *= U.get(l, k);
 				else if(r == l)
 					grad *= U.get(i, k);
 			}			
 		}

 		else if( kernel.type == KernelType.Gaussian)
 		{
 			if( r == i || r == l) 
 			{
 				grad = (r == i) ? 1 : -1;
 				
 				double kerVal = Math.exp(-kernel.EuclideanDistance(U.getRow(i), U.getRow(l))/kernel.sig2);
 				
 				grad *= -2*((U.get(i,k) - U.get(l,k))/kernel.sig2) * kerVal;  
 				
 				//System.out.println(grad + ", diff: " + (U.get(i,k) - U.get(l,k)) + ", kernel" + kerVal + ", sig2" + kernel.sig2);
 				
 			}
 		}

 		else if( kernel.type == KernelType.Gaussian)
 		{
 			
 		}
 		
 		return grad;
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
    	
		biasV[j] = biasV[j] - eta*prefix*grad_v_term;
    	
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
    	
    	biasV[j] = biasV[j] - eta*alphaD*2*(d_err_ij - d_err_ijprev);
    	
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
		return MatrixUtilities.getRowByColumnProduct(U, i, V, j) + biasV[j];
	}
	
	public double PredictLabel(int i)
    {
    	double y_hat_i = targetModel.PredictInstance(U.getRow(i), observedLabelIndices);
    
    	return y_hat_i >= 0 ? 1.0 : -1.0;
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
	
	public double GetTargetLoss(int startIndex, int endIndex )
	{
		double mse = 0;
		int numInstances = 0;
		
		for( int i = startIndex; i < endIndex; i++)
		{
			double err = 
					Y.get(i) - targetModel.PredictInstance(U.getRow(i), observedLabelIndices);
			
			mse += err*err;
			numInstances++;
		}
		
		return  mse/numInstances;
	}
	
}
