/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

import DataStructures.DataSet;
import MatrixFactorization.*;
import Utilities.Logging;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

/**
 * A matrix classification and an svm on top
 * 
 * @author Josif Grabocka
 */
public class MFSVM extends Classifier
{
    public int latentDimensions;
    public double lambda, learningRate, svmC, alpha;
    public double updateFractionPerEpoch;
    public String kernelType;
    public double svmPolynomialKernelExp;
    public double svmRBFKernelGamma;
    public double warpingWindowFraction;
    public DataSet latentTrainSet, latentTestSet;
    public int k, maxEpocs;
    public boolean enableTimeWarping;
    public boolean factorizeTrainOnly;
    public boolean unsupervisedFactorization;
    public double maxMargin;
    public double avgClassificationTime;
    
    public String factorizationDescription;
    
    public MFSVM()
    { 
    	updateFractionPerEpoch = 0.3;
    	alpha = 0;
        kernelType="rbf";
        enableTimeWarping = false;
        unsupervisedFactorization = false;
        maxMargin = 1;
    }
    
    
    /*
     * Classify the given train and test sets 
     */
    @Override
    public double Classify(DataSet trainSet, DataSet testSet) 
    {
        latentTrainSet = new DataSet();
        latentTestSet = new DataSet();
        boolean useSecondClassifier = false;  
        double errorRate = 1;
        
    	FactorizationsCache.getInstance().GetLatent(factorizationDescription, learningRate, lambda, latentDimensions, 
    			latentTrainSet, latentTestSet, alpha, maxEpocs);
    	
    	if( latentTrainSet.instances.size() <= 0 && latentTestSet.instances.size() <= 0)
    	{
    		if(enableTimeWarping)
    		{
    			WarpedMF wmf = new WarpedMF( latentDimensions );
                wmf.lambdaU = wmf.lambdaV = wmf.lambdaW = lambda;
    			wmf.learningRate = learningRate;
    			wmf.alpha = alpha;    			
    			wmf.stochasticUpdatesPercentage = updateFractionPerEpoch;
    			wmf.maxEpocs = maxEpocs;
    			wmf.warpingWindowFraction = warpingWindowFraction;
    			wmf.maxMargin = maxMargin;
    			errorRate = wmf.Factorize(trainSet, testSet, latentTrainSet, latentTestSet);
    			
    		}
    		else if (unsupervisedFactorization == true)
    		{
	    		MatrixFactorization mf = new MatrixFactorization( latentDimensions );
		        mf.lambdaU = mf.lambdaV = lambda;
		        mf.learningRate = learningRate;
	        	mf.alpha = alpha;
	        	mf.maxEpocs = maxEpocs;
	        	mf.Factorize(trainSet, testSet, latentTrainSet, latentTestSet);
	        	useSecondClassifier = true;
    		}
    		else
    		{
    			SupervisedMatrixFactorization mf = new SupervisedMatrixFactorization( latentDimensions );
		        mf.lambdaU = mf.lambdaV = mf.lambdaW = lambda;
		        mf.learningRate = learningRate;
		        mf.stochasticUpdatesPercentage = updateFractionPerEpoch;
	        	mf.alpha = alpha;
	        	mf.maxEpocs = maxEpocs;
	        	mf.maxMargin = maxMargin;
	        	errorRate = mf.Factorize(trainSet, testSet, latentTrainSet, latentTestSet);
	        	
    		}
    		
	        FactorizationsCache.getInstance().SaveLatent(latentTrainSet, latentTestSet, 
	        		factorizationDescription, learningRate, lambda, latentDimensions, alpha, maxEpocs);
    	}
    	
        if(latentTrainSet == null || latentTestSet == null)
            Logging.println("Latent Datasets empty", Logging.LogLevel.DEBUGGING_LOG);

        
        if( useSecondClassifier || alpha == 1.0 || unsupervisedFactorization == true)
        {
        	
	        Instances latentTrainSetWeka =  latentTrainSet.ToWekaInstances();
	        Instances latentTestSetWeka =  latentTestSet.ToWekaInstances(); 
	
	        // create a svm classifier with specified hyperparameter parameters
	        // the rest are weka-defaults
	        SMO svm = null;
	        
	        if( kernelType.compareTo("polynomial") == 0 )
	        {
	            svm = WekaClassifierInterface.getPolySvmClassifier(svmC,svmPolynomialKernelExp);
	        }
	        else if( kernelType.compareTo("rbf") == 0 )
	        {
	            svm = WekaClassifierInterface.getRbfSvmClassifier(svmC,svmRBFKernelGamma);
	        }
	        else
	        {
	            Logging.println("MFSVM:: Kernel Type not recognized " + kernelType, Logging.LogLevel.ERROR_LOG);
	            return 0;
	        }
	        
	        long startClassification = 0;
	        
	        Evaluation eval = null;
	        try
	        {
	            svm.buildClassifier(latentTrainSetWeka);
	            
	            startClassification = System.currentTimeMillis();
	    		
	            eval = new Evaluation(latentTrainSetWeka);
	            eval.evaluateModel(svm, latentTestSetWeka);
	        }
	        catch(Exception exc)
	        {
	            exc.printStackTrace();
	            Logging.println(exc.getMessage(), Logging.LogLevel.PRODUCTION_LOG);
	        }
	        
	        errorRate = eval.errorRate();
	        
	        long endClassification = System.currentTimeMillis();
			avgClassificationTime = (double)(endClassification-startClassification)/(double)testSet.instances.size();
			
        }
        
        return errorRate;
    
    }
    
    
    public MatrixFactorizationModel FactorizeOnly(DataSet trainSet) 
    {
    	MatrixFactorizationModel factorizationModel = null;
    	
		if(enableTimeWarping)
		{
			WarpedMF wmf = new WarpedMF( latentDimensions );
			wmf.lambdaU = wmf.lambdaV = wmf.lambdaW = lambda;
			wmf.learningRate = learningRate;
			wmf.stochasticUpdatesPercentage = updateFractionPerEpoch;
			wmf.alpha = 0.7;
			wmf.Factorize(trainSet);
			factorizationModel = wmf;
		}
		
		else
		{
			
			SupervisedMatrixFactorization mf = new SupervisedMatrixFactorization( latentDimensions );
	        mf.lambdaU = mf.lambdaV = mf.lambdaW= lambda;
	        mf.learningRate = learningRate;
	        mf.stochasticUpdatesPercentage = updateFractionPerEpoch;
	        
	        mf.Factorize(trainSet);
	        factorizationModel = mf;
		}
    	
    	return factorizationModel;
    }
    

    @Override
    protected void TuneHyperParameters(DataSet trainSubset) 
    {
        
    }
}
