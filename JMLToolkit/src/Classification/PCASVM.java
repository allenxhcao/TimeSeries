/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

import DataStructures.DataSet;
import MatrixFactorization.*;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

/**
 * A matrix classification and an svm on top
 * 
 * @author Josif Grabocka
 */
public class PCASVM extends Classifier
{
    public double svmC, variance;
    public String kernelType;
    public double svmPolynomialKernelExp;
    public double svmRBFKernelGamma;
    public DataSet latentTrainSet, latentTestSet;
    public double avgClassificationTime;
    
    
    public PCASVM()
    { 
        kernelType="rbf";
    }
    
    
    /*
     * Classify the given train and test sets 
     */
    @Override
    public double Classify(DataSet trainSet, DataSet testSet) 
    {
        boolean useSecondClassifier = false;  
        double errorRate = 1;

        Instances trainSetWeka =  trainSet.ToWekaInstances();
        Instances testSetWeka =  testSet.ToWekaInstances(); 

        
        // apply pca filter to data
        try
        {
        	PrincipalComponents pca = new PrincipalComponents();
        	// set input format of pca according to dataset
            pca.setInputFormat(trainSetWeka);
            // set the variance covered
            pca.setVarianceCovered(variance);
            // set other attributes unconstrained
            pca.setMaximumAttributes(-1);
            pca.setCenterData(false);
            pca.setMaximumAttributeNames(-1);
            
            
        	trainSetWeka = Filter.useFilter(trainSetWeka, pca);
        	testSetWeka = Filter.useFilter(testSetWeka, pca);
        }
        catch(Exception exc)
        {
        	Logging.println(exc.getMessage(), LogLevel.ERROR_LOG);
        }
        
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
            svm.buildClassifier(trainSetWeka);
            
            startClassification = System.currentTimeMillis();
    		
            eval = new Evaluation(trainSetWeka);
            eval.evaluateModel(svm, testSetWeka);
        }
        catch(Exception exc)
        {
            exc.printStackTrace();
            Logging.println(exc.getMessage(), Logging.LogLevel.PRODUCTION_LOG);
        }
        
        errorRate = eval.errorRate();
        
        long endClassification = System.currentTimeMillis();
		avgClassificationTime = (double)(endClassification-startClassification)/(double)testSet.instances.size();
			
        
        return errorRate;
    
    }
    
    
    @Override
    protected void TuneHyperParameters(DataSet trainSubset) 
    {
        
    }
}
