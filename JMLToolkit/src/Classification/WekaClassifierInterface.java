package Classification;

import java.util.Random;

import Utilities.Logging;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import DataStructures.DataSet; 
import weka.classifiers.functions.supportVector.DTWKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;

public class WekaClassifierInterface 
{
    public static final int FILTER_NORMALIZE = 0;
    /** The filter to apply to the training data: Standardize */
    public static final int FILTER_STANDARDIZE = 1;
    /** The filter to apply to the training data: None */
    public static final int FILTER_NONE = 2;
    /** The filter to apply to the training data */
    public static final Tag[] TAGS_FILTER =
    {
    new Tag(FILTER_NORMALIZE, "Normalize training data"),
    new Tag(FILTER_STANDARDIZE, "Standardize training data"),
    new Tag(FILTER_NONE, "No normalization/standardization"),
    };
	 
        public static Evaluation Classify(DataSet trainingSet, DataSet testingSet)
	{
            Instances trainingSetWeka =  trainingSet.ToWekaInstances();
            Instances testingSetWeka =  testingSet.ToWekaInstances();
		
            return Classify(trainingSetWeka, testingSetWeka);
        }
        
	public static Evaluation Classify(Instances trainingSetWeka, Instances testingSetWeka)
	{
		Evaluation evaluation = null; 
		
		 
		try
		{
		        SMO classifier = getPolynomialKernelSVM();
                        
			classifier.buildClassifier(trainingSetWeka);
			
			evaluation = new Evaluation(trainingSetWeka);
			evaluation.evaluateModel(classifier, testingSetWeka);
			
			//Logging.println( evaluation.toSummaryString(), Logging.DEBUGGING_LOG );
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
		
		//System.out.println( wekaInstances.toString() );
		
		
		return evaluation;
	}
	
	
	public static void Classify(DataSet trainingSet, int crossValidationFolds)
	{
		Instances trainingSetWeka =  trainingSet.ToWekaInstances();
		
		try 
		{
			SMO svm = getPolynomialKernelSVM();
			
                         
                        
			//IBk knn = new IBk(1);
			//knn.buildClassifier(trainingSetWeka);
			
			/*
			J48 j48 = new J48();
			j48.buildClassifier(trainingSetWeka);
			*/
			
			Evaluation evaluation = new Evaluation(trainingSetWeka);
			evaluation.crossValidateModel(svm, trainingSetWeka, crossValidationFolds, new Random(1));
			
                        
                        
			System.out.println( evaluation.toSummaryString() );
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
		
		//System.out.println( wekaInstances.toString() );
				
	}
        
        /*
         * A polynomial kernel SVM 
         */
        public static SMO getPolynomialKernelSVM()
        {
            return getPolySvmClassifier(1, 3);
        }
        
        public static SMO getPolySvmClassifier(double svmComplexity, double polyKernelExponent)
        {
            SMO classifier = new SMO();
			
            classifier.setBuildLogisticModels(false);
            classifier.setC(svmComplexity);
            classifier.setChecksTurnedOff(false);
            classifier.setDebug(false);
            classifier.setEpsilon(1.0E-12); 
            classifier.setFilterType(new SelectedTag( FILTER_NONE, TAGS_FILTER));
            PolyKernel pk = new PolyKernel(); 
                pk.setCacheSize(250007);
                pk.setChecksTurnedOff(false);
                pk.setDebug(false);                         
                pk.setExponent(polyKernelExponent);
                pk.setUseLowerOrder(false);
            classifier.setKernel(pk);
            
            classifier.setNumFolds(-1);
            classifier.setRandomSeed(1);
            classifier.setToleranceParameter(0.001);

            return classifier;
        }
        
        
        public static SMO getRbfSvmClassifier(double svmComplexity, double gamma)
        {
            SMO classifier = new SMO();
			
            classifier.setBuildLogisticModels(false);
            classifier.setC(svmComplexity);
            classifier.setChecksTurnedOff(false);
            classifier.setDebug(false);
            classifier.setEpsilon(1.0E-12); 
            classifier.setFilterType(new SelectedTag( FILTER_NONE, TAGS_FILTER));
            
            RBFKernel rk = new RBFKernel(); 
            rk.setGamma(gamma);
            
            rk.setChecksTurnedOff(false);
            rk.setDebug(false);   
            rk.setCacheSize(250007); 
            
            classifier.setKernel(rk);
            
            classifier.setNumFolds(-1);
            classifier.setRandomSeed(1);
            classifier.setToleranceParameter(0.001);

            return classifier;
        }
        
        public static SMO getDtwSvmClassifier(double svmComplexity)
        {
            SMO classifier = new SMO();
			
            classifier.setBuildLogisticModels(false);
            classifier.setC(svmComplexity);
            classifier.setChecksTurnedOff(false);
            classifier.setDebug(false);
            classifier.setEpsilon(1.0E-12); 
            classifier.setFilterType(new SelectedTag( FILTER_NONE, TAGS_FILTER));
            
            DTWKernel dk = new DTWKernel(); 
                dk.setCacheSize(250007);
                dk.setChecksTurnedOff(false);
                dk.setDebug(false);                         
            classifier.setKernel(dk);
            
            classifier.setNumFolds(-1);
            classifier.setRandomSeed(1);
            classifier.setToleranceParameter(0.001);

            return classifier;
        }
}
