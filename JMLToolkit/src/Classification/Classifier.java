package Classification;

import DataStructures.DataSet;
import MatrixFactorization.MatrixFactorizationModel;
import java.util.*;
import weka.classifiers.Evaluation;

/*
 * A model for a classifier
 */

public abstract class Classifier 
{
	/*
	 * The original dataset
	 */
	protected DataSet trainSet, testSet;
	
        // a list of hyperparameters
        public List<HyperParameter> hyperParameterDefinitions; 
	/*
	 * Constructor 
	 */
	public Classifier()
	{
            hyperParameterDefinitions = new ArrayList<HyperParameter>();
	}
	
        // set the range values (min,max) of a parameter
        public void SetParameterRange(String paramName, double paramMinValue, double paramMaxValue)
        {
            for(HyperParameter hp : hyperParameterDefinitions)
                if( hp.name.compareTo(paramName) == 0)
                    hp.SetRange(paramMinValue, paramMaxValue); 
        }
        
        // set the range values (min,max) of a parameter
        public HyperParameter GetHyperParameter(String paramName)
        {
            HyperParameter param = null;
            
            for(HyperParameter hp : hyperParameterDefinitions)
                if( hp.name.compareTo(paramName) == 0)
                    param = hp; 
            
            return param;
        }
        
        /*
         * Tune the hyperparameters of the classifier
         */
        protected abstract void TuneHyperParameters(DataSet trainSest);
        
	/*
	 * Run the classifier with the train and test set, parameters will be tuned 
         * from the trainSet
	 */
	public abstract double Classify(DataSet trainSet, DataSet testSet);
        
        /*
         * Optionally run the classifier with a provided values of the parameters
         * 
         * This method is especially useful for running in a batch jobs where we want 
         * to distribute the runs of different parameter sets into a cluster
         */
        //public abstract Evaluation Classify(DataSet trainSet, DataSet testSet, 
          //      List<Double> hyperParametersValues);
	
	
}
