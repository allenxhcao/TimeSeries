package Classification;

import java.util.ArrayList;
import java.util.List;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import Utilities.Logging;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint.PointStatus;

// implement a SVM interface based on LibSVM

public class SVMInterface extends Classifier
{
	DataSet train, test;
	
	svm_parameter svmParameters;
	public double svmC;
	
    public String kernel;
    
    public double degree, gamma;

	@Override
	protected void TuneHyperParameters(DataSet trainSest) {
		// TODO Auto-generated method stub
		
	}
	
	// the constructor
	public SVMInterface()
	{
		// set kernel to linear by default 
		kernel = "linear";
		gamma = 1;
		degree = 1;
				
	}
	
	// classify the dataset
    public svm_model TrainSVM()
    {
        int numInstances = trainSet.instances.size(),
            numFeatures = trainSet.numFeatures;
        
        //System.out.println("NoIns: " + numInstances + ", NoFeat: " + numFeatures);
        
        // create svm problem data
        svm_problem data = trainSet.ToLibSvmProblem();        
        
        svmParameters = new svm_parameter();
        
        svmParameters.svm_type = svm_parameter.EPSILON_SVR;
        svmParameters.C = svmC;
        
        if(kernel.compareTo("rbf") == 0)
        {
        	svmParameters.degree = (int)degree;
        	svmParameters.kernel_type = svm_parameter.RBF;
        }
        else if(kernel.compareTo("poly") == 0)
        {
        	svmParameters.degree = (int)degree;
        	svmParameters.gamma = gamma;
            svmParameters.kernel_type = svm_parameter.POLY;
        }
        else if(kernel.compareTo("linear") == 0)
            svmParameters.kernel_type = svm_parameter.LINEAR;
        
        
        svmParameters.coef0 = 1;
        svmParameters.eps = 0.000000000001;
        svmParameters.nr_weight = 0;
        
        Logging.println("Exp " + svmParameters.degree, Logging.LogLevel.DEBUGGING_LOG);
        
        svm_model model = svm.svm_train(data, svmParameters);
        
        return model;
    }
    
    public double EvaluatePerformance(svm_model model)
    {
        double noErrors = 0.0;
        
        DataSet ds = testSet;
        
        int numInstances = ds.instances.size(),
                numFeatures = ds.numFeatures;
        
        for(int i = 0; i < numInstances; i++)
        {
            DataInstance testIns = ds.instances.get(i);
            
            List<svm_node> nodes = new ArrayList<svm_node>();
            
            // iterate through the feature values of the instance and create nodes
            // for the nonmissing values only 
            for(int j = 0; j < numFeatures; j++)
            {
            	if( testIns.features.get(j).status != PointStatus.MISSING )
            	{
            	    svm_node node = new svm_node();
	                node.index = j+1;
	                node.value = testIns.features.get(j).value; 
	                
	                nodes.add(node);
            	}
            } 
            
            // flush the list as array
            svm_node [] testInsNodes = new svm_node[nodes.size()];
            for(int k = 0; k < nodes.size(); k++)
            {
            	testInsNodes[k] = nodes.get(k);
            }
            
            double predicted = svm.svm_predict(model,testInsNodes);
            
            double actual = testIns.target;
            
            if( actual != predicted )
            {
                noErrors += 1.0;
            }
        }
        
        return noErrors / ds.instances.size();
    }

	@Override
	public double Classify(DataSet trainDataSet, DataSet testDataSet) 
	{
		trainSet = trainDataSet;
		testSet = testDataSet;
		
		svm_model model = TrainSVM();
		
		return EvaluatePerformance(model);
	}

}
