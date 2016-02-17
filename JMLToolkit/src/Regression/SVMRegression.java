package Regression;

import java.util.ArrayList;
import java.util.List;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import Utilities.BlindPrint;
import Utilities.GlobalValues;
import Utilities.Logging;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint.PointStatus;
import DataStructures.Matrix;

// implement a SVM interface based on LibSVM

public class SVMRegression
{
	DataSet train, test;
	
	svm_parameter svmParameters;
	public double svmC;
	
    public String kernel;
    
    public double degree, gamma;

	
	
	// the constructor
	public SVMRegression()
	{
		// set kernel to linear by default 
		kernel = "linear";
		gamma = 1;
		degree = 1;
				
	}
	
	// classify the dataset
    public svm_model TrainSVM(Matrix predictors, double [] trainTargets)
    {
        int numInstances =predictors.getDimRows(),
            numFeatures = predictors.getDimColumns();
        
        // create svm problem data from the predictors, without targets
        svm_problem data = predictors.ToLibSvmProblem();        
        // assign the targets
        data.y = trainTargets;
        
        svmParameters = new svm_parameter();
        
        svmParameters.svm_type = svm_parameter.EPSILON_SVR;
        svmParameters.C = svmC;
        
        if(kernel.compareTo("gaussian") == 0)
        {
        	svmParameters.degree = (int)degree;
        	svmParameters.gamma = gamma;
        	svmParameters.kernel_type = svm_parameter.RBF;
        }
        else if(kernel.compareTo("polynomial") == 0)
        {
        	svmParameters.degree = (int)degree;
        	svmParameters.gamma = 1; 
            svmParameters.kernel_type = svm_parameter.POLY;
            
            //System.out.println("polynomial - degree" + svmParameters.degree);
        }
        else if(kernel.compareTo("linear") == 0)
            svmParameters.kernel_type = svm_parameter.LINEAR;
        
        
        svmParameters.coef0 = 1;
        svmParameters.eps = 0.0001;
        svmParameters.nr_weight = 0; 
        
        svm.svm_set_print_string_function(new BlindPrint());
        
        svm_model model = svm.svm_train(data, svmParameters);
        
        return model;
    }
    
    public double EvaluatePerformance(svm_model model, Matrix testPredictors, double [] testTargets)
    {
        double mse = 0.0;
        
        
        int numTestInstances = testPredictors.getDimRows(),
                numFeatures = testPredictors.getDimColumns();
        
        for(int i = 0; i < numTestInstances; i++)
        {
            List<svm_node> nodes = new ArrayList<svm_node>();
            
            // iterate through the feature values of the instance and create nodes
            // for the nonmissing values only 
            for(int j = 0; j < numFeatures; j++)
            {
            	if( testPredictors.get(i, j) != GlobalValues.MISSING_VALUE )
            	{
            	    svm_node node = new svm_node();
	                node.index = j+1;
	                node.value = testPredictors.get(i, j);  
	                
	                nodes.add(node);
            	}
            } 
            
            // flush the list as array
            svm_node [] testInsNodes = new svm_node[nodes.size()];
            for(int k = 0; k < nodes.size(); k++)
            {
            	testInsNodes[k] = nodes.get(k);
            }
            
            double predictedTarget = svm.svm_predict(model,testInsNodes);
            
            double error = testTargets[i] - predictedTarget;
            
            //System.out.println("true="+testTargets[i] + ", predicted="+ predictedTarget); 
            
            mse += error*error;
        }
        
        return mse / numTestInstances;
    }

	public double Regress(Matrix trainPredictors, double [] trainTargets,
							Matrix testPredictors, double [] testTargets) 
	{
		svm_model model = TrainSVM(trainPredictors, trainTargets);
		
		return EvaluatePerformance(model, testPredictors, testTargets);
	}

}
