/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import MatrixFactorization.CollaborativeImputation;
import TimeSeries.Distorsion;
import TimeSeries.TransformationFieldsGenerator;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import libsvm.*;

/**
 *
 * @author josif
 */
public class FastInvariantSVM 
{
    DataSet trainSet;
    DataSet testSet;
    // the svm parameters
    svm_parameter svmParameters;
    
    public double svmC;
    public double svmPKExp;
    public String kernel;
    public double svmRKGamma;
    
    public double eps;
    
    // used for reporting the computational time 
    public long elapsedSVMTime, elapsedISVMTime;
    
    
    public void InvariantSVM()
    {
        svmParameters = new svm_parameter();
        elapsedSVMTime = 0;
        elapsedISVMTime = 0;
        
    }
    
    
    // classify the dataset
    public svm_model TrainSVM()
    {
        int numInstances = trainSet.instances.size(),
            numFeatures = trainSet.numFeatures;
        
        //System.out.println("NoIns: " + numInstances + ", NoFeat: " + numFeatures);
        
        // create svm problem data
        svm_problem data = new svm_problem();
        
        data.l = numInstances;
        data.x = new svm_node[numInstances][numFeatures];
        data.y = new double[numInstances];
                
        // iterate through all the instances
        for(int i = 0; i < numInstances; i++)
        {
            DataInstance ins = trainSet.instances.get(i);
            
            // iterate through the feature values of the instance and create nodes
            // for the nonmissing values only 
            for(int j = 0; j < numFeatures; j++)
            {
                data.x[i][j] = new svm_node();
                data.x[i][j].index = j+1;
                data.x[i][j].value = ins.features.get(j).value; 
            } 
            
            data.y[i] = ins.target;
            
            
        }
        
        
        
        svmParameters = new svm_parameter();
        
        svmParameters.svm_type = svm_parameter.C_SVC;
        svmParameters.C = svmC;
        svmParameters.kernel_type = kernel.compareTo("rbf")==0 ? svm_parameter.RBF 
                                        : svm_parameter.POLY;
        svmParameters.degree = (int)svmPKExp;
        svmParameters.gamma = svmRKGamma;
        svmParameters.coef0 = 1;
        svmParameters.eps = 0.000000000001;
        svmParameters.nr_weight = 0;
        
        
        
        //svmParameters.kernel_type = svm_parameter.PRECOMPUTED;
        
        
        Logging.println("Exp " + svmParameters.degree, Logging.LogLevel.DEBUGGING_LOG);
        
        svm_model model = svm.svm_train(data, svmParameters);
        
        return model;
    }
    
    // create virtual support vectors
    public void CreateVirtualSV(svm_model model)
    {
        int numFeatures = trainSet.numFeatures;
        
        // replicate the support vector
        
        Logging.println("SV fraction: " + (double)model.l / (double)trainSet.instances.size(), Logging.LogLevel.DEBUGGING_LOG);
       
        // store the offset to access the support vector
        // it is stored as e.g.: assume class 1 has 20 sv and label 2 has 30 sv
        // then there will be 50 svs order s.t. the first 20 have label 1
        int svIndexOffset = 0;
        
        DataSet svDS = new DataSet();
        svDS.name = trainSet.name;
        svDS.numFeatures = trainSet.numFeatures;
        
        
        // iterate through all the classes
        for(int i = 0; i < model.nSV.length; i++)
        {
            // get the label of the class and the number of support vectors
            int noSVPerClass = model.nSV[i];
            double label = (double)model.label[i];
                        
            // replicate all support vectors of the class
            for(int j = svIndexOffset; j < svIndexOffset+noSVPerClass; j++)
            {
                DataInstance instance = new DataInstance();
                
                for(int k = 0; k < numFeatures; k++)
                {
                    instance.features.add(new FeaturePoint(model.SV[j][k].value));
                }
                
                		
                instance.target = label; 
                instance.name = String.valueOf(i);
                
                //Logging.println("Label: " + label + ", sv_coeff: " + model.sv_coef[0][j], LogLevel.DEBUGGING_LOG);
                 
                List<DataInstance> distortedInstances =  TransformationFieldsGenerator.getInstance().Transform(instance);
                
                for( DataInstance di : distortedInstances )
                {
                    svDS.instances.add(di);
                }
          
            }
            
            
            // update the index for the first svm of the next class
            svIndexOffset += noSVPerClass;
        }

        trainSet.AppendDataSet(svDS); 
    }
    
    // evaluate the performance of the learned model on the test set
    public double EvaluatePerformance(svm_model model, boolean useValidationSet)
    {
        double noErrors = 0.0;
        
        DataSet ds = testSet;
        
        int numInstances = ds.instances.size(),
                numFeatures = ds.numFeatures;
        
        for(int i = 0; i < numInstances; i++)
        {
            DataInstance testIns = ds.instances.get(i);
            
            // create node representation of the test instance
            svm_node [] testInsNodes = new svm_node[numFeatures];
            for(int j = 0; j < numFeatures; j++)  
            {
                testInsNodes[j] = new svm_node();
                testInsNodes[j].index = j+1;
                testInsNodes[j].value = testIns.features.get(j).value;
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
    
    class BlindPrint implements svm_print_interface
    {
        @Override
        public void print(String string) {
            
        }
    }
    
    public double[] Classify(DataSet trainDS, DataSet testDS)
    {
        
        trainSet = trainDS;
        testSet = testDS;
        
        //svm.rand.setSeed(0);
        
        svm.svm_set_print_string_function(new BlindPrint());
        
        long start = System.currentTimeMillis();
        
        trainSet.ReadNominalTargets();
        
        svm_model svmModel = TrainSVM();
        double errorRate = EvaluatePerformance(svmModel, false);
        Logging.println("Initial error: " + errorRate, Logging.LogLevel.DEBUGGING_LOG);
        double initialErrorRate = errorRate;
        
        
        long end = System.currentTimeMillis();
        elapsedSVMTime = end - start;
        
        if( eps == 0.0)
            return new double[]{initialErrorRate, initialErrorRate};
        
        CreateVirtualSV(svmModel); 
        svmModel = TrainSVM();
        errorRate = EvaluatePerformance(svmModel, false);

        //DumpDataSet("vsv");
        
        Logging.println("VSV1 error: " + errorRate, Logging.LogLevel.DEBUGGING_LOG);

        end = System.currentTimeMillis();
        elapsedISVMTime = end - start;
        
        
        
        return new double[]{errorRate, initialErrorRate};
    }
    
    
    public void DumpDataSet(String str)
    {
        String folder="/home/josif/Documents/invariant svm/examples/";
        
        trainSet.SaveToArffFile(folder+trainSet.name+"_"+str+".arff"); 
        testSet.SaveToArffFile(folder+testSet.name+".arff"); 
    }
    
}
