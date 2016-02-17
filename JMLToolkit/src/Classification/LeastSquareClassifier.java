/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import TimeSeries.CleanDTW;
import TimeSeries.DTW;
import TimeSeries.DistanceOperator;
import TimeSeries.EuclideanDistance;
import TimeSeries.GlobalAlignmentKernel;
import TimeSeries.TotalVariation;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import weka.classifiers.Evaluation;

/**
 * The nearest neighbour classifier
 * 
 * @author Josif Grabocka
 */
public class LeastSquareClassifier extends Classifier
{
    double lambda;
    
    // the list of the nearest neihbours and the distances
    
    /*
     * constructor with the distance operator (DTW, Euclidean, etc ..) and the 
     * number of nearest neighbours
     */
    public LeastSquareClassifier(double regularizationParameter)
    {
        lambda = regularizationParameter;
    }

    @Override
    protected void TuneHyperParameters(DataSet trainSest) 
    {
        // no parameter to tune
    }

    @Override
    public double Classify(DataSet trainSet, DataSet testSet) 
    {   
        // correct classifications
        double correct = 0;
        
        String dsNameFull=trainSet.name;
        String dsName = "";
        
        if(dsNameFull.indexOf("TRAIN") >= 0)
            dsName = dsNameFull.split("_TRAIN")[0];
        else if(dsNameFull.indexOf("TEST") >= 0)
            dsName = dsNameFull.split("TEST")[0];
        else if(dsNameFull.indexOf("VALIDATION") >= 0)
            dsName = dsNameFull.split("VALIDATION")[0];
        
        DTW.getInstance().dsName = dsName;

        //Logging.println("Train Instances: " + trainSet.instances.size() + ", Test Instances: " + testSet.instances.size(), LogLevel.DEBUGGING_LOG);
        
        // for every test instance 
        for(int i = 0;  i < testSet.instances.size(); i++)
        {
            DataInstance testInstance = testSet.instances.get(i);
            double realLabel = testInstance.target;
            
            double nearestLabel = 0;
            double nearestDistance = Double.MAX_VALUE;
            int nearestIndex = 0;

            // iterate through trainin instances and find the closest neighbours
            for(int j = 0; j < trainSet.instances.size(); j++)
            {   
            }
            
        }
        
        
        return 1 - (correct/ (double)testSet.instances.size());
    }
    
    

    public double Classify(DataSet trainSet, DataInstance ins) 
    {          
        
        double nearestLabel = 0;
        double nearestDistance = Double.MAX_VALUE;

        // iterate through trainin instances and find the closest neighbours
        for(int j = 0; j < trainSet.instances.size(); j++)
        {   
            DataInstance trainInstance = trainSet.instances.get(j);

            // compute the distance
            double distance = Double.MAX_VALUE;

          
            // if there are less then required candidates then add it as a candidate directly
            if( distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestLabel = trainInstance.target;
            }
        }
        
        return nearestLabel;
    }
    
}
