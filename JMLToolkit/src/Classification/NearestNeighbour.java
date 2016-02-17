/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import TimeSeries.CIDDistance;
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
public class NearestNeighbour extends Classifier
{
    String distanceOp;
    
    // the list of the nearest neihbours and the distances
    
    /*
     * constructor with the distance operator (DTW, Euclidean, etc ..) and the 
     * number of nearest neighbours
     */
    public NearestNeighbour(String distanceOperator)
    {
        distanceOp = distanceOperator;
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
                DataInstance trainInstance = trainSet.instances.get(j);
                
                // compute the distance
                double distance = Double.MAX_VALUE;
                
                if( distanceOp.compareTo("dtw") == 0 ) 
                    distance = DTW.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("euclidean") == 0 )
                    distance = EuclideanDistance.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("ga") == 0 )
                	distance = GlobalAlignmentKernel.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("cleandtw") == 0 )
                	distance = CleanDTW.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("cid") == 0 )
                	distance = CIDDistance.getInstance().CalculateDistance(trainInstance, testInstance);
 
                
                // if there are less then required candidates then add it as a candidate directly
                if( distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestLabel = trainInstance.target;
                    nearestIndex = j;
                }
            }
            
            if( realLabel == nearestLabel )
            {
                correct += 1.0;
                
                //System.out.println(i + " - " + nearestIndex + ", dist="+nearestDistance + ", label=" + nearestLabel);
            }
            
            //Logging.println("Real: " + realLabel + ", Nearest: " + nearestLabel, LogLevel.DEBUGGING_LOG);
        }
        
        //Logging.println("Correct classifications: " + correct, LogLevel.DEBUGGING_LOG);
        
        // return the error rate
        return 1.0 - (correct/ (double)testSet.instances.size());
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

            if( distanceOp.compareTo("dtw") == 0 )
                distance = DTW.getInstance().CalculateDistance(trainInstance, ins);
            else if( distanceOp.compareTo("euclidean") == 0 )
                distance = EuclideanDistance.getInstance().CalculateDistance(trainInstance, ins);
            else if( distanceOp.compareTo("ga") == 0 )
                distance = GlobalAlignmentKernel.getInstance().CalculateDistance(trainInstance, ins);


            // if there are less then required candidates then add it as a candidate directly
            if( distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestLabel = trainInstance.target;
            }
        }
        
        return nearestLabel;
    }
    
    
    public double ClassifyLeaveOneOut(DataSet trainSet) 
    {   
        // correct classifications
        double correct = 0;
        
        String dsNameFull=trainSet.name;
        String dsName = "";        
        DTW.getInstance().dsName = dsName;

        //Logging.println("Train Instances: " + trainSet.instances.size() + ", Test Instances: " + testSet.instances.size(), LogLevel.DEBUGGING_LOG);
        
        // for every train instances 
        for(int i = 0;  i < trainSet.instances.size(); i++)
        {
            DataInstance testInstance = trainSet.instances.get(i);
            double realLabel = testInstance.target;
            
            double nearestLabel = 0;
            double nearestDistance = Double.MAX_VALUE;
            int nearestIndex = 0;

            // iterate through trainin instances and find the closest neighbours
            for(int j = 0; j < trainSet.instances.size(); j++)
            {   
            	// avoid comparing against itself
            	if( i == j) continue;
            	
                DataInstance trainInstance = trainSet.instances.get(j);
                
                // compute the distance
                double distance = Double.MAX_VALUE;
                
                if( distanceOp.compareTo("dtw") == 0 ) 
                    distance = DTW.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("euclidean") == 0 )
                    distance = EuclideanDistance.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("ga") == 0 )
                	distance = GlobalAlignmentKernel.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("cleandtw") == 0 )
                	distance = CleanDTW.getInstance().CalculateDistance(trainInstance, testInstance);
                else if( distanceOp.compareTo("cid") == 0 )
                	distance = CIDDistance.getInstance().CalculateDistance(trainInstance, testInstance); 
 
                
                // if there are less then required candidates then add it as a candidate directly
                if( distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestLabel = trainInstance.target;
                    nearestIndex = j;
                }
            }
            
            if( realLabel == nearestLabel )
            {
                correct += 1.0;
                
                //System.out.println(i + " - " + nearestIndex + ", dist="+nearestDistance + ", label=" + nearestLabel);
            }
            
            //Logging.println("Real: " + realLabel + ", Nearest: " + nearestLabel, LogLevel.DEBUGGING_LOG);
        }
        
        //Logging.println("Correct classifications: " + correct, LogLevel.DEBUGGING_LOG);
        
        // return the error rate
        return 1.0 - (correct/ (double)trainSet.instances.size());
    }
    
}
