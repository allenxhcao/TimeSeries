/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package TimeSeries;

import DataStructures.DataInstance;
import Utilities.Logging;
import weka.core.Instance;

/**
 *
 * @author Josif Grabocka
 */
public class EuclideanDistance extends DistanceOperator 
{
    public static EuclideanDistance instance = null;
    
    private EuclideanDistance()
    {
    }
    
    public static EuclideanDistance getInstance()
    {
        if(instance == null)
            instance = new EuclideanDistance();
        
        return instance;
    }

    @Override
    public double CalculateDistance(DataInstance ts1, DataInstance ts2) 
    {
        if( ts1.features.size() != ts2.features.size() )
        {
            Logging.println("Euclidean distance: feature size differs", Logging.LogLevel.ERROR_LOG);
            return Double.MAX_VALUE;
        }
        
        double distance = 0;
        
        for(int i = 0; i < ts1.features.size(); i++)
            distance += ts1.features.get(i).distanceSquare(ts2.features.get(i));
        
        return Math.sqrt(distance);
    }
    
    public double CalculateDistance(Instance ts1, Instance ts2) 
    {
        return CalculateDistance(
                new DataInstance("", ts1), 
                new DataInstance("", ts2));
    }
}
