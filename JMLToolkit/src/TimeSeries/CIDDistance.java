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
public class CIDDistance extends DistanceOperator 
{
    public static CIDDistance instance = null;
    
    private CIDDistance()
    {
    }
    
    public static CIDDistance getInstance()
    {
        if(instance == null)
            instance = new CIDDistance();
        
        return instance;
    }

    public double CalculateDistance(double [] ts1, double [] ts2) 
    {
    	double edDistance = 0, diff=0; 
        
        for(int i = 0; i < ts1.length; i++)
        {
        	diff = ts1[i] - ts2[i]; 
        	edDistance += diff*diff;
        }        
        edDistance = Math.sqrt(edDistance);
        
        
        double ce1 = ComplexityEstimate(ts1);
        double ce2 = ComplexityEstimate(ts2); 
        
        return edDistance * (Math.max(ce1, ce2) / Math.min(ce1, ce2)); 
    }
    
    @Override
    public double CalculateDistance(DataInstance ts1, DataInstance ts2) 
    {
        if( ts1.features.size() != ts2.features.size() )
        {
            Logging.println("Euclidean distance: feature size differs", Logging.LogLevel.ERROR_LOG);
            return Double.MAX_VALUE;
        }
        
        double edDistance = 0; 
        
        for(int i = 0; i < ts1.features.size(); i++)
        {
        	double diff = ts1.features.get(i).value - ts2.features.get(i).value;
        	edDistance += diff*diff;
        }        
        edDistance = Math.sqrt(edDistance);
        
        
        double ce1 = ComplexityEstimate(ts1);
        double ce2 = ComplexityEstimate(ts2);
        
        return edDistance * (Math.max(ce1, ce2) / Math.min(ce1, ce2));  
    }
    
    public double CalculateDistance(Instance ts1, Instance ts2) 
    {
        return CalculateDistance(
                new DataInstance("", ts1), 
                new DataInstance("", ts2));
    }
    
    public double ComplexityEstimate(DataInstance ts) 
    {
    	double ce = 0;
    	
    	for(int p = 1; p < ts.features.size(); p++)
    	{
    		double diff = ts.features.get(p).value - ts.features.get(p-1).value;
    		ce += diff*diff;
    	}
    		
       return Math.sqrt(ce);
    }
    
    public double ComplexityEstimate(double[] ts) 
    {
    	double ce = 0;
    	
    	for(int p = 1; p < ts.length; p++)
    	{
    		double diff = ts[p] - ts[p-1];
    		ce += diff*diff;
    	}
    		
       return Math.sqrt(ce);
    }
}
