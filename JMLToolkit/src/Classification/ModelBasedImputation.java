/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import weka.classifiers.functions.SMOreg;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Josif Grabocka
 */
public class ModelBasedImputation 
{
    public ModelBasedImputation()
    {
    }
    
    
    // impute the values of a dataset
    public void Impute(DataSet ds)
    {
        Instances instances = ds.ToWekaInstances();
                
        try
        {
            for( int attrIndex = 0; attrIndex < instances.numAttributes() - 1; attrIndex++)
            {
                instances.setClassIndex(attrIndex);
             
                SMOreg regressionModel = new SMOreg();
                regressionModel.buildClassifier(instances);
                
                for(int i = 0; i < instances.numInstances(); i++)
                {
                    Instance instance = instances.instance(i);

                    FeaturePoint tsp =  ds.instances.get(i).features.get(attrIndex);
                    
                    if( tsp.status == FeaturePoint.PointStatus.MISSING )
                    {
                        if( instance.isMissing(attrIndex) )
                        {
                            double predicted = regressionModel.classifyInstance(instance);
                            
                            tsp.value = predicted;
                            tsp.status = FeaturePoint.PointStatus.PRESENT;
                        }
                        else
                        {
                            Logging.println("Discrepancies, point missing in dataset but present in wekaset", 
                                    LogLevel.ERROR_LOG);
                        }
                    }
                }
                
            }
            
        }
        catch(Exception exc)
        {
        }
        
        
        ds = new DataSet(instances);
    }
    
    public static void main(String [] args)
    {
        
    }
    
}
