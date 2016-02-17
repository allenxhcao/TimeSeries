/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package MatrixFactorization;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint.PointStatus;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

/**
 * Impute missing values in a dataset by using collaborative imputation 
 * as a result of matrix factorization
 *  
 * @author Josif Grabocka
 */
public class CollaborativeImputation 
{
    public int k = 10;
    public double lambda = 0.01,
            learnRate = 0.003; 
    
    SupervisedMatrixFactorization mf = null;
    
    public CollaborativeImputation()
    {
        
    }
    
    
    public void Impute(DataSet dataSet)
    {
        mf = new SupervisedMatrixFactorization(k);
        mf.lambdaU = mf.lambdaV = lambda;
        mf.learningRate = learnRate;
        
        mf.Factorize(dataSet);
        
        for( int i = 0; i < dataSet.instances.size(); i++ )
        {
            DataInstance ins = dataSet.instances.get(i);
            
            for( int j = 0; j < ins.features.size(); j++ )
            {
                if(ins.features.get(j).status == PointStatus.MISSING)
                {
                    double prediction = MatrixUtilities.getRowByColumnProduct( mf.getU(), i, mf.getV(), j );
                    
                    
                    ins.features.get(j).value = prediction;
                    ins.features.get(j).status = PointStatus.PRESENT;
                }           
            }
        }
    }
    
}
