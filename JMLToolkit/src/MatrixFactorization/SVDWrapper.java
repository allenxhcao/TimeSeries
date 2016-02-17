/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package MatrixFactorization;

import DataStructures.DataSet;
import DataStructures.FeaturePoint;
/*
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.SingularValueDecomposition;
*/
/**
 *
 * @author Josif Grabocka
 */
public class SVDWrapper 
{
    public void Impute( DataSet ds )
    {
        double [][] dsValues = new double[ds.instances.size()][ds.numFeatures];
        
        for(int i = 0; i < ds.instances.size(); i++)
        {
            for(int j = 0; j < ds.numFeatures; j++)    
            {
                FeaturePoint p = ds.instances.get(i).features.get(j);
                
                if( p.status == FeaturePoint.PointStatus.PRESENT )
                {
                    dsValues[i][j] = p.value;
                }
                else if( p.status == FeaturePoint.PointStatus.MISSING )
                {
                    dsValues[i][j] = 0;
                }
                    
            }
        }
        /*
        DoubleMatrix2D dsMatrix = new DenseDoubleMatrix2D(dsValues);
        SingularValueDecomposition svd = new SingularValueDecomposition(dsMatrix);
        */
        // svd.getU().
    }
}
