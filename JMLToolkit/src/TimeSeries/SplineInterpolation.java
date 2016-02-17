/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package TimeSeries;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.FeaturePoint.PointStatus;
import Utilities.DataStructureConversions;
import Utilities.Logging;
import java.util.ArrayList;
import java.util.List;
import umontreal.iro.lecuyer.functionfit.BSpline;
import umontreal.iro.lecuyer.functionfit.LeastSquares;
import umontreal.iro.lecuyer.functionfit.PolInterp;
import umontreal.iro.lecuyer.functionfit.SmoothingCubicSpline;

/**
 *
 * @author Josif Grabocka
 */
public class SplineInterpolation 
{

    public static int LAGRANGE_EXTRAPOLATION = 0;
    public static int LINEAR_EXTRAPOLATION = 1;
    public static int POLYNOMIAL_EXTRAPOLATION = 2;
    public static int LEASTSQUARE_EXTRAPOLATION = 3;
    
    public int preferredExtrapolation;
    
    double gapRatio;
    
    public SplineInterpolation(double missingGapRatio) 
    { 
        preferredExtrapolation = LEASTSQUARE_EXTRAPOLATION;
        gapRatio = missingGapRatio;
    }
        
    // interpolate the dataset using a cubic spline
    public void CubicSplineInterpolation(DataSet ds)
    {
        for( int i = 0; i < ds.instances.size(); i++ )
        {
            CubicSplineInterpolation( ds.instances.get(i) ); 
        }        
    }
    
    public void CubicSplineInterpolation(DataInstance di)
    {
        try
        {
            List<Double> tsTimes = new ArrayList<Double>(), 
                    tsValues = new ArrayList<Double>();
            
            for(int i = 0; i < di.features.size(); i++)
            {
                FeaturePoint p =  di.features.get(i);
                
                if( p.status == PointStatus.PRESENT )
                {
                    tsTimes.add( (double)i );
                    tsValues.add( p.value );
                }
            }

            // create double vectors to be compatible with spline library
            double [] points = DataStructureConversions.ListToArrayDouble(tsTimes);
            double [] values = DataStructureConversions.ListToArrayDouble(tsValues);
            
            SmoothingCubicSpline spl = new SmoothingCubicSpline(points, values, 0.1);
            
            for(int i = 0; i < di.features.size(); i++)
            {
                FeaturePoint p =  di.features.get(i); 
                
                if( p.status == PointStatus.MISSING )
                {
                    // extrapolate if found on the extremes
                    if( i < tsTimes.get(0) || i > tsTimes.get( tsTimes.size()-1 )) 
                    {
                        boolean isLeftExtreme = i < tsTimes.get(0); 
                        p.value = Extrapolate(di, i, isLeftExtreme, points, values);
                    }
                    else // interpolate from the spline if it is in the between 
                    {
                        di.features.get(i).value = spl.evaluate(i);
                    }
                    
                    di.features.get(i).status = PointStatus.PRESENT;
                }
            }
            
        }
        catch(Exception exc)
        {
            Logging.println(exc.getMessage(), Logging.LogLevel.ERROR_LOG);
        }
    }
    
    // interpolate the dataset using a cubic spline 
    public void BSplineInterpolation(DataSet ds)
    {
        for( int i = 0; i < ds.instances.size(); i++ )
        {
            BSplineInterpolation( ds.instances.get(i) ); 
        }
    }
    
    /*
     * BSpline interpolation of the missing points of the time series
     */
    public void BSplineInterpolation(DataInstance di)
    {
        List<Double> tsTimes = new ArrayList<Double>(), 
        tsValues = new ArrayList<Double>();

        for(int i = 0; i < di.features.size(); i++)
        {
            FeaturePoint p =  di.features.get(i);

            if( p.status == PointStatus.PRESENT )
            {
                tsTimes.add( (double) i );
                tsValues.add( p.value );
            }
        }

        // create double vectors to be compatible with spline library
        double [] points = DataStructureConversions.ListToArrayDouble(tsTimes);
        double [] values = DataStructureConversions.ListToArrayDouble(tsValues);
        
        BSpline bsp = new BSpline(points, values, 3);
        
         for(int i = 0; i < di.features.size(); i++)
        {
                FeaturePoint p =  di.features.get(i); 
                
                if( p.status == PointStatus.MISSING )
                {
                    // extrapolate if found on the extremes
                    if( i < tsTimes.get(0) || i > tsTimes.get( tsTimes.size()-1 )) 
                    {
                        boolean isLeftExtreme = i < tsTimes.get(0);
                        p.value = Extrapolate(di, i, isLeftExtreme, points, values);
                    }
                    else // interpolate from the spline if it is in the between 
                    {
                        di.features.get(i).value = bsp.evaluate(i);
                    }
                    
                    di.features.get(i).status = PointStatus.PRESENT;
                }
            }
        
    }
    
    public double Extrapolate(DataInstance di, int missingIndex, boolean isLeftExtreme, double [] times, double [] values )
    {
        
        double missingTime = missingIndex;
        
        int numCloseNeighbors = ((int) (gapRatio * (di.features.size()-1)))/2; 
        if( numCloseNeighbors < 2) numCloseNeighbors = 2; 
        
        double [] closeNeighborTimes = new double[numCloseNeighbors];
        double [] closeNeighborValues = new double[numCloseNeighbors];

        if( isLeftExtreme )
        {
            for( int i = 0; i < numCloseNeighbors; i++ )
            {
                closeNeighborTimes[i] = times[i];
                closeNeighborValues[i] = values[i];
            }
        }
        else
        {
            for( int i = 0; i < numCloseNeighbors; i++ )
            {
                int index = times.length - 1 - i; 
                closeNeighborTimes[i] = times[index]; 
                closeNeighborValues[i] = values[index]; 
            }
        }

        double extrapolatedValue = 0;
        if( preferredExtrapolation == LAGRANGE_EXTRAPOLATION)
        {
            extrapolatedValue = LagrangeInterpolation(missingTime, closeNeighborTimes, closeNeighborValues);
        }
        else if( preferredExtrapolation == LINEAR_EXTRAPOLATION )
        {
            LinearInterpolation li = new LinearInterpolation(gapRatio); 
            extrapolatedValue = li.LinearlyExtrapolate(di, missingIndex, isLeftExtreme); 
        }
        else if( preferredExtrapolation == LEASTSQUARE_EXTRAPOLATION )
        {
            LeastSquares lsq = new LeastSquares(closeNeighborTimes, closeNeighborValues, 1);
            extrapolatedValue = lsq.evaluate(missingTime);  
        }
        else if( preferredExtrapolation == POLYNOMIAL_EXTRAPOLATION )
        { 
            PolInterp pi = new PolInterp(closeNeighborTimes, closeNeighborValues);
            extrapolatedValue = pi.evaluate(missingTime);
        }

        
        return extrapolatedValue;
    }
    
    /*
     * Return the lagrangian interpolation of a point x given observations xs and ys
     */
    public double LagrangeInterpolation(double xp, double xs[], double ys[])
    { 
        int n = xs.length; 
        double fp = 0, nominator, denominator; 
        
        for (int k=0; k<n; k++) 
        { 
            nominator = 1; 
            denominator = 1; 
            
            for (int j=0; j<n; j++) 
            { 
                if(k != j)
                { 
                    nominator *= xp - xs[j]; 
                    denominator *= xs[k] - xs[j]; 
                } 
            } 
            
            fp += nominator * ys[k] / denominator; 
        } 
        
        return fp; 
    }
    
  
    
}
