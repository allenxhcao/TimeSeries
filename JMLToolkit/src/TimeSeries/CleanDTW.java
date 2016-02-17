package TimeSeries;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import DataStructures.DataInstance;

public class CleanDTW extends DistanceOperator
{
	/*
	 * singleton implementation
	 */
	private static CleanDTW instance = null;
	
	Random rand = new Random();
	
	private CleanDTW() 
	{
		
	}
	
	public static CleanDTW getInstance()
	{
		if(instance == null)
			instance = new CleanDTW();
		
		return instance;
	}
	
	public double warpingWindowFraction = 0.1;
	
	/*
	 * alignment chosen to be same as dtw
	 */
	public List<List<Integer>> GetWarpingAlignment(List<Double> ts1, List<Double> ts2, int warpingWindow)
	{
		return DTW.getInstance().CalculateWarpingPath(ts1, ts2, warpingWindow);
	}
	
	/*
	 * distance by taking the average of the warping values at the second time series
	 */

	@Override
	public double CalculateDistance(DataInstance timeSeries1,
			DataInstance timeSeries2) {
		
		List<Double> ts1 = timeSeries1.GetFeatureValues();
		List<Double> ts2 = timeSeries2.GetFeatureValues();
		
		int warpingWindow = (int)( ts1.size() * warpingWindowFraction);
		
		List<List<Integer>> tau = GetWarpingAlignment(ts1, ts2, warpingWindow);
		
		double distance = 0;
		
		for(int t = 0; t < ts1.size(); t++)
		{
			double val = ts1.get(t);
			
			double warpedVal = 0;
			
			// get average of warped value
			for(int t1 = 0; t1 < tau.get(t).size(); t1++)
			{
				warpedVal += ts2.get(tau.get(t).get(t1));
			}
			warpedVal /= tau.get(t).size();
			
			distance += (val-warpedVal)*(val-warpedVal);
		}
		
		/*
		for(int t = 0; t < tau.size(); t++)
		{
	        for(int i = 0; i < tau.get(t).size(); i++) 
	        {
	        	double diff = ts1.get(t) - ts2.get(tau.get(t).get(i));
	           				
				distance += diff*diff;
	        }
			
		}
		*/
		
		return distance;
	}
	
    /*
     * get the average warped value of X_i(j) warped to V_k
     */
    public double GetWarpedValue(List<Double> timeSeries, List<List<Integer>> tau, int t)
    {
        double warpedVal = 0;
        int numWarpedIndexes = tau.get(t).size();

        for(int i = 0; i < numWarpedIndexes; i++) 
        {
            int warpedIndex = tau.get(t).get(i); 
            warpedVal  += timeSeries.get(warpedIndex);
        }

        warpedVal /= (double)numWarpedIndexes;

        return warpedVal; 
    }
	
}
