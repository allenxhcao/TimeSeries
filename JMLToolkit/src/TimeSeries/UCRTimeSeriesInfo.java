package TimeSeries;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class UCRTimeSeriesInfo 
{
	public File timeSeriesLocation;
	public int  advisedCrossFold;
	
	/*
	 * A list of precomputed hyperparameters
	 * where we have the String name of the class as key 
	 * and the double array of values as the entry value
	 */
	Map<String,List<Double>> precomputedHyperParameters;
	
	public UCRTimeSeriesInfo()
	{
		precomputedHyperParameters = new HashMap<String, List<Double>>();
		timeSeriesLocation = null;
		advisedCrossFold = 0;
	}
	
	public UCRTimeSeriesInfo(File ts, int crossFold)
	{
		timeSeriesLocation = ts;
		advisedCrossFold = crossFold;
		precomputedHyperParameters = new HashMap<String, List<Double>>();
	}
	
}
