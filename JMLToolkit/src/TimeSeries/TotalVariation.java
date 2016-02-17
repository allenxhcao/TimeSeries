package TimeSeries;

import java.util.Random;

import DataStructures.DataInstance;

public class TotalVariation 
{
	private static TotalVariation instance = null;
	
	private TotalVariation() 
	{
		
	}
	
	public static TotalVariation getInstance()
	{
		if(instance == null)
			instance = new TotalVariation();
		
		return instance;
	}
	
	// compute the total variation of the series
	public double GetTotalVariation(double [] series)
	{
		double totalVariation = 0;
		
		for(int i = 0; i < series.length - 1; i++)
		{
			double diff = series[i] - series[i+1];
			totalVariation += diff*diff;
		}
		
		return totalVariation;
	}
	
	// compute the total variation of the series
	public double GetTotalVariation(DataInstance seriesIns)
	{
		double totalVariation = 0;
		
		for(int i = 0; i < seriesIns.features.size() - 1; i++)
		{
			double diff = seriesIns.features.get(i).value - seriesIns.features.get(i+1).value;
			totalVariation += diff*diff;
		}
		
		return totalVariation;
	}
	
	// get the difference in total variation among the 
	public double GetTotalVariationDistance(double [] series1, double [] series2)
	{
		double diff = GetTotalVariation(series1) - GetTotalVariation(series2);
		
		return diff*diff;
	}
	
	public double GetTotalVariationDistance(DataInstance ins1, DataInstance ins2)
	{
		double tvDiff = Double.MAX_VALUE;
		
		tvDiff = GetTotalVariation(ins1) - GetTotalVariation(ins2); 
		
		return tvDiff*tvDiff; 
	}
	
}
