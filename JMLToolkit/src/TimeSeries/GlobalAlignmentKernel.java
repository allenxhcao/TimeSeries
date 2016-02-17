package TimeSeries;

import DataStructures.DataInstance;

public class GlobalAlignmentKernel extends DistanceOperator
{
	public String dsName = "";
    
	// singleton implementation of the kernel
	private static GlobalAlignmentKernel instance = null;
	
	// parameters gamma and lambda
	public double gamma, lambda;
	
    private GlobalAlignmentKernel()
    {
        gamma = 0.25;
        lambda = 1;
    }
    
    public static GlobalAlignmentKernel getInstance()
    {
        if(instance == null)
            instance = new GlobalAlignmentKernel();
        
        return instance;
    }
    
    // 
    private double k(double x, double y)
    {
    	double val = gamma*(x-y)*(x-y);
    	
    	double distance = val + Math.log( 2 - Math.exp(-val) );
    	
    	return Math.exp( - lambda * distance ); 
    }
    
	@Override
	public double CalculateDistance(DataInstance timeSeries1,
			DataInstance timeSeries2) 
	{
		int n = timeSeries1.features.size(),
				m = timeSeries2.features.size();
		
		double [][] costMatrix = new double[n][m];

		// initialize the cost matrix to 0
		for(int i = 0; i < n; i++)
			for(int j = 0; j < m; j++)
				costMatrix[i][j] = 0;
		
		// compute the first row of the cost matrix
		for(int j = 0; j < m; j++)
		{
			int i = 0;
			
			double x = timeSeries1.features.get(i).value;
			double y = timeSeries2.features.get(j).value; 
			costMatrix[i][j] = k(x,y);
		}
		// compute the first column of the cost matrix
		for(int i = 0; i < n; i++)
		{
			int j = 0;
			
			double x = timeSeries1.features.get(i).value;
			double y = timeSeries2.features.get(j).value; 
			costMatrix[i][j] = k(x,y);
		}
		
		// update all the other values of the cost matrix
		for(int i = 1; i < n; i++)
		{
			for(int j = 1; j < m; j++)
			{
				double x = timeSeries1.features.get(i).value;
				double y = timeSeries2.features.get(j).value; 
				
				costMatrix[i][j] = k(x,y) * 
						( costMatrix[i-1][j] + 
						  costMatrix[i][j-1] + 
						  costMatrix[i-1][j-1] );
			}
		}
		
		return costMatrix[n-1][m-1];
	}
	
}
