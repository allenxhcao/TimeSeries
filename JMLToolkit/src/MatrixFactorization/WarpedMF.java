package MatrixFactorization;

import java.util.ArrayList;
import java.util.List;

import DataStructures.Tripple;
import DataStructures.DataSet;
import DataStructures.Matrix;
import TimeSeries.DTW;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class WarpedMF extends SupervisedMatrixFactorization 
{
	// the warping path of every instance to the alignment 
    //public List<List<Coordinate>> warpingPaths;
    public List<List<List<Integer>>> warpingPaths;


    public double warpingWindowFraction;
    
	public WarpedMF(int factorsDim) 
	{
		super(factorsDim);
		
		warpingWindowFraction = 0.2;
		
	}

	@Override
	public void TrainReconstructionLoss(int i, int j)
	{
		// some variables to be used through computations
		double U_i_k, error_U_i_k, grad_U_i_k;		
		double V_k_jp, error_V_k_jp, grad_V_k_jp;
		
		for( int k = 0; k < latentDim; k++)
		{
			U_i_k = U.get(i,k);
			error_U_i_k = 0;
			
			for(int p = 0; p < warpingPaths.get(i).get(j).size(); p++) 
			{
				int t = warpingPaths.get(i).get(j).get(p);
				error_U_i_k = (X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, t))*V.get(k, t);
			}
			
			grad_U_i_k = -2*alpha*error_U_i_k + rec_features*lambdaU*U_i_k;
			
			// update the cell of U
			U.set(i, k, U_i_k - learningRate*grad_U_i_k);
			
			// go to every point jp aligned to point j of original series
			
			error_V_k_jp = 0; 
			
			for(int p = 0; p < warpingPaths.get(i).get(j).size(); p++) 
			{
				int t = warpingPaths.get(i).get(j).get(p);
				
				V_k_jp = V.get(k, t);
				error_V_k_jp = (X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, t));
				grad_V_k_jp = -2*alpha*error_V_k_jp*U_i_k + rec_totalInstances*2*lambdaV*V_k_jp;
				V.set(k, t, V_k_jp - learningRate*grad_V_k_jp);
			}

			
		}		
	}
	
	@Override
	public double GetTotalReconstructionLoss()
    {
        double XTrainingLoss = 0;
        
        for(int i = 0; i < numTotalInstances; i++)
        {
        	for(int j = 0; j< warpingPaths.get(i).size(); j++) 
			{
	        	for(int p = 0; p < warpingPaths.get(i).get(j).size(); p++) 
				{
					int t = warpingPaths.get(i).get(j).get(p);
	            	
	            	double pairDiff = X.get(i, j) - MatrixUtilities.getRowByColumnProduct(U, i, V, t);
	            	
	            	XTrainingLoss += pairDiff*pairDiff;
	            }
			}
        }
                    
        return XTrainingLoss;
    }
	
		// reconstruct all the series from the latent matrices
	    // and save the reconstructed series in a list
	    public List<List<Double>> ReconstructSeries()
	    {
	        // a list of reconstructed series
	        List<List<Double>> Xrec = new ArrayList<List<Double>>();
	    	
	    	for(int i = 0; i < numTotalInstances; i++ )
	    	{
		    	List<Double> X_i_reconstructed = new ArrayList<Double>();
				// compute the value for every time point and add it to the reconstructed
				// series
				for(int j = 0; j < X.getDimColumns(); j++)
					X_i_reconstructed.add(MatrixUtilities.getRowByColumnProduct(U, i, V, j));
				
				// add the reconstructed series to a list, (in order to avoid re-computation)
				Xrec.add(X_i_reconstructed);
	    	}
	    	
	    	return Xrec;
	    }
	    
	    // build prototype warpings of all instances to all the 
	    public void BuildWarpingPaths()
	    {
	    	List<List<Double>> Xrec = ReconstructSeries();
	    	
	    	// initialize the lists
	    	warpingPaths = new ArrayList<List<List<Integer>>>();  
	    	
	    	int warpingWindow = (int)(X.getDimColumns() * warpingWindowFraction); 
	    	
	    	// go through every series and 
	    	for(int i = 0; i < numTotalInstances; i++ )
	    	{
	    		// compute the warping path betwenn i-th series and its reconstruction
	    		//List<List<Integer>> x_i_warpingPath = 
	    			//	DTW.getInstance().CalculateWarpingPath(
	    				//		X.getRow(i), Xrec.get(i), warpingWindow);
	    		
	    		// add it to the list of warping paths
	    		//warpingPaths.add(x_i_warpingPath);    		
	    	}
	    }
	
    @Override
    public void PostInitializationRoutine() 
    {
    	if(true) return; 
    	
    	SupervisedMatrixFactorization mf = new SupervisedMatrixFactorization( latentDim );
        mf.lambdaU = mf.lambdaV = mf.lambdaW = lambdaU;
        mf.learningRate = learningRate;
        mf.stochasticUpdatesPercentage = stochasticUpdatesPercentage;
    	mf.alpha = alpha; 
    	mf.maxEpocs = maxEpocs;  
    	mf.maxMargin = maxMargin; 
    	
    	maxEpocs /= 5;
    	
        mf.Factorize(trainSet, testSet, null, null);  
	
    	U = new Matrix(mf.U);    
    	V = new Matrix(mf.V);   
    	W = new Matrix(mf.W);    	 
    	
    	intercepts = new double[mf.intercepts.length];
    	for(int i = 0; i < mf.intercepts.length; i++)
    		intercepts[i] = mf.intercepts[i];
    } 
	    
    @Override
	public void PreEpochRoutine()
    {
    		ReconstructSeries();   
			BuildWarpingPaths();
    }
	
    
}
