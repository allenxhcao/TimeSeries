package TimeSeries;

import org.apache.commons.math3.analysis.function.Sigmoid;

public class SupervisedBagOfPatterns 
{
	// number of patterns
	public int K;
	// number of instances
	public int ITrain, ITest;
	// length of series
	public int Q;
	// length of pattern
	public int L;	
	
	// time series data
	public double [][] T;
	// labels
	double Y[];
	
	// the patterns
	double [][] P;
	double [] W;
	double biasW;
	
	// regularization
	public double lambdaW;
	
	// precomputed frequencies
	double [][] F;
	
	
	// sigmoid function 
	Sigmoid sig;
	
	// learning rate
	public double eta;
	
	// estimate the frequency
	public double Frequency(int i, int k)
	{
		double f_ik = 0;
		
		return f_ik;
	}
	
	// estimate the target variable
	public double Predict(int i)
	{
		double Y_i = biasW;
		
		for(int k = 0; k < K; k++)
			Y_i += F[i][k]*W[k];
		
		return Y_i;
	}
	
	public void Learn()
	{
		
		double e_i = 0;
		double temp;
		double cte = ((double)2.0/(double)L);
		
		for(int i = 0; i < ITrain; i++)
		{
			e_i = -( Y[i] -  sig.value( Predict(i) ) );
			
			// update the bias
			biasW -= eta*e_i;
			
			//iterate over all the K patterns and weights
			for(int k = 0; k < K; k++)
			{
				// update the classification weights
				W[k] -= eta*( e_i*F[i][k] + 2*lambdaW*W[k] ) ;
				
				// update the patterns and all their L-many points
				for(int l = 0; l < L; l++)
				{
					temp = 0;
					for(int j = 0; j < Q - L; j++)
						temp += P[k][l] - T[i][j+l];
					
					P[k][l] -= eta*e_i*W[k]*cte*temp; 
				}
			}
		}
	}
	
}
