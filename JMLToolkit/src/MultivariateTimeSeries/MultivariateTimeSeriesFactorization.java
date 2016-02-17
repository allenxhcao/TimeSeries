package MultivariateTimeSeries;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;

import Utilities.Logging;

public class MultivariateTimeSeriesFactorization 
{
	// the time series data structure
	public MultivariateTimeSeriesData tsData;
	
	// number of instances and number of channels
	public int N, C;
	// number of segments per instance
	public int [] J;
	
	// the degrees of membership
	double [][][][] D;
	// the patterns
	double [][] P;
	// length of a segment
	public int L;
	// number of patterns
	public int K;
	
	// the frequencies
	public double [][][] F;
	
	// number of train series
	public int NTrain;
	
	// number of iterations
	public int maxEpochs;
	// the learning rate 
	public double eta;
	// gradient accumulators for the parameters
	double [][][][] gradHistD;
	double [][] gradHistP;
	
	// regularization parameters
	public double lambdaD, lambdaP;
	
	// the indices of the instances
	List<Integer> instanceIdxs;
	
	// approximate the time series value: for the i-th instance, c-th channel, j-th segment at relative index l
	public double estimateValue(int i, int c, int j, int l)
	{
		double S_hat_icjk = 0;
		
		for(int k=0; k<K; k++)
			S_hat_icjk += D[i][c][j][k]*P[k][l];
		
		return S_hat_icjk; 
	}
	
	// measure the MSE
	public double computeMSE()
	{
		double mse = 0, err_icjl = 0;
		double numCells=0;
		
		for(int i = 0; i < N; i++)
			for(int c=0; c < C; c++)
				for(int j=0; j < J[i]; j++)
					for(int l=0; l < L; l++)
					{
						err_icjl = tsData.getValue(i, c, j, l) - estimateValue(i, c, j, l);
						mse += err_icjl*err_icjl;
						
						numCells+=1.0; 
					}
		
		return mse / numCells;
	}
	
	// compute the frequencies per each instance, channel at every dimension K
	public void computeFrequencies()
	{
		for(int i = 0; i < N; i++)
			for(int c=0; c < C; c++)
				for(int k=0; k < K; k++)
				{
					F[i][c][k] = 0;
					
					for(int j=0; j < J[i]; j++)
						F[i][c][k] += D[i][c][j][k];
					
					F[i][c][k] /= (double) J[i]; 
				}		
	}
	
	public void LearnSGDEpoch()
	{
		// parallel update of each instance
		Parallel_1x0.ForEach(instanceIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer i)
		    {
				double err_icjl = 0;
				double dOicjl_Dicjk = 0, dOicjl_Pkl = 0;
				
				for(int c=0; c < C; c++)
				{
					for(int j=0; j < J[i]; j++)
					{
						for(int l=0; l<L; l++)
						{
							err_icjl = tsData.getValue(i, c, j, l) - estimateValue(i, c, j, l);

							// update the degrees of membership and the patterns
							for(int k=0; k < K; k++)
							{
								// compute the gradient of D
								dOicjl_Dicjk = -2*err_icjl*P[k][l] + 2*lambdaD*D[i][c][j][k];
								// upgrade the gradient accumulator of D
								gradHistD[i][c][j][k] += dOicjl_Dicjk*dOicjl_Dicjk;
								// update the value of D
								D[i][c][j][k] -= (eta/Math.sqrt(gradHistD[i][c][j][k]))*dOicjl_Dicjk;
								
								// compute the gradient of P
								dOicjl_Pkl = -2*err_icjl*D[i][c][j][k] + 2*lambdaP*P[k][l];
								// update the gradient accumulator of P
								gradHistP[k][l] += dOicjl_Pkl*dOicjl_Pkl;
								// update the value of P
								P[k][l] -= (eta/Math.sqrt(gradHistP[k][l]) )*dOicjl_Pkl;
							}
						}
					}
						
				}
		    }
		});	
	}
	
	// decompose a multivariate time series dataset, into a number of dimension given as input
	// the length of a segment
	public void Decompose(MultivariateTimeSeriesData mts)
	{
		tsData = mts;
		
		// read the data dimensions from the data structure class
		N = tsData.getNumSeries();
		C = tsData.getNumChannels();
		
		J = tsData.J;
		
		// initialize parameters
		Random rand = new Random();
		
		P = new double[K][L];
		gradHistP = new double[K][L]; 
		// initialize the patterns from random segments
		for(int k=0; k<K; k++)
		{
			int i = rand.nextInt(N);
			int c = rand.nextInt(C);
			int j = rand.nextInt(J[i]);
			
			for(int l=0; l<L; l++)
			{
				P[k][l] = tsData.getValue(i, c, j, l); 
				gradHistP[k][l] = 0;
			}
		}
		
		// initialize degrees of membership randomly
		D = new double[N][C][][];
		gradHistD = new double[N][C][][];
		
		for(int i=0; i<N; i++)
			for(int c=0; c<C; c++)
			{
				D[i][c] = new double[J[i]][K];
				gradHistD[i][c] = new double[J[i]][K]; 
				
				for(int j=0; j<J[i]; j++)
				{
					for(int k=0; k<K; k++)
					{
						D[i][c][j][k] = 2*rand.nextDouble()-1;
						gradHistD[i][c][j][k] = 0;
					}
				}
				
			}
		
		// initialize the frequencies tensor
		F = new double[N][C][K];
		
		// initialize the indices of series
		instanceIdxs = new ArrayList<Integer>();
		for(int i=0; i<N; i++)
			instanceIdxs.add(i);
		Collections.shuffle(instanceIdxs); 
		
		Logging.println("N=" + N + ", C=" + C + ", K=" + K + ", L=" + L);  
		
		// iterate through a series of epochs
		for(int epoch=0; epoch<maxEpochs; epoch++)
		{
			LearnSGDEpoch();
			
			if(epoch%10 == 0)
			{
				double mse = computeMSE();
				Logging.println("Epoch="+ epoch + ", MSE=" + mse); 
			}
		}
		
	}
}
