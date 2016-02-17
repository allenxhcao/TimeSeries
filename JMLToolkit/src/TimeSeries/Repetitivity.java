package TimeSeries;

import java.util.Random;

import Utilities.StatisticalUtilities;
import DataStructures.Matrix;

public class Repetitivity 
{

	
	
	public double MeasureRepetitivity(Matrix X)
	{
		int N = X.getDimRows();
		int M = X.getDimColumns();
		
		double smallestLScore = Double.MAX_VALUE;
		
		int deltaL = (int)(0.05*X.getDimColumns());
		
		for(int L = deltaL ; L <= 0.2*M ; L+=deltaL )
		{
			double repetitivityScore = 0;
			
			System.out.println("N=" + N + ", M_i=" + M + ", L=" + L);  
			
			for(int i = 0; i < N; i++)
			{
				int totalPairs = 0;
				double instanceRepetitivityScore = 0;
				
				for( int jIndex = 0; jIndex < M/L; jIndex++ )
				{
					for( int kIndex = jIndex+1; kIndex < M/L; kIndex++ )
					{	
						int j = jIndex*L;
						int k = kIndex*L;
						
						if( j <= M-L && k <= M-L )
						{
							double dist = DistanceSubseries(X, i, j, k, L);
							
							instanceRepetitivityScore += ((double)dist/(double)L);
							
							totalPairs++;
						}
					}
				}
				
				instanceRepetitivityScore = instanceRepetitivityScore / (double) totalPairs;
				
				//System.out.println("i=" + i + ", D_i=" + instanceRepetitivityScore); 
				
				repetitivityScore += instanceRepetitivityScore;
			}
			
			repetitivityScore /= N; 
			
			System.out.println("L=" + L + ", rep=" + repetitivityScore); 
			
			if( repetitivityScore < smallestLScore)
				smallestLScore = repetitivityScore;
		}
		
		return smallestLScore;
	}
	
	// return the distance of X,i,start1,start1+L and X,i,start2,start2+L
	public double DistanceSubseries(Matrix X, int i, int start1, int start2, int L)
	{
		double [] subsequence1 = new double[L];
		double [] subsequence2 = new double[L];
		
		for(int l = 0; l < L; l++)
		{
			subsequence1[l] = X.get(i, start1+l);
			subsequence2[l] = X.get(i, start2+l);
		}
		
		//subsequence1=StatisticalUtilities.Normalize(subsequence1);
		//subsequence2=StatisticalUtilities.Normalize(subsequence2);
		
		
		return  DTW.getInstance().CalculateDistance(subsequence1, subsequence2);
		
				//StatisticalUtilities.SumOfSquares(subsequence1, subsequence2);
				//DTW.getInstance().CalculateDistance(subsequence1, subsequence2);
	}
	
}
