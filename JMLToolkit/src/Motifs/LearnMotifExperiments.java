package Motifs;

public class LearnMotifExperiments 
{
	public void ShowFAsFunctionOfM(int K, int J, int L, double T, double alpha, double [][] S, double[][] M )
	{
		LearnMotifs lm = new LearnMotifs();
		lm.J = J;
		lm.K = K;
		lm.L = L;
		lm.alpha = alpha;
		lm.T = T;
		lm.S = S;
		lm.M = M;		
		lm.perSegmentFrequencies = new double[K][J];
		
		lm.PreComputePerSegmentFrequencies();
		double optFrequency = lm.ComputeFrequency();
		
		for(double m_1l = -2.0; m_1l <= 2.0; m_1l += 0.2) 
			for(double m_2l = -2.0; m_2l <= 2.0; m_2l += 0.2)
				{
					for(int l=0; l < L/2; l++)
						lm.M[0][l] = m_1l; 
					for(int l=L/2; l < L; l++)
						lm.M[0][l] = m_2l;
					
					lm.PreComputePerSegmentFrequencies();
					double f = lm.ComputeFrequency( lm.perSegmentFrequencies[0] );
					
					System.out.println( f + ", " + m_1l + ", " + m_2l ); 

				}
	}
	
	
}
