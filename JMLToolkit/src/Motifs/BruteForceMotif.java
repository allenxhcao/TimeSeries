package Motifs;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class BruteForceMotif 
{
	// the segments
	public double [][] S;
	// the motif to be learned
	public double [][] M;
	// the number of motifs
	public int K;
	// number of segments
	public int J; 
	// the length of motif and segments
	public int L;
	// the threshold parameter T
	public double T; 
	
	Random rand = new Random();
	
	
	// compute the hard occurrence of a motif
	public int ComputeHardFrequency(double [] motifCandidate)
	{
		int hardFrequency = 0; 
		double dist = 0; 
		
		int lastMatchIndex = -2;
		
		for(int j=0; j < J; j++) 
		{
			dist=0;
			for(int l=0; l < L; l++)
				dist += (motifCandidate[l] - S[j][l])*(motifCandidate[l] - S[j][l]); 
			
			if(dist <= T)
			{
				// avoid trivial matches  
				if( j - lastMatchIndex > 1 )
					hardFrequency++;
				
				lastMatchIndex = j;
			}
		}
		
		return hardFrequency;
	}
	
		
	// learn the motif
	public int Search() 
	{
		// compute the hard frequencies of all the segments
		int [] hardFrequencies = new int[J];
		for(int j=0; j < J; j++) 
			hardFrequencies[j] = ComputeHardFrequency(S[j]);
		
		// search the top motifs		
		M = new double[K][L];
		
		int hardScoreTotal = 0;

		// select K many motifs
		for(int k=0; k < K; k++)
		{
			int motifSegmentIndex = rand.nextInt(J); 
			
			// check through all the candidates
			for(int j=0; j < J; j++)
			{
				// see if the candidate is diverse, i.e. is 2T away from the previous frequent 
				// motifs
				if( isDiverse(S[j], k) ) 
				{
					if( hardFrequencies[j] > hardFrequencies[motifSegmentIndex] )
						motifSegmentIndex = j;
				} 
			}
			
			for(int l=0; l < L; l++)
				M[k][l] = S[motifSegmentIndex][l];
			
			hardScoreTotal += hardFrequencies[motifSegmentIndex];
			
			//System.out.println( "Segment=" + motifSegmentIndex + ", Frequency=" + hardFrequencies[motifSegmentIndex] ); 
		} 
		
		return hardScoreTotal;
	}
	
	// find the minimum index out of a list of indices from the hard scores
	public boolean isDiverse(double [] candidate, int currentNumMotifsSelected)
	{
		double dist=0;
		
		for(int k=0; k<currentNumMotifsSelected; k++)
		{
			dist=0;
			
			for(int l=0; l<L; l++)
				dist += (candidate[l]-M[k][l])*(candidate[l]-M[k][l]);
				
			// if the candidate is up to 2T close to a previous motif
			// then it is not diverse
			if(dist < 2.0*T)
				return false;
		}
		
		return true;
	}

	
}
