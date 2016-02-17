package Motifs;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;

import Utilities.Logging;

public class LearnMotifs 
{
	// the segments
	public double [][] S;
	// the motif to be learned
	public double [][] M;
	// number of segments
	public int J;
	// number of motifs
	public int K;
	// the length of motif and segments
	public int L;
	
	// the gradient accumulator
	public double [][] nabla;
	
	// the per segment occurrence
	double [][] perSegmentFrequencies;
	
	// the segmentwise distances
	double [][] phi;
	
	// the learning rate 
	public double eta;
	
	// the number of iterations
	public int maxIter; 
	
	// the threshold parameter T
	public double T; 
	
	// smoothness of the soft frequency
	public double alpha;
	
	// constants for the frequency and violation terms of the objective function 
	double c_F, c_V;
	
	// best objective score and motifs used for random restarts
	double bestF = -1;
	double [][] bestM = null;
	
	Random rand = new Random();
	
	public LearnMotifs()
	{
		
	}
	
	// load the segments file
	public void LoadSegments(String segmentsFile)
	{
		try
		{
			BufferedReader br = new BufferedReader(new FileReader( segmentsFile ));
			String line = null;
			J = 0;
			String delimiters = "\t ,;";
			
			while( (line=br.readLine()) != null)
			{
				StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
				L = tokenizer.countTokens();
				J++;
			} 
			
			br.close();
			
			Logging.println("J=" + J + ", L=" + L); 
			
			// initialize the segments
			S = new double[J][L];
			
			// load the segments
			br = new BufferedReader(new FileReader( segmentsFile ));
			line = null;
			
			int lineCount = 0;
			
			while( (line=br.readLine()) != null)
			{
				StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
			
				for(int l=0; l < L; l++)
					S[lineCount][l] = Double.parseDouble( tokenizer.nextToken() );
				
				lineCount++;
			}
			
			br.close();
			
		}
		catch(Exception exc)
		{
			exc.getStackTrace();
		}
		
	}
	
	// initialize the motifs and accumulators
	public void InitializeMethod()
	{
		// initialize motifs and their gradient accumulators 
		M = new double[K][L];
		nabla = new double[K][L];
		
		
		double dist = 0;
		
		for(int k=0; k < K; k++)
		{
			boolean isDiverse = true;
			int selectedSegmentIdx = 0;
			
			// try to select 'diverse' random segments as initial seeds
			// in case there no diverse random segments are selectable then
			// try at maxNumTrials, otherwise keep any random non-diverse segment
			int numTrials = 0, maxNumTrials = 1000;
			
			do{
				// select one random segment j that is diverse from the 
				// previous randomly selected segments
				selectedSegmentIdx = rand.nextInt(J);
				
				// assume it is diverse
				isDiverse = true;
				
				// check if the assumption is violated
				for(int q=0; q<k; q++)
				{
					dist=0;
					for(int l=0; l<L; l++)
						dist += (S[selectedSegmentIdx][l]-M[q][l])*(S[selectedSegmentIdx][l]-M[q][l]);
					
					if(dist < 2.0*T) 
						isDiverse = false;
				} 
				
				numTrials++; 
				
			}
			while(!isDiverse && numTrials < maxNumTrials);    
			
			//System.out.println(numTrials);
			
			for(int l=0; l < L; l++)
			{
				M[k][l] = S[selectedSegmentIdx][l]; 
				nabla[k][l] = 0;
			}
		}
		// initialize the per segment occurrence score
		perSegmentFrequencies = new double[K][J];
		
		// initialize the pairwise distances
		phi = new double[K][K]; 
		
		// initialize constants
		c_F = (-2.0*alpha) / ( ((double)J*K) *T); 
		c_V = 2.0/( (double) K*(K-1)*(T*T) );  

	}
	
	// compute the occurrence score per segment
	public double ComputeFrequencyPerSegment(int k, int j) 
	{
		double dist_kj = 0, err = 0;  
		for(int l=0; l < L; l++) 
		{
			err = M[k][l] - S[j][l];  
			dist_kj += err*err;  
		}
		
		return Math.exp( ( -alpha/ ((double) T) ) * dist_kj ); 
	} 
		
	// compute the occurrence score per segment
	public double ComputeFrequencyPerSegment(double[][] motifs, int k, int j)
	{
		double dist_kj = 0, err = 0; 
		for(int l=0; l < L; l++) 
		{
			err = motifs[k][l] - S[j][l]; 
			dist_kj += err*err;  
		} 
		
		return Math.exp( ( -alpha/ ((double) T) ) * dist_kj ); 
	} 
	
	// compute the occurrence score of all the motifs
	public double ComputeFrequency()
	{
		double score = 0;
		
		for(int k=0; k < K; k++)
			for(int j=0; j < J; j++)
				score += perSegmentFrequencies[k][j]; 
		
		return score / ((double) J*K);
	}
	
	
	// compute the occurrence score of one particular motif, given its occurrence per segment	
	public double ComputeFrequency(double [] motifPerSegmentFrequencies)
	{
		double score = 0;
		
		for(int j=0; j < J; j++)
			score += motifPerSegmentFrequencies[j]; 
		
		return score;
	}
	
	// pre compute the per segment Occurrences
	public void PreComputePerSegmentFrequencies()
	{
		for(int k=0; k < K; k++) 
			for(int j=0; j < J; j++)
				perSegmentFrequencies[k][j] = ComputeFrequencyPerSegment(k, j);
	}
	
	// pre compute the per segment Occurrences
	public void PreComputePairwiseMotifDistance() 
	{
		double err=0;
		
		for(int k=0; k < K; k++)
		{
			for(int q=0; q < K; q++)
			{
				phi[k][q] = 0;
				
				for(int l=0; l < L; l++)
				{
					err = M[k][l] - M[q][l];
					phi[k][q] += err*err; 
				}
			}
		}
	}
	
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
			
			if( dist <= T)
			{
				// avoid trivial matches  
				if( j - lastMatchIndex > 1 )
					hardFrequency++;
				
				lastMatchIndex = j;
			}
		}
		
		return hardFrequency;
	}
	
		
	// compute the hard occurrence of a set of motifs
	public int ComputeHardFrequency(double [][] M)
	{
		int hardFrequency=0;
		
		for(int k=0; k < K; k++) 
			hardFrequency += ComputeHardFrequency(M[k]);
		
		return hardFrequency;
	}
	
	// measure the distribution of distances
	// estimate the pruning distance
	public DescriptiveStatistics SegmentDistancesStats()
	{
		DescriptiveStatistics stat = new DescriptiveStatistics();
		
		double dist = 0;
		
		// select a large number of segment random pairs
		for(int j = 0; j < J; j++) 
		{
			for(int q = 0; q < J; q++)
			{
				if(j == q) 
					continue;
				
				// select only 1000*J pairs, so accept q only randomly with 1000/J probability
				// that is to avoid all J^2 pairs, which can be time consuming 
				if( rand.nextDouble() > (1000.0/((double)J)) )
					continue;
				
				dist = 0; 
				for(int l = 0; l < L; l++)  
					dist += (S[j][l] - S[q][l])*(S[j][l] - S[q][l]);
				
				stat.addValue(dist);
			}
		}
		
		return stat; 
	} 
	
	// compute the violations
	public double ComputeViolations()
	{
		double V = 0;
		
		for(int k=0; k < K; k++) 
			for(int p=k+1; p < K; p++) 
				if(phi[k][p] < (2.0*T))
					V += ( 1 - phi[k][p]/(2.0*T) )*( 1 - phi[k][p]/(2.0*T) );
				
		// divide by number of pairs
		V *= c_V; 
		
		return V;
	}
	
	// learn the motif
	public double Learn()
	{
		double F_grad_kl = 0, V_grad_kl = 0, O_grad_kl = 0;   
	
		// initialize the motifs randomly
		InitializeMethod(); 
		
		for(int iterIdx=0; iterIdx<maxIter; iterIdx++)
		{
			// precompute the per segment frequency
			PreComputePerSegmentFrequencies();
			// precompute the pairwise motif distance 
			PreComputePairwiseMotifDistance();
			
			for(int k=0; k < K; k++)
			{
				for(int l=0; l < L; l++) 
				{
					// compute the partial derivative of the frequency term
					F_grad_kl = 0;
					
					for(int j=0; j < J; j++)
						F_grad_kl += (M[k][l]-S[j][l])*perSegmentFrequencies[k][j];
					
					F_grad_kl *= c_F; 
					
					// compute the partial derivative of the violation
					V_grad_kl = 0;
					
					for(int q=0; q < K; q++)
						if( phi[k][q] < (2.0*T) )
							V_grad_kl += (phi[k][q]-(2.0*T))*(M[k][l]-M[q][l]);
					
					V_grad_kl *= c_V;
					
					// gradient as sum
					O_grad_kl = F_grad_kl - V_grad_kl;
					
					// accumulate the gradients
					nabla[k][l] += O_grad_kl*O_grad_kl;
					
					M[k][l] += (eta/Math.sqrt(nabla[k][l])) * O_grad_kl;   
				
				}
			}
			
			if( iterIdx % 10 == 0)
			{
			//	PreComputePerSegmentFrequencies();
			//	double F= ComputeFrequency(); 
			//	System.out.println("Iter="+iterIdx+", F="+F);
			} 
		}
		
		return	ComputeHardFrequency(M); 
	}
	
	// run parallel random restarts and return hard frequency
	public double RunParallelRandomRestarts(int numRandomRestarts)
	{
		// create a list of restart ids
		List<Integer> restartIdxs = new ArrayList<Integer>();
		for(int restartIdx = 0; restartIdx < numRandomRestarts; restartIdx++)
			restartIdxs.add(restartIdx);
		
		// run the restarts in parallel
		Parallel_1x0.ForEach(restartIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer restartIdx)
		    {
				LearnMotifs lm = new LearnMotifs();
				lm.S = S;
				lm.J = J;
				lm.L = L;
				lm.K = K;
				lm.maxIter = maxIter; 
				lm.eta = eta; 
				lm.alpha = alpha; 
				lm.T = T;
				
				// if this restart has a better objective score than store its motifs
				double F = lm.Learn(); 
				
				if( F > bestF) 
				{
					bestF = F;
					bestM = lm.M; 
				}
		    }
		});
		
		// return the frequency of the set of best motifs
		return bestF;  
	}
	
	public static void main(String[] args)
	{
		if (args.length == 0) { 
			
			args = new String[] { 
					"dataSet=E:\\subversion\\pubs\\grabocka2015c-kdd\\data\\insect_b_1000_segments.txt",     
					"eta=0.1", 
	 				"maxIter=1000", 
	 				"numRandomRestarts=1", 
	 				"alpha=2", 
	 				"K=10",
	 				"pct=0.1"
				};
		}
		
		// the parameters
		int numRandomRestarts=0; 
		int maxIter=0, K=0;
		double eta=0.0;
		double alpha = 0, pct = 0;
		String dataSet = "";
		
		for (String arg : args) {
			String[] argTokens = arg.split("=");
			
			if (argTokens[0].compareTo("eta") == 0)  
				eta = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("maxIter") == 0) 
				maxIter = Integer.parseInt(argTokens[1]); 
			else if (argTokens[0].compareTo("numRandomRestarts") == 0) 
				numRandomRestarts = Integer.parseInt(argTokens[1]); 
			else if (argTokens[0].compareTo("K") == 0) 
				K = Integer.parseInt(argTokens[1]); 
			else if (argTokens[0].compareTo("alpha") == 0) 
				alpha = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("pct") == 0) 
				pct = Double.parseDouble(argTokens[1]); 
			else if (argTokens[0].compareTo("dataSet") == 0)   
				dataSet = argTokens[1];	
		} 
		
		System.out.println(dataSet); 
		
		long startTime = System.currentTimeMillis(); 
		
		// create a master instance used to load segments
		LearnMotifs lmParallel = new LearnMotifs();
		lmParallel.K = K;
		lmParallel.maxIter = maxIter; 
		lmParallel.eta = eta; 
		lmParallel.alpha = alpha; 
		lmParallel.LoadSegments(dataSet);
		DescriptiveStatistics descStats = lmParallel.SegmentDistancesStats();
		lmParallel.T = descStats.getPercentile(pct); 
		
		System.out.println("Pct=" + pct + ";");
		System.out.println("T=" + lmParallel.T + ";"); 
		
		double lmHardFrequency = lmParallel.RunParallelRandomRestarts(numRandomRestarts);
		
		long lmTime = System.currentTimeMillis() - startTime; 
		
		// run the brute force search
		startTime = System.currentTimeMillis();
		
		BruteForceMotif bfm = new BruteForceMotif();
		bfm.T = lmParallel.T; 		
		bfm.S = lmParallel.S;  
		bfm.J = lmParallel.J; 
		bfm.L = lmParallel.L; 
		bfm.K = K; 
				
		int bfmHardFrequency = bfm.Search();
		
		long bfmTime = System.currentTimeMillis()-startTime;
		
		// print the motifs learned
		System.out.print("LearnMotifM=["); 
		for(int k=0; k < lmParallel.K; k++)
		{
			for(int l=0; l < lmParallel.L; l++)
				System.out.print(lmParallel.bestM[k][l] + " "); 
			
			if(k < lmParallel.K -1)
				System.out.println(";");
		}
		System.out.println("];");
		
		// print the frequencies of the learned motifs
		System.out.print("LearnMotifsFrequencies=["); 
		for(int k=0; k < lmParallel.K; k++)
		{
			System.out.print(lmParallel.ComputeHardFrequency(lmParallel.bestM[k]) + " ");  
		}
		System.out.println("];");
		
		// print the violations score
		lmParallel.M = lmParallel.bestM;
		lmParallel.InitializeMethod();
		lmParallel.PreComputePairwiseMotifDistance();
		double v = lmParallel.ComputeViolations(); 
		System.out.println("lmViolationScore=" + v + ";");  
		
		System.out.print("BruteForceM=["); 
		for(int k=0; k < bfm.K; k++)
		{
			for(int l=0; l < bfm.L; l++)
				System.out.print(bfm.M[k][l] + " "); 
			
			if(k < bfm.K -1)
				System.out.println(";");
		}
		System.out.println("];");
		
		System.out.print("BruteForceFrequencies=["); 
		for(int k=0; k < bfm.K; k++)
		{
			System.out.print(lmParallel.ComputeHardFrequency(bfm.M[k]) + " ");  
		}
		System.out.println("];");
		
		
		// output final log
		System.out.println(
				"K=" + K 
				+ ", Percentile=" + pct 
				+ ", " + bfmHardFrequency + ", " +  (int) lmHardFrequency 
				+ ", " + bfmTime + ", " + lmTime 
				+ ", T=" + lmParallel.T  
				+ ", numRestarts=" + numRandomRestarts 
				+ ", J=" + lmParallel.J 
				+ ", maxIter=" + lmParallel.maxIter 
				+ ", eta=" + lmParallel.eta
				+ ", alpha=" + lmParallel.alpha ); 
	
	}	
}