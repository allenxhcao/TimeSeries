package Motifs;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class Experiments 
{
	public static void main(String[] args)
	{
		if (args.length == 0) { 
			
			args = new String[] { 
					"dataSet=E:\\subversion\\pubs\\ideas\\learningMotifs\\data\\TAO_500_segments.txt",       
					"eta=0.3", 
	 				"maxIter=300", 
	 				"numRandomRestarts=1", 
	 				"alpha=2" 
				};
		}
		
		// the parameters
		int numRandomRestarts=0; 
		int maxIter=0;
		double eta=0.0;
		double alpha = 0;
		String dataSet = "";
		
		for (String arg : args) {
			String[] argTokens = arg.split("=");
			
			if (argTokens[0].compareTo("eta") == 0) 
				eta = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("maxIter") == 0)
				maxIter = Integer.parseInt(argTokens[1]);
			else if (argTokens[0].compareTo("numRandomRestarts") == 0) 
				numRandomRestarts = Integer.parseInt(argTokens[1]); 
			else if (argTokens[0].compareTo("alpha") == 0)
				alpha = Double.parseDouble(argTokens[1]);
			else if (argTokens[0].compareTo("dataSet") == 0)  
				dataSet = argTokens[1];			
		}
		
		
		System.out.println(dataSet);
		
		List<Integer> rangeK = new ArrayList<Integer>(); 
		rangeK.add(3); 
		//rangeK.add(10);  
		//rangeK.add(30); 
		
		List<Double> rangePercentiles = new ArrayList<Double>(); 
		//rangePercentiles.add(0.001);  
		rangePercentiles.add(0.01); 
		//rangePercentiles.add(0.1);
		
		
		for(int K : rangeK)
		{
			for(double percentile : rangePercentiles) 
			{
				long startTime = System.currentTimeMillis(); 
				
				// create a master instance used to load segments
				LearnMotifs lmParallel = new LearnMotifs(); 
				lmParallel.K = K;
				lmParallel.maxIter = maxIter; 
				lmParallel.eta = eta; 
				lmParallel.alpha = alpha; 
				lmParallel.LoadSegments(dataSet);
				DescriptiveStatistics descStats = lmParallel.SegmentDistancesStats();
				lmParallel.T = descStats.getPercentile(percentile); 
				
				
				System.out.println("T="+ lmParallel.T); 
				
				double lmHardFrequency = lmParallel.RunParallelRandomRestarts(numRandomRestarts); 
				
				long lmTime = System.currentTimeMillis() - startTime;
				
				for(int k=0; k < K; k++)
				{
					System.out.println("F("+k+")=" + lmParallel.ComputeHardFrequency( lmParallel.bestM[k] ) );
				}
				
				System.out.print("M=["); 
				for(int k=0; k < K; k++) 
				{
					Logging.print(lmParallel.bestM[k], LogLevel.DEBUGGING_LOG); 
					System.out.println(";"); 
				} 
				System.out.println("];"); 
				
				/*
				System.out.print("O=["); 
				for(int k=0; k < K; k++)
				{
					for(int j=0; j < lmParallel.J; j++)
						System.out.print( lmParallel.ComputeFrequencyPerSegment(lmParallel.bestM, k, j) + " " ); 
					System.out.println(";"); 
				}
				System.out.println("];");
				
				*/
				
				// output final log
				System.out.println(
						"K=" + K 
						+ ", Percentile=" + percentile 
						+ ", "  +  lmHardFrequency
						+ ", " + lmTime   
						+ ", T=" + lmParallel.T  
						+ ", numRestarts=" + numRandomRestarts
						+ ", J=" + lmParallel.J 
						+ ", maxIter=" + lmParallel.maxIter 
						+ ", eta=" + lmParallel.eta
						+ ", alpha=" + lmParallel.alpha ); 
			}
		}
	}
}
