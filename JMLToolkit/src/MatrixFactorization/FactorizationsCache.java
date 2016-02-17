package MatrixFactorization;

import java.io.File;
import java.io.InputStreamReader;

import Utilities.Logging;
import Utilities.Logging.LogLevel;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import DataStructures.DataSet;

public class FactorizationsCache 
{
	public String cacheDir;
	
	/*****************Singleton implementations ****************************/
	
	public static FactorizationsCache instance = null;
	
	private FactorizationsCache()
	{
		cacheDir = null;				
	}
	
	public static FactorizationsCache getInstance()
	{
		if(instance == null)
			instance = new FactorizationsCache();
		
		return instance;
	}
	
	/***********************************************************/
	
	
	public void GetLatent( String factorizationDescription, double learnRate, double lambda, int k, 
			DataSet latentTrainSet, DataSet latentTestSet, double alpha, int maxEpocs)
	{
		if(cacheDir == null)
		{
			Logging.println("Cache Folder not specified!", LogLevel.ERROR_LOG);
			return;
		}
		
		String filePathSeparator = File.separator;
		
		String latentTrainStr = cacheDir + filePathSeparator + factorizationDescription + "_train_" +  learnRate + "_" +lambda + "_" + k + "_" + alpha + "_" + maxEpocs + ".arff";
		String latentTestStr = cacheDir + filePathSeparator + factorizationDescription + "_test_"  + learnRate + "_" + lambda + "_" + k + "_" + alpha + "_" + maxEpocs + ".arff";
		
		File latentTrainFile = new File(latentTrainStr);
		File latentTestFile = new File(latentTestStr); 
		
		// check if the latent train and latent test files exists
		if( latentTrainFile.exists() && latentTestFile.exists() )
		{
			
			Logging.println("Found latent factorizations!", LogLevel.INFORMATIVE_LOG);
			
			try
			{
				DataSource latentTrainDataSource = new DataSource(latentTrainStr);
				Instances latentTrainInstances =  latentTrainDataSource.getDataSet();
				latentTrainSet.LoadWekaInstances(latentTrainInstances); 
				
				DataSource latentTestDataSource = new DataSource(latentTestStr);
				Instances latentTestInstances =  latentTestDataSource.getDataSet();
				latentTestSet.LoadWekaInstances(latentTestInstances);
			}
			catch(Exception exc)
			{
				Logging.println(exc.getMessage(), Logging.LogLevel.ERROR_LOG);
			}
		}
		else
		{
			Logging.println("No latent factorization found: " + latentTrainStr, LogLevel.INFORMATIVE_LOG);
		}
	}
	
	public void SaveLatent(DataSet latentTrainSet, DataSet latentTestSet, 
			String factorizationDescription, double learnRate, double lambda, int k, double alpha, int maxEpocs)
	{
		if(cacheDir == null)
		{
			Logging.println("Cache Folder not specified!", LogLevel.ERROR_LOG);
			return;
		}
		
		String filePathSeparator = File.separator;
		
		String latentTrainStr = cacheDir + filePathSeparator + factorizationDescription + "_train_" + learnRate + "_" + lambda + "_" + k + "_" + alpha + "_" + maxEpocs + ".arff";
		String latentTestStr = cacheDir + filePathSeparator + factorizationDescription + "_test_"  + learnRate + "_" + lambda + "_" + k + "_" + alpha + "_" + maxEpocs + ".arff";
		
		File latentTrainFile = new File(latentTrainStr);
		File latentTestFile = new File(latentTestStr); 
		
		if( !latentTrainFile.exists() && !latentTestFile.exists() )
		{
			latentTrainSet.SaveToArffFile( latentTrainStr );
			
			latentTestSet.SaveToArffFile( latentTestStr );
		}
	}
}
