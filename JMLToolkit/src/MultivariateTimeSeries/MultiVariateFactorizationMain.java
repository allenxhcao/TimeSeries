package MultivariateTimeSeries;

import java.io.File;
import java.io.PrintStream;

import Utilities.Logging;
import Utilities.Logging.LogLevel;

public class MultiVariateFactorizationMain 
{

	public static void main(String [] args)
	{
		
		MultivariateTimeSeriesData mtsData = new MultivariateTimeSeriesData();
		
		int L = 300;
		double eta=1;
		int maxEpochs = 50;
		int K = 40; 
		double lambdaD = 0.000001, lambdaP = 0.000001;
		String dataFolder = "H:\\bci-eeg-challenge-2014-mini\\"; 
		
		// first only read the number of instances
		mtsData.loadData(dataFolder);
		mtsData.computeNumberOfSegments(L);
		// load the training data 
		mtsData.loadLabels(dataFolder + File.separator + "TrainLabels.csv"); 
		
		MultivariateTimeSeriesFactorization mtsFact = new MultivariateTimeSeriesFactorization();
		mtsFact.eta = eta;
		mtsFact.maxEpochs = maxEpochs;
		mtsFact.C = mtsData.C;
		mtsFact.K = K;
		mtsFact.L = L;
		mtsFact.lambdaD = lambdaD;
		mtsFact.lambdaP = lambdaP;
		
		mtsFact.Decompose( mtsData );  
		// compute the frequencies
		mtsFact.computeFrequencies();
		
		// create a classic predictors matrix and save it to a csv file
		double [][] predictors = new double[mtsData.N][mtsData.C*K];
		for(int i = 0; i < mtsData.N; i++)
			for(int c = 0; c < mtsData.C; c++)
				for(int k = 0; k < K; k++)
					predictors[i][c*K+k] = mtsFact.F[i][c][k];
				
		try
		{
			Logging.print(predictors, new PrintStream(dataFolder+File.separator+"factorizedPredictors.csv"), LogLevel.PRODUCTION_LOG);
		}
		catch(Exception exc)
		{
			exc.printStackTrace();
		}
	}
}
