package Classification;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import DataStructures.DataSet;
import DataStructures.Matrix;
import MatrixFactorization.SupervisedMatrixFactorization;
import TimeSeries.UCRTimeSeriesCollection;
import Utilities.Logging;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.SelectedTag;

public class HyperParameterSearchNonGeneric 
{
	public UCRTimeSeriesCollection ucrCollection;
	
	// internal variables used for the random search of the BaseLineMF
	Random rand = new Random();
	private int randomKCandidate, randomEpocCandidate,
				minK=5, maxK=50, minEpoc = 2000, maxEpoc = 3000;
	
	private double randomUCandidate, randomLambdaUCandidate, 
					minU=0.5, maxU=1.0, minLambdaU=0.0001, maxLambdaU=0.1;
	private boolean randomNormalizedCandidate;

        
        public Evaluation RunTSMF(
                DataSet trainSet, DataSet testSet, 
                int k, double learnRate, double lambda, double svmC, double svmPKExp)
	{
            SupervisedMatrixFactorization mf = new SupervisedMatrixFactorization(k);
            mf.lambdaU = mf.lambdaV = lambda;
            mf.learningRate = learnRate;
            mf.lambdaW = 0;
            mf.maxEpocs = 5000;

            DataSet factorizedTrainSet = new DataSet(), 
                    factorizedTestSet = new DataSet();
            
            // train and test the latent representations of the data
            mf.Factorize(trainSet, testSet, factorizedTrainSet, factorizedTestSet);
            
            Instances factorizedTrainSetWeka = factorizedTrainSet.ToWekaInstances();
            Instances factorizedTestSetWeka = factorizedTestSet.ToWekaInstances();

            Evaluation eval = null;

            // create a svm classifier with specified hyperparameter parameters
            // the rest are weka-defaults
            SMO svm = WekaClassifierInterface.getPolySvmClassifier(svmC, svmPKExp);
            
            try
            {
                svm.buildClassifier(factorizedTrainSetWeka);

                eval = new Evaluation(factorizedTrainSetWeka);
                eval.evaluateModel(svm, factorizedTestSetWeka);
            }
            catch(Exception exc)
            {
                exc.printStackTrace();;
                Logging.println(exc.getMessage(), Logging.LogLevel.PRODUCTION_LOG);
            }

            return eval;            	
        }


        public void GridSearchBaselineMF(DataSet trainSet, DataSet testSet, 
                double learnRate, double lambda, double svmC, double svmPKExp)
	{
            int bestK = 0;	

            double 
                bestLambda = 0,  
                bestSvmC = 0, 
                bestSvmPKExp = 0,
                bestLearnRate = 0;

            double minErrorRate = Double.MAX_VALUE;  

            String dsName = trainSet.name; 

            int numFeatures = trainSet.instances.get(0).features.size();

            int minK = (int) (numFeatures*0.1), maxK = (int) (numFeatures*0.2);
            
            for( int k = minK; k <= maxK; k *= 2 )
            {   
                Evaluation eval = RunTSMF(trainSet, testSet, 
                                            k, learnRate, lambda, svmC, svmPKExp);
                
                double errorRate = eval.errorRate();

                if(errorRate < minErrorRate)
                {
                        minErrorRate = errorRate;

                        bestK = k;
                        bestLambda = lambda;
                        bestSvmC = svmC;
                        bestSvmPKExp = svmPKExp;
                        bestLearnRate = learnRate;

                }	
            }


		
		Logging.println( 
                    dsName + "," + minErrorRate +
                    ", hp: (" +
                    "dim=" + bestK + ";"+
                    "lambda=" + bestLambda + ";"+
                    "learnRate=" + bestLearnRate + ";"+
                    "SvmC=" + bestSvmC + ";"+
                    "SvmPKExp=" + bestSvmPKExp + ")", 
                    Logging.LogLevel.PRODUCTION_LOG);
		
	}
	
	/*
	 * A grid search of the BaselineMF hyperparameters
	 */
	public void GridSearchBaselineMF(DataSet trainSet, DataSet testSet)
	{
		int bestK = 0;	
		
		double 
                    bestLambda = 0,  
                    bestSvmC = 0, 
                    bestSvmPKExp = 0,
                    bestLearnRate = 0;
			 
		double minErrorRate = Double.MAX_VALUE;  
		boolean bestNormalized = false;
		int maxEpoc = 5000;
                
                String dsName = trainSet.name;
		
		int numFeatures = trainSet.instances.get(0).features.size() / 2;
		
		for( int k = numFeatures/8; k <= numFeatures/2; k *= 2 )
                    for( double learnRate = 0.0001; learnRate <= 0.01; learnRate *= 10 )
                        for( double lambda = 0.0001; lambda <= 1; lambda *= 10 )
                            for( double svmC = Math.pow(2.0,-1); svmC <= 4.1; svmC *= 2.0 )
                                for( double svmPKExp = 1; svmPKExp <= 4; svmPKExp += 1.0 )
                                {
                                    Evaluation eval = RunTSMF(trainSet, testSet, k, learnRate, lambda, svmC, svmPKExp);
                                    double errorRate = eval.errorRate();

                                    Logging.println( 
                                            dsName + ": Trial: ("+
                                            "dim=" + k + ","+
                                            "lambda=" + lambda + ","+
                                            "learnRate=" + learnRate + ","+
                                            "SvmC=" + svmC + ","+
                                            "SvmPKExp=" + svmPKExp + "),"+
                                            "Error: " + eval.errorRate(), 
                                            Logging.LogLevel.PRODUCTION_LOG);



                                    if(errorRate < minErrorRate)
                                    {
                                            minErrorRate = errorRate;

                                            bestK = k;
                                            bestLambda = lambda;
                                            bestSvmC = svmC;
                                            bestSvmPKExp = svmPKExp;
                                            bestLearnRate = learnRate;

                                            Logging.println( 
                                            dsName + ": Improvement: ("+
                                            "dim=" + k + ","+
                                            "lambda=" + lambda + ","+
                                            "learnRate=" + learnRate + ","+
                                            "SvmC=" + svmC + ","+
                                            "SvmPKExp=" + svmPKExp + "),"+
                                            "Error: " + eval.errorRate(), 
                                            Logging.LogLevel.PRODUCTION_LOG);


                                    }	
                                }

		
		
		Logging.println( 
                    dsName + ": Final: ("+
                    "dim=" + bestK + ","+
                    "lambda=" + bestLambda + ","+
                    "learnRate=" + bestLearnRate + ","+
                    "SvmC=" + bestSvmC + ","+
                    "SvmPKExp=" + bestSvmPKExp + "),"+
                    "Error: " + minErrorRate, 
                    Logging.LogLevel.PRODUCTION_LOG);
		
	}
	
	
        /*
	 * Search randomly for the hyperparameters of the BaselineMF
	 */
	public void RandomSearchBaselineMF(DataSet ds)
	{
		int crossFolds = ucrCollection.GetAdvisedFoldNumber(ds.name);
		
		double 
			bestK = 0,
			bestU = 0, 
			bestLabmdaU = 0,  
			bestLambdaV = 0,
			bestLambdaW = 0, 
			bestEpoc = 0, 
			minErrorRate = Double.MAX_VALUE;  
		boolean bestNormalized = false;
		
			
		for(int i = 0; i < 50; i++)
		{
			GenerateRandomCandidates();
			
			DataSet dataSet = new DataSet(ds);
			
			if( randomNormalizedCandidate )
			{
				dataSet.NormalizeDatasetInstances();
			}
			
			KFoldGenerator kFoldGenerator = new KFoldGenerator(dataSet, crossFolds);
			
			double errorRate = 0;
			int fold = 1;
			
			while( kFoldGenerator.HasNextFold() )
			{
				kFoldGenerator.NextFold();
				
				DataSet validationTrain = kFoldGenerator.GetTrainingFoldDataSet();
				DataSet validationTest = kFoldGenerator.GetTestingFoldDataSet();
				
				SupervisedMatrixFactorization mf = new SupervisedMatrixFactorization(randomKCandidate);
				
				mf.latentDim = randomKCandidate;
				mf.lambdaU = randomLambdaUCandidate;
				mf.lambdaV = randomLambdaUCandidate;
				mf.lambdaW = randomLambdaUCandidate;
				mf.maxEpocs = randomEpocCandidate;
				mf.alpha = randomUCandidate;
				
				DataSet factorizedTrainSet = null,
                                        factorizedTestSet = null;
                                
                                mf.Factorize(validationTrain, validationTrain, 
                                        factorizedTrainSet, factorizedTestSet); 
                                
				Evaluation eval = WekaClassifierInterface.Classify(factorizedTrainSet, factorizedTestSet);
				
				errorRate += eval.errorRate();
				
				Logging.println( 
						ds.name + ":"+
						randomKCandidate + ","+
						randomUCandidate + ","+
						randomEpocCandidate + ","+
						randomLambdaUCandidate + ","+
						(randomNormalizedCandidate ? "Normalized" : "NotNormalized") + "," +
						"fold-" + fold + "," +
						eval.errorRate(), Logging.LogLevel.PRODUCTION_LOG);
				
				fold++;
			}
			
			errorRate /= crossFolds;
			
			Logging.println( 
					ds.name + ":"+
					randomKCandidate + ","+
					randomUCandidate + ","+
					randomEpocCandidate + ","+
					randomLambdaUCandidate + ","+
					(randomNormalizedCandidate ? "Normalized" : "NotNormalized") + "," +
					"trial," +
					errorRate, Logging.LogLevel.PRODUCTION_LOG);
			
			if(errorRate < minErrorRate)
			{
				minErrorRate = errorRate;
				
				bestLabmdaU = randomLambdaUCandidate;
				bestLambdaV = randomLambdaUCandidate;
				bestLambdaW = randomLambdaUCandidate;
				bestU = randomUCandidate;
				bestK = randomKCandidate;
				bestEpoc = randomEpocCandidate;
				bestNormalized = randomNormalizedCandidate;
				
				Logging.println( 
						ds.name+":"+
						bestK+","+
						bestU+","+
						bestEpoc+","+
						bestLabmdaU+","+
						(bestNormalized ? "Normalized" : "NotNormalized") + "," +
						"improvement,"+
						minErrorRate, Logging.LogLevel.PRODUCTION_LOG);
			}
			
		}
	}

	public void GenerateRandomCandidates()
	{
		randomKCandidate = minK + rand.nextInt(maxK-minK);
		randomEpocCandidate = minEpoc + rand.nextInt(maxEpoc-minEpoc);
		randomLambdaUCandidate = minLambdaU + rand.nextDouble()*(maxLambdaU-minLambdaU);
		randomUCandidate = minU + rand.nextDouble()*(maxU-minU);
		randomNormalizedCandidate = rand.nextInt(2) == 0 ? true : false;
	}
}
