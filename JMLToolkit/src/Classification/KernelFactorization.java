package Classification;

import java.util.Random;

import weka.classifiers.Evaluation;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.FeaturePoint.PointStatus;
import DataStructures.Matrix;
import MatrixFactorization.SupervisedMatrixFactorization;
import MatrixFactorization.SymmetricMF;
import TimeSeries.DTW;
import TimeSeries.EuclideanDistance;

/*
 * Precomputed Kernel SVM 
 */
public class KernelFactorization extends Classifier 
{
	DataSet trainKernel, testKernel;
	
	// parameters of the factorization
	public double lambda = 0.0001, 
			learnRate = 0.001;
	
	public KernelFactorization()
	{
		trainKernel = testKernel = null;
	}

	// build the kernel datasets
	public void BuildKernels() 
	{
		int n = trainSet.instances.size();
		int m = testSet.instances.size();
		
		trainKernel = new DataSet();
		trainKernel.name = trainSet.name;
		trainKernel.numFeatures = n;
		testKernel = new DataSet();
		testKernel.name = testSet.name; 
		testKernel.numFeatures = n;
		
		for( int i = 0; i < n; i++)
		{
			DataInstance ins = new DataInstance();
			
			DataInstance train_i = trainSet.instances.get(i);
			
			for( int j = 0; j < n; j++)
			{
				DataInstance train_j = trainSet.instances.get(j);
				
				double k_ij = EuclideanDistance.getInstance().CalculateDistance(train_i, train_j);
				//double k_ij = DTW.getInstance().CalculateDistance(train_i, train_j);
				ins.features.add(new FeaturePoint(k_ij));
			}
			
			ins.target = train_i.target;
			
			trainKernel.instances.add(ins);
		}
		
		
		for( int i = 0; i < m; i++)
		{
			DataInstance ins = new DataInstance();
			
			DataInstance test_i = testSet.instances.get(i);
			
			for( int j = 0; j < n; j++)
			{
				DataInstance train_j = trainSet.instances.get(j); 
				
				double k_ij = EuclideanDistance.getInstance().CalculateDistance(test_i, train_j);
				//double k_ij = DTW.getInstance().CalculateDistance(test_i, train_j);
				ins.features.add(new FeaturePoint(k_ij));
			}
			
			ins.target = test_i.target;
			testKernel.instances.add(ins);
		}
	}
	
	@Override
	protected void TuneHyperParameters(DataSet trainSest) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double Classify(DataSet trainSet, DataSet testSet) 
	{
		this.trainSet = trainSet;
		this.testSet = testSet;
		
		// build the kernels
		BuildKernels();
		
		int numInstances = trainSet.instances.size();
		int numFeatures = trainSet.numFeatures;
		int latentDim = 0;
		
		if(numInstances < numFeatures)
			latentDim = numInstances/3;
		else
			latentDim = numFeatures/3;
		
		latentDim = 4;
		/*
		SymmetricMF mf = new SymmetricMF(latentDim);
		//BaselineMF mf = new BaselineMF(latentDim);
		mf.lambdaU = mf.lambdaV = mf.lambdaW = lambda;
		mf.learningRate = learnRate;
		
		// factorize the train kernel and return the latent representation
		DataSet latentTrainSet = mf.Factorize(trainKernel);
		// afterwards fold the testKernel into the dimensionality learned
		// by the train kernel
		DataSet latentTestSet = mf.FoldIn(testKernel);
		
		
		//return svmInterface.Classify(latentTrainSet, latentTestSet);
		
		
		
		SVMInterface svmInterface = new SVMInterface();
		svmInterface.kernel = "linear";
		svmInterface.svmC = 1;
		svmInterface.degree = 1;
		
		return svmInterface.Classify(latentTrainSet, latentTestSet);
		
		*/
		//Evaluation eval = WekaClassifierInterface.Classify(trainKernel, testKernel);
		//return eval.errorRate(); 
		
		return 0; 
	}
	
	
	
}
