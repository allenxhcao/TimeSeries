package Classification;

import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;

import MatrixFactorization.MatrixUtilities;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import DataStructures.Matrix;

// an implementation of the kernel logistic regression model
public class KernelLogisticRegression 
{
	int numTrainInstances;
	
	// the training data, predictors and labels
	Matrix trainPredictors;
	Matrix trainLabels;
	
	// the alphas weights  
	double [] alphas;
	// the bias parameter
	double bias;
	
	
	// the number of iterations
	public int maxEpochs;
	// the learning rate
	public double eta;
	// the regularization parameter
	public double lambda;

	// the gamma parameter 
	public double gamma;
	
	
	// a random number generator 
	Random rand;
	
	double minLogLoss;
	double [] optimumAlphas;
	double optimumBias;
	
	// contructor for the kernel ridge regression
	public KernelLogisticRegression()
	{
		numTrainInstances = -1;
		trainPredictors = null;
		trainLabels = null;
		
		gamma = 0.1; 
		
		alphas = null;
		
		rand = new Random();
		
		minLogLoss = Double.MAX_VALUE;
	}
	
	public void Train(Matrix trainPredictors, Matrix trainLabels) 
	{
		numTrainInstances = trainPredictors.getDimRows(); 
		
		if( numTrainInstances != trainLabels.getDimRows() ) 
		{
			Logging.println("Dimensions of predictors and labels dont match," + numTrainInstances + " and " + trainLabels.getDimRows(), 
					LogLevel.ERROR_LOG);
			return;
		}
		else if(numTrainInstances < 1) 
		{ 
			Logging.println("Empty training set.", LogLevel.ERROR_LOG); 
			return; 
		} 
		
		Train(trainPredictors, trainLabels, numTrainInstances); 
		
	}
	
	// run the training of the klr with only the first numTrainInstances
	// that is useful in cases of semisupervised predictors and labels
	public void Train(Matrix trainPredictors, Matrix trainLabels, int numTrainInstances) 
	{
		// initialize
		Initialize(trainPredictors, trainLabels, numTrainInstances);
		
		for(int epoch = 0; epoch < maxEpochs; epoch++)
		{
			// train for 1 step
			double trainLogLoss = TrainStep(1);
			
			if(trainLogLoss < minLogLoss)
			{
				minLogLoss = trainLogLoss;
				optimumAlphas = alphas.clone();
				optimumBias = bias;
				
				Logging.println("Epoch: " + epoch + ", loss: " + minLogLoss, LogLevel.DEBUGGING_LOG);
			}
			
		}
		
		// in the end store the optimum alphas
		alphas = optimumAlphas.clone();
		bias = optimumBias;
		
}
	
	// test the learned model over test predictors and labels
	public double Test(Matrix testPredictors, Matrix testLabels)
	{
		return Test(testPredictors, testLabels, 0, testPredictors.getDimRows());
	}
	
	// test only on a specified range of the test predictors and labels
	// useful in case of semisupervised learning
	public double Test(Matrix testPredictors, Matrix testLabels, int fromIndex, int toIndex)
	{
		int numErrors = 0;
		double logLoss = 0.0;
		int numObservedTestInstances = 0;
		
		double Y_hat_i, Y_i, estimatedLabel;
		
		// go through the test instances
		for(int i = fromIndex; i < toIndex; i++)
		{
			Y_i = testLabels.get(i);
			
			if( Y_i != GlobalValues.MISSING_VALUE )
			{
				Y_hat_i = Probability(testPredictors.getRow(i));
				
				logLoss += - Y_i * Math.log( Y_hat_i ) - ( 1 - Y_i )*Math.log( 1- Y_hat_i );
				
				estimatedLabel = Y_hat_i > 0.5 ? 1.0 : 0.0;
				if( Y_i != estimatedLabel )
					numErrors++;
				
				numObservedTestInstances++;
			}
		}
		
		double errorRate = (double)numErrors / (double)numObservedTestInstances;
		logLoss /= (double) numObservedTestInstances;
		
		Logging.println("LogLoss: "+ logLoss + ", ErrorRate: " + errorRate , LogLevel.INFORMATIVE_LOG);
		
		// return the error rate 
		return errorRate;
	}
	
	
	// compute the probability of a training instance
	public double Probability(int i)
	{
		return Probability(trainPredictors.getRow(i));
	}
	
	// compute the probability of an arbitrary predictors feature vector
	public double Probability(double [] predictorFeatures)
	{
		double val = bias;
		
		for(int i = 0; i < numTrainInstances; i++)
			val += ComputeRBFKernel(trainPredictors.getRow(i), predictorFeatures) * alphas[i];
		
		return Utilities.Sigmoid.Calculate(val);
	}
	
	
	// compute rbf kernel of U_i and U_l 
	public double ComputeRBFKernel(double [] U_i, double [] U_l)
	{
		double k_il = 0;
		
		double euclideanDistance = 0, dist_il = 0;
		for(int k = 0; k < U_i.length; k++)
		{
			dist_il = U_i[k]-U_l[k];
			euclideanDistance += dist_il*dist_il; 
		}
		
		k_il = Math.exp( - gamma * euclideanDistance );
		
		return k_il;
	}
	
	// the logloss performance on the training set
	public double TrainLogLoss()
	{
		double logLoss = 0.0;
		
		double Y_hat_i, Y_i;
		
		// go through the test instances
		for(int i = 0; i < numTrainInstances; i++)
		{
			Y_i = trainLabels.get(i);
			
			if( Y_i != GlobalValues.MISSING_VALUE )
			{
				Y_hat_i = Probability(trainPredictors.getRow(i));
				logLoss += - Y_i * Math.log( Y_hat_i ) - ( 1 - Y_i )*Math.log( 1- Y_hat_i);
			}
		}
		
		logLoss /= (double) numTrainInstances;
		
		return logLoss;
	}
	
	
	// for merging this method with other supervised factorizations we provide an incremental learning option
	
	public void Initialize(Matrix trainPredictors, Matrix trainLabels)
	{
		Initialize(trainPredictors, trainLabels, trainPredictors.getDimRows());
	}
	
	public void Initialize(Matrix trainPredictors, Matrix trainLabels, int numTrainInstances)
	{
		// set the local references to the dataset
		this.trainPredictors = trainPredictors;
		this.trainLabels = trainLabels;
		
		this.numTrainInstances = numTrainInstances;
		
		if( numTrainInstances != trainLabels.getDimRows())
		{
			Logging.println("Dimensions of predictors and labels dont match," + numTrainInstances + " and " + trainLabels.getDimRows(), 
					LogLevel.ERROR_LOG);
			return;
		}
		else if(numTrainInstances < 1)
		{
			Logging.println("Empty training set.", LogLevel.ERROR_LOG);
			return;
		}
		
		// initialize the alphas
		alphas = new double[numTrainInstances];
		
		// initialize the weights between -epsilon +epsilon
		for(int i = 0; i < numTrainInstances; i++)
			alphas[i] = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON;
		bias = 2*rand.nextDouble()*GlobalValues.SMALL_EPSILON - GlobalValues.SMALL_EPSILON;
		
		
	}

	
	// train the algorithm for a number of epochs 
	public double TrainStep(int numEpochs)
	{
		// iterate through a number of epochs
		double Y_i, err_i, grad_l, k_il;
		
		for(int epoch = 0; epoch < numEpochs; epoch++)
		{
			// iterate through every training instance i
			for(int i = 0; i < numTrainInstances; i++)
			{
				Y_i = trainLabels.get(i);
				
				if( Y_i == GlobalValues.MISSING_VALUE )
					continue;
				
				err_i = Y_i - Probability(i); 
				
				// update every alpha_l w.r.t to the error in predicting Y_i
				for(int l = 0; l < numTrainInstances; l++)
				{
					// compute the gradient
					k_il = ComputeRBFKernel(trainPredictors.getRow(i), trainPredictors.getRow(l));
					grad_l = -err_i*k_il + 2*lambda*alphas[i]*k_il; 
					
					// update the alpha in a gradient descent fashion
					alphas[l] = alphas[l] - eta*grad_l;
				}
				
				// update the bias parameter
				bias = bias + eta*err_i; 
			}
		}
		
		return TrainLogLoss();
	}
	
	

}
