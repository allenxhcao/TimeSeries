package Classification;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import MatrixFactorization.MatrixUtilities;
import TimeSeries.BagOfPatterns;
import TimeSeries.BagOfPatterns.RepresentationType;
import Utilities.GlobalValues;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import DataStructures.DataSet;
import DataStructures.Matrix;
import DataStructures.Tripple;

// multirelational factorization of time series

public class BOPSF 
{
	// the time series dataset 
	public Matrix X, Y;
	
	// extend the labels into all-vs-one representation
	public Matrix YExtended;
	
	// data description
	public int numTrainInstances, numTotalInstances, numPoints, numPatterns; 
	
	// the histogram of patterns
	public Matrix H;
	
	// the latent matrices of the series, time, pattern and classification weights
	public Matrix S, P, W;
	
	public double [] biasP;
	public double [] biasW;
	
	// the regularization parameters
	public double lambdaS, lambdaP, lambdaW;
	
	// the dimensionality of the latent space
	public int D;
	
	// the number of labels
	public int numLabels;
	
	// the learn rate 
	public double eta;
	
	// the impact weights
	public double alphaH, alphaY;
	
	// the bags of patterns parameters
	public int slidingWindowSize, innerDimension, alphabetSize, degree;
	
	// the maximum number of iterations
	public int maxEpochs;
	
	DecimalFormat df = new DecimalFormat("#.######");
    
    protected List<Tripple> HObserved, YObserved;

	    
	public BOPSF()
	{
		alphaY = 1.0;
		alphaH = 1.0;
	}
	
	// compute the histogram matrix
	public void ComputeHistogram()
	{
		BagOfPatterns bop = new BagOfPatterns();
		
		bop.representationType = RepresentationType.Polynomial;
		
		bop.slidingWindowSize = slidingWindowSize;
		 
		bop.representationType = RepresentationType.Polynomial;
		bop.innerDimension = innerDimension;
		bop.alphabetSize = alphabetSize;
        bop.polyDegree = degree; 
         
		H = bop.CreateWordFrequenciesMatrix(X);
		numPatterns = H.getDimColumns();
		
		for(int i = 0; i < H.getDimRows(); i++)
			for(int j = 0; j < H.getDimColumns(); j++)
				if( H.get(i, j) == 0)
				{
					//H.set(i, j, GlobalValues.MISSING_VALUE);
				}
		
		
		Logging.println("Histogram Sparsity: " + H.GetSparsityRatio(), LogLevel.DEBUGGING_LOG);
	}
		
	// initialize the matrices
	public void Initialize()
	{
		// compute the histogram matrix
        ComputeHistogram();
        
   	 	// create the extended Y
        YExtended = new Matrix(numTotalInstances, numLabels);
        
        // set all the cells to zero initially
        for(int i = 0; i < numTrainInstances; i++)
        	for(int l = 0; l < numLabels; l++)
        		YExtended.set(i, l, 0.0);
        
        // set to 1 only the column corresponding to the label
        for(int i = 0; i < numTotalInstances; i++)
            YExtended.set(i, (int)Y.get(i), 1.0);  
        
		// randomly initialize the latent matrices
		S = new Matrix(numTotalInstances, D);
        S.RandomlyInitializeCells(0, 1);
        
        P = new Matrix(D, numPatterns);
        P.RandomlyInitializeCells(0, 1);
        
        biasP = new double[numPatterns];
        for(int l = 0; l < numPatterns; l++)
        	biasP[l] = H.GetColumnMean(l);
        
        W = new Matrix(D, numLabels);
        W.RandomlyInitializeCells(0, 1);
        
        biasW = new double[numLabels];
        for(int l = 0; l < numLabels; l++)
        	biasW[l] = YExtended.GetColumnMean(l);
        
        // record the observed histogram values
        HObserved = new ArrayList<Tripple>();
        for(int i=0; i < H.getDimRows(); i++)
            for(int j=0; j < H.getDimColumns(); j++)
                if( H.get(i, j) != GlobalValues.MISSING_VALUE )
                    HObserved.add(new Tripple(i, j)); 
        
        Collections.shuffle(HObserved);

        // record the observed label values
        YObserved = new ArrayList<Tripple>();
        for(int i=0; i < numTrainInstances; i++)
            for(int l=0; l < YExtended.getDimColumns(); l++)
                if( YExtended.get(i, l) != GlobalValues.MISSING_VALUE )
                    YObserved.add(new Tripple(i, l)); 
        
        Collections.shuffle(YObserved);

	}
	
	public double Optimize()
	{
		// initialize the data structures
		Initialize();
		
		Random rand = new Random();
		
		double prevLossH = Double.MAX_VALUE;
		
		int YUpdatefrequency = HObserved.size()/numTotalInstances, 
				i, l, idxY = 0;
		
		
		for(int epoch = 0; epoch < maxEpochs; epoch++)
		{			
			// update H loss
			if(alphaH > 0)
			{
				double err_il;
				
				for(int idx = 0; idx < HObserved.size(); idx++)
				{
					i = HObserved.get(idx).row;
					l = HObserved.get(idx).col;
					
					err_il = H.cells[i][l] - MatrixUtilities.getRowByColumnProduct(S, i, P, l) - biasP[l];  
			        
			        for(int k = 0; k < D; k++)
			        { 
			        	S.cells[i][k] -= eta * ( -2*alphaH*err_il*P.cells[k][l] + lambdaS*S.cells[i][k] ); 
			            P.cells[k][l] -= eta * ( -2*alphaH*err_il*S.cells[i][k] + lambdaP*P.cells[k][l] ); 
			    	} 
			       
			        biasP[l] -= eta * (-2*alphaH*err_il);
			        
			        if( idx % YUpdatefrequency == 0)
			        {
			        	if(alphaY > 0)
						{
							i = YObserved.get(idxY).row;
							l = YObserved.get(idxY).col;
						    
							double err_i = YExtended.cells[i][l] - Sigmoid.Calculate(MatrixUtilities.getRowByColumnProduct(S, i, W, l));// + biasW[l]); 
					        
					        for(int k = 0; k < D; k++)
					        {
					            S.cells[i][k] -= eta * (alphaY*-err_i*W.cells[k][l] + lambdaS*S.cells[i][k]);
					        	W.cells[k][l] -= eta * (alphaY*-err_i*S.cells[i][k] + lambdaW*W.cells[k][l]);
					        }
					        
					        biasW[l] -= eta * (-alphaY*err_i); 

					        idxY = (idxY+1) % YObserved.size();
						}
       	
			        }
				}
			}
			
			double lossH = GetLossH();
			
			if(epoch %3 == 0)
			{
				// compute the losses of each relation and print the result
				double lossYTrain = GetLossY(0, numTrainInstances),
						lossYTest = GetLossY(numTrainInstances, numTotalInstances),
						mcrTrain = GetErrorRate(0, numTrainInstances), 
						mcrTest = GetErrorRate(numTrainInstances, numTotalInstances),
						mcrNN = GetTestErrorNN();
				
				Logging.println(
						"Epoch=" + df.format(epoch)
						+ ", Eta=" + df.format(eta)
						+ ", LH=" + df.format(lossH)  
						+", LY=" + df.format(lossYTrain) + "/" + df.format(lossYTest) 
						+", MCR=" + df.format(mcrTrain) + "/" + df.format(mcrTest) + "/" + df.format(mcrNN), LogLevel.DEBUGGING_LOG);
				
				//Logging.println("LX="+lossX+", LH="+lossH + ", LY="+lossY+", MCR=["+ mcrTrain+","+mcrTest+"]", LogLevel.DEBUGGING_LOG);
			}
			
			if( lossH < prevLossH )
			{
				//eta *= 1.01;
				
				prevLossH = lossH;	
			}
			else
			{
				//eta *= 0.7;
			}
			
		}
		
		// return the ultimate MCR
		return GetErrorRate(numTrainInstances, numTotalInstances);
		//return GetTestErrorNN();
	}
	
	public double GetTestErrorNN()
    {
	    DataSet trainSet = new DataSet();
	    trainSet.LoadMatrixes(S, Y, 0, numTrainInstances);

	    DataSet testSet = new DataSet();
	    testSet.LoadMatrixes(S, Y, numTrainInstances, numTotalInstances);
	    
	    NearestNeighbour nn = new NearestNeighbour("euclidean");

		return nn.Classify(trainSet, testSet);
		
    }

	// get the log loss of the target prediction
	public double GetLossY(int startIndex, int endIndex) 
	{
		double YTrainLoss = 0;
        int numObservedCells = 0;
        
        for(int i = startIndex; i < endIndex; i++)
            for(int l = 0; l < numLabels; l++) 
                if( YExtended.get(i, l) != GlobalValues.MISSING_VALUE )
                {
					double y_hat_i = Sigmoid.Calculate(MatrixUtilities.getRowByColumnProduct(S, i, W, l));// + biasW[l]);            
					double y_i = YExtended.get(i, l);
					    
					YTrainLoss += -y_i*Math.log( y_hat_i ) - (1-y_i)*Math.log(1-y_hat_i);
					
					numObservedCells++;
        		   
                }
                    
        return YTrainLoss / (double) numObservedCells;
	}
	
	// get the log loss of the target prediction
	public double GetErrorRate(int startIndex, int endIndex)
	{
        int numIncorrectClassifications = 0;
        int numInstances = 0;
        
        for(int i = startIndex; i < endIndex; i++)
        {
        	if( Y.get(i) != GlobalValues.MISSING_VALUE )
        	{
				double y = Y.get(i);
				double y_predicted = PredictLabel(i);
				    
				if(y != y_predicted)
					numIncorrectClassifications++;
				
				numInstances++;
        	}
        }
                    
        return (double) numIncorrectClassifications / (double) numInstances; 
	}
	
	// predict the label of the i-th instance
	public double PredictLabel(int i)
    {
    	double label = 0;
    	double maxConfidence = 0;
	    		
    	for(int l = 0; l < numLabels; l++)
    	{
    		double confidence = Sigmoid.Calculate(MatrixUtilities.getRowByColumnProduct(S, i, W, l));// + biasW[l]);
    		
    		if(confidence > maxConfidence)
    		{
    			maxConfidence = confidence;
    			label = (double) l;
    		}
    	}
    	
    	return label;
    }
	
	
	
	// the the MSE of the H loss
	public double GetLossH() 
	{
		double numInstances = 0;
		double errorSum = 0;
		
		for(int i = 0; i < numTotalInstances; i++)
			for(int l = 0; l < numPatterns; l++)
			{
				if(H.get(i, l) != GlobalValues.MISSING_VALUE)
				{
					double err = H.get(i, l) - MatrixUtilities.getRowByColumnProduct(S, i, P, l);
					errorSum += err*err;
					numInstances += 1.0;
				}
				
			}
				
		return errorSum/numInstances; 
	}
	
}
