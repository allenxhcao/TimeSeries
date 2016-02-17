package MatrixFactorization;

import DataStructures.Tripple;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.Matrix;
import DataStructures.Tripples;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

import javax.rmi.CORBA.Util;

/*
 * The standard matrix factorization is best described by the loss function
 *  argmin_UV { u* loss_x(X, pl_x(U,Psi_i')) + (1-u)*loss_y(Y, pl_y(U,W')) + l_u*||U||+l_v*||Psi_i||+l_w*||W||) }, 
 *  where ||.|| is frobenius second norm
 *  l_u, l_v, l_w are lambdas, coefficients of the regularization 
 *  
 */

public class UnbiasedMatrixFactorization 
{
	/*
	 * The alpha coefficient is used to denote the speed of convergence of the 
	 * gradient descent
	 */ 
	public double eta;

	/*
	 * The coefficient is used to denote the weight of the regularization
	 * terms in the loss function
	 */
	public double lambdaU, lambdaV;
	
	public Matrix U, V;
    
	// data dimensions
	public int N,M,K;
	
	// max epocs
	public int maxEpocs;

	Random rand = new Random();
	
	
	List<Integer> rowIds, colIds;

    /*
     * The list of observed items in the XTraining
     */ 
    public Tripples trainData, testData;
    
	/*
	 * Constructor for the regularized SVD 
	 */
	public UnbiasedMatrixFactorization() 
	{
            eta = 0.0001; 
            maxEpocs  = 24000;
            lambdaU = 0;  
            lambdaV = 0;  
            trainData = testData = null;
	}

	/*
	 * Gradient descent training implementation matching X = UV'
	 */
	public void TrainReconstructionLoss(int i, int j, double X_ij)
	{
        double error_ij = X_ij - Predict(i, j); 
        
        double u_ik, v_kj, grad_u_ik, grad_v_kj;
        
        for(int k = 0; k < K; k++)
        {
        	v_kj = V.get(k, j);
        	u_ik =  U.get(i, k);
        	
        	grad_u_ik = -2*error_ij*v_kj + 2*lambdaU*u_ik;
        	
        	U.set(i, k, u_ik - eta * grad_u_ik);
			
	        grad_v_kj = -2*error_ij*u_ik + 2*lambdaV* v_kj;
    	 	
            V.set(k, j, v_kj - eta * grad_v_kj);
	    }
	}
	
	
	// get the total X loss of the reconstruction, the loss will be called
        // for getting the loss of a cell, so it must be initialized first
	public double GetTotalReconstructionLoss()
    {
        double XTrainingLoss = 0;
        
        for(Tripple trp : trainData.cells)
        {
        	double err =  trp.value - MatrixUtilities.getRowByColumnProduct(U, trp.row, V, trp.col);
            XTrainingLoss += err*err;
        }
                    
        return XTrainingLoss;
    }
	
    // initialize the matrixes before training
    public void Initialize()
    {
	
    	
        // initialize latent representation of X into latent space represented by U
        U = new Matrix(N, K);
        U.RandomlyInitializeCells(-0.0001, 0.0001);
       
    	// initialize a transposed Psi_i
        V = new Matrix(K, M);
        V.RandomlyInitializeCells(-0.0001, 0.0001);
        
    }
    
    public double Predict(int i, int j)
    {
    	return MatrixUtilities.getRowByColumnProduct(U, i, V, j);
    }
    
    public double ComputeRMSE(int trainOrTest)
	{
		double error = 0;
		// check whether we want to compute the RMS of train cells or test cells
		Tripples data = trainOrTest == 0 ? trainData : testData;
		
		// get the cumulative error in approximating each cell
		for( Tripple trp : data.cells )
		{
			double cellError = trp.value - Predict(trp.row, trp.col);
			error += cellError*cellError;
		}
		
		return Math.sqrt( error / data.cells.size() );
	}
	
    
	/*
	 * Train and generate the decomposition 
	 * */
    public double Decompose()
	{ 
        for(int epoc = 0; epoc < maxEpocs; epoc++)
		{
            for( Tripple trp : trainData.cells )
                TrainReconstructionLoss(trp.row, trp.col, trp.value); 
            
            double XLoss = GetTotalReconstructionLoss(),
		            regLossU = U.getSquaresSum(),
		            regLossV = V.getSquaresSum();
            
            double epocLoss = XLoss + lambdaU*regLossU +lambdaV*regLossV;  
            epocLoss /= trainData.cells.size();
           
            double trainRMSE = ComputeRMSE(0),
            		testRMSE = ComputeRMSE(1);
            
           DecimalFormat twoDForm = new DecimalFormat("#.###");
           
           Logging.println(
                   epoc + ": " +
                   "Loss=" + twoDForm.format(epocLoss) + 
                   ", TrainRMSE=" + twoDForm.format(trainRMSE) +
                   ", TestRMSE=" + twoDForm.format(testRMSE) 
                   , Logging.LogLevel.DEBUGGING_LOG);     
               
		}

        // return RMSE on test
        return ComputeRMSE(1);
	}

	public void FixIndices()
	{
		// fist merge elements in a set
		SortedSet<Integer> rowIdsSet = new TreeSet<Integer>();
		SortedSet<Integer> colIdsSet = new TreeSet<Integer>();
		
		// merge the row and col ids of the train and test sets
		rowIdsSet.addAll( trainData.rowIds ); 
		rowIdsSet.addAll( testData.rowIds );
		
		colIdsSet.addAll( trainData.colIds );
		colIdsSet.addAll( testData.colIds );
		
		// move the sets to plain arrays to 
		rowIds = new ArrayList<Integer>(); 
		rowIds.addAll(rowIdsSet); 
		
		colIds = new ArrayList<Integer>(); 
		colIds.addAll(colIdsSet);
		
		// convert the indices to the index of the ordered set
		for( Tripple trp : trainData.cells )
		{
			trp.row = rowIds.indexOf( trp.row );
			trp.col = colIds.indexOf( trp.col );
		}
		for( Tripple trp : testData.cells )
		{
			trp.row = rowIds.indexOf( trp.row );
			trp.col = colIds.indexOf( trp.col );
		}
		
		// randomly shuffle the cells
		Collections.shuffle( trainData.cells );
		Collections.shuffle( testData.cells );
		
    	N = rowIds.size();
    	M = colIds.size();    
	}
	
	
}
