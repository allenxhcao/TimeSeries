/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Experiments;

import DataStructures.Tripple;
import DataStructures.Matrix;
import MatrixFactorization.MatrixFactorization;
import MatrixFactorization.MatrixUtilities;
import MatrixFactorization.SupervisedMatrixFactorization;
import Utilities.Logging;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 *
 * @author Josif Grabocka
 */
public class ReccomenderTest 
{
    int numUsers = 21;
    int numMovies = 838; 
    int latentDimensions = 2; 
    double learningRate = 0.00001; 
    double lambdaU = 0.00001;
    double lambdaV = 0.00001;
    double updateFraction = 1.0;
    
    // read the observed cells from the file
    public List<Tripple> ReadObserved(String file)
    {
        List<Tripple> observedCells = new ArrayList<Tripple>();
        
        Logging.println("Reading observe cells file: " + file, Logging.LogLevel.DEBUGGING_LOG);
        
        try
        {
            BufferedReader br = new BufferedReader( new FileReader(file) );

            String line = null;

            while( (line = br.readLine()) != null )
            {
                StringTokenizer tokenizer = new StringTokenizer(line, ",");

                int x = Integer.parseInt(tokenizer.nextToken());
                int y = Integer.parseInt(tokenizer.nextToken());
                double value = Double.parseDouble(tokenizer.nextToken());

                observedCells.add( new Tripple(x,y-1,value) );
            }
        }
        catch(Exception exc)
        {
            exc.printStackTrace();
        }
        
        Logging.println("Loaded: " + observedCells.size() + " observed dyadic values", Logging.LogLevel.DEBUGGING_LOG);
        
        return observedCells;
    }
    
    // initialize a matrix with the observed values
    public Matrix InitializeObservedMatrix(List<Tripple> observed)
    {
        Matrix m = new Matrix(numUsers, numMovies);
            
        for(Tripple c : observed)
        {
            m.set(c.row, c.col, c.value);
        }

        return m;
    }
    
 // test the RMSE performance of the latent representation
    public double TrainRMSE(Matrix U, Matrix V, List<Tripple> observedTrain)
    {
        double squareError = 0;
        
        for(Tripple c : observedTrain)
        {
            double prediction = MatrixUtilities.getRowByColumnProduct(U, c.row, V, c.col);
            
            squareError += Math.pow(prediction - c.value, 2);
        }
        
        return Math.sqrt( squareError / observedTrain.size() );
    }
    
    
    // test the RMSE performance of the latent representation
    public double TestRMSE(Matrix U, Matrix V, List<Tripple> observedTest, List<Tripple> observedTrain)
    {
        double squareError = 0;
        int numEvaluatedCells = 0;
        int numNewFeatureCells = 0;
        
        for(Tripple c : observedTest)
        {
        	boolean appearsOnTrain = false;
        	
        	for(Tripple trainC : observedTrain)
        	{
        		if(trainC.col == c.col)
        		{
        			appearsOnTrain = true;
        			break;
        		}
        	}
        	
        	if( appearsOnTrain )
        	{
	            double prediction = MatrixUtilities.getRowByColumnProduct(U, c.row, V, c.col);
	            
	            prediction = prediction < 0 ? 0 : prediction;
	            
	            squareError += Math.pow(prediction - c.value, 2);
	            
	            numEvaluatedCells++;
        	}
        	else
        	{
        		numNewFeatureCells++;
        	}
        }
        
        System.out.println("TestRMSE:: Ommitted " + numNewFeatureCells + " cells with new columns.");
        
        return Math.sqrt( squareError / numEvaluatedCells );
    }
    
    
    public static void main(String [] args)
    {
    	/*
        ReccomenderTest rst = new ReccomenderTest();
        String folder = "F:\\movielens_dataset\\leave-one-out\\";
        
        double avgRMSE = 0;
        
        for(int i = 0; i < 10; i++)
        { 
            String trainFile = folder + "ml1m-" + i + ".train.txt";
            String testFile = folder + "ml1m-" + i + ".test.txt";
            
            List<Coordinate> observedTrain = rst.ReadObserved(trainFile);
            List<Coordinate> observedTest = rst.ReadObserved(testFile);
            
            Matrix X = rst.InitializeObservedMatrix(observedTrain);
            
            MatrixFactorization mf = new MatrixFactorization(rst.latentDimensions);
            mf.learningRate = rst.learningRate;
            mf.lambdaU = rst.lambdaU;
            mf.lambdaV = rst.lambdaV;
            mf.maxEpocs = 110;
            mf.numTotalInstances = rst.numUsers;
            mf.numFeatures = rst.numMovies;
            
            
            mf.Decompose( X );
            
            double rmse = rst.TestRMSE(mf.getU(), mf.getV(), observedTest);
            
            System.out.println("Fold: " + i + ", RMSE= " + rmse); 
            
            avgRMSE += rmse;
        }
        
        avgRMSE /= 10;
        
        */
        
        ReccomenderTest rst = new ReccomenderTest();
        String folder = "F:\\data\\gps data\\infati\\gps_car2_no_home\\matlab preprocessing\\";
        
        String trainFile = folder + "infati_tripples_train.txt";
        String testFile = folder + "infati_tripples_test.txt";
        
        List<Tripple> observedTrain = rst.ReadObserved(trainFile);
        List<Tripple> observedTest = rst.ReadObserved(testFile);
        
        Matrix X = rst.InitializeObservedMatrix(observedTrain);
        
        MatrixFactorization mf = new MatrixFactorization(rst.latentDimensions);
        mf.learningRate = rst.learningRate;
        mf.lambdaU = rst.lambdaU;
        mf.lambdaV = rst.lambdaV;
        mf.maxEpocs = 2000;
        mf.numTotalInstances = rst.numUsers;
        mf.numFeatures = rst.numMovies;
        
        mf.Decompose( X );
        
        double trainRmse = rst.TrainRMSE(mf.getU(), mf.getV(), observedTrain);
        double testRmse = rst.TestRMSE(mf.getU(), mf.getV(), observedTest, observedTrain);
        
        
        System.out.println("RMSE_train= " + trainRmse + ", RMSE_test= " + testRmse); 
        
    }
    
}
