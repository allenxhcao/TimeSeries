package Experiments;

import Classification.*;
import DataStructures.DataSet;
import DataStructures.ObjectSaver;
import MatrixFactorization.CollaborativeImputation;
import TimeSeries.DTW;
import TimeSeries.Distorsion;
import TimeSeries.LinearInterpolation;
import TimeSeries.SAXRepresentation;
import TimeSeries.SplineInterpolation;
import Utilities.ExpectationMaximization;
import Utilities.Logging;
import java.io.File;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;

public class TestMain 
{
	public static void main(String [] args)
	{
            Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG;
            
            String dir = "/home/josif/Documents/ucr_ts_data/", foldNo = "3";
            String savesDir = "/home/josif/Documents/ucr_ts_data/saves";
         
            String ds = "";
            
            if( args.length == 0 )
            {
                Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG;
                //Logging.currentLogLevel = Logging.LogLevel.PRODUCTION_LOG; 
            
                
                ds = "Coffee";
                    
                String sp = File.separator;
                
                args = new String[]{
                    "fold="+foldNo,
                    "model=wmfsvm", 
                    "runMode=test",
                    "trainSet=" + dir + ds + sp +"folds"+ sp + foldNo + sp+ ds +"_TRAIN" ,
                    "validationSet=" + dir + ds + sp +"folds"+ sp + foldNo + sp+ ds +"_VALIDATION",
                    "testSet=" + dir + ds + sp +"folds"+ sp + foldNo + sp+ ds +"_TEST",
                    "learnRate=0.01", 
                    "latentDimensionsRatio=0.3",
                    "lambda=0.01", 
                    "svmC=1", 
                    "svmKernel=polynomial",                     
                    "svmRKGamma=1.5",
                    "svmPKExp=4",
                    "totalMissingRatio=0.0", 
                    "gapRatio=0.05",
                    "interpolation=mbi",
                    "avoidExtrapolation=true",
                    "distorsion=true"
                };  
            }
            
            if( args.length > 0)
            {
                String model = "", runMode = "";
                String trainSetPath = "", validationSetPath = "", testSetPath = "", fold = "";
                double lambda = 0, learnRate = 0, svmC = 0, svmPKExp = 0, svmRKGamma = 0;
                double latentDimensionsRatio = 0;
                int k = -1;
                double totalMissingRatio = 0.0, gapRatio = 0.0; 
                String interpolationTechnique = "linear"; 
                boolean avoidExtrapolation = false;
                String svmKernel="";
                
                

                for( String arg : args )
                {
                    String [] argTokens = arg.split("=");

                    if( argTokens[0].compareTo("model") == 0 )
                        model = argTokens[1];
                    else if( argTokens[0].compareTo("runMode") == 0 )
                        runMode = argTokens[1];
                    else if( argTokens[0].compareTo("trainSet") == 0 ) 
                        trainSetPath = argTokens[1]; 
                    else if( argTokens[0].compareTo("validationSet") == 0 ) 
                        validationSetPath = argTokens[1];
                    else if( argTokens[0].compareTo("testSet") == 0 ) 
                        testSetPath = argTokens[1]; 
                    else if( argTokens[0].compareTo("fold") == 0 ) 
                        fold = argTokens[1]; 
                    else if( argTokens[0].compareTo("latentDimensionsRatio") == 0 ) 
                        latentDimensionsRatio = Double.parseDouble( argTokens[1] );  
                    else if( argTokens[0].compareTo("learnRate") == 0 ) 
                        learnRate = Double.parseDouble( argTokens[1] );  
                    else if( argTokens[0].compareTo("lambda") == 0 ) 
                        lambda = Double.parseDouble( argTokens[1] );
                    else if( argTokens[0].compareTo("svmKernel") == 0 ) 
                        svmKernel = argTokens[1]; 
                    else if( argTokens[0].compareTo("svmC") == 0 ) 
                        svmC = Double.parseDouble( argTokens[1] ); 
                    else if( argTokens[0].compareTo("svmPKExp") == 0 ) 
                        svmPKExp = Double.parseDouble( argTokens[1] ); 
                    else if( argTokens[0].compareTo("svmRKGamma") == 0 ) 
                        svmRKGamma = Double.parseDouble( argTokens[1] ); 
                    else if( argTokens[0].compareTo("dimensions") == 0 )  
                        k = Integer.parseInt(argTokens[1] );  
                    else if( argTokens[0].compareTo("totalMissingRatio") == 0 ) 
                        totalMissingRatio = Double.parseDouble( argTokens[1] ); 
                    else if( argTokens[0].compareTo("gapRatio") == 0 ) 
                        gapRatio = Double.parseDouble( argTokens[1] ); 
                    else if( argTokens[0].compareTo("interpolation") == 0 ) 
                        interpolationTechnique = argTokens[1];    
                    else if( argTokens[0].compareTo("avoidExtrapolation") == 0 ) 
                        avoidExtrapolation = argTokens[1].toUpperCase().compareTo("TRUE") == 0 ? true : false;   
                } 
                
                
                
                // load the train, validation and test sets
                DataSet trainSet = new DataSet();
                trainSet.LoadDataSetFile(new File( trainSetPath ));
                DataSet validationSet = new DataSet();
                validationSet.LoadDataSetFile(new File( validationSetPath ));
                DataSet testSet = new DataSet();
                testSet.LoadDataSetFile(new File( testSetPath ));
                
                // normalize the data instance
                trainSet.NormalizeDatasetInstances();
                testSet.NormalizeDatasetInstances();
                validationSet.NormalizeDatasetInstances();
                
                // the number of dimensions we want to project the time series dataset into
                k = (int) (trainSet.numFeatures * latentDimensionsRatio);
                
                String dsName = new File(trainSetPath).getName().split("_TRAIN")[0];
                
                Logging.println(
                        "DataSet: " + dsName + ", model:" + model + ", runMode:" + runMode + 
                        ", k :" + k + 
                        ", lambda: " + lambda + ", svmC: " + svmC + ", svmKernel: " + svmKernel +
                        (svmKernel.compareTo("rbf")==0? ", svmRKGamma: " + svmRKGamma : ", svmPKExp: " + svmPKExp) +
                        ", learnRate:" + learnRate,
                        Logging.LogLevel.DEBUGGING_LOG);
                
                // if wmfsvm warped matrix factorization and SVM
                if( model.compareTo("mfsvm") == 0)
                {
                	
                }
                else if( model.compareTo("mfsvm") == 0)
                {
                	
                    MFSVM mfSvm = new MFSVM(); 
                    mfSvm.latentDimensions = k;
                    mfSvm.learningRate = learnRate; 
                    mfSvm.lambda = lambda; 
                    mfSvm.svmC = svmC; 
                    mfSvm.kernelType = svmKernel;
                    mfSvm.svmRBFKernelGamma = svmRKGamma;
                    mfSvm.svmPolynomialKernelExp = svmPKExp;
                    
                    
                    if( runMode.compareTo("test") == 0 )
                    {
                        DataSet mergedTrain = new DataSet(trainSet);
                        mergedTrain.AppendDataSet(validationSet);
                        
                        // distord the trainset
                        mergedTrain = Distorsion.getInstance().Distort(mergedTrain, 0.03);
                        
                        double errorRate = mfSvm.Classify(mergedTrain, testSet);
                        Logging.println( 
                                errorRate + " " + 
                                dsName + " " + 
                                fold + " " +
                                model + " " +
                                k + " " + 
                                lambda + " " + 
                                svmC + " " + 
                                svmPKExp + " " + 
                                totalMissingRatio + " " + 
                                gapRatio, Logging.LogLevel.PRODUCTION_LOG);
                        
                        // save the latent representation into a file
                        
                        
                        mfSvm.latentTrainSet.SaveToArffFile("/home/josif/" + ds + "_"+foldNo+"_latent_train_distorted.arff");
                        mfSvm.latentTestSet.SaveToArffFile("/home/josif/" + ds + "_"+foldNo+"_latent_test.arff");
                        
                        /*
                        ObjectSaver os = new ObjectSaver();
                        
                        os.SaveToObjectFile(latentTrain, 
                        		savesDir + "/" +  dsName + "_" + lambda + "_" + k + "_" + learnRate + "_unsupervised_distorted.sav");
                       */
                    }
                    
                }
                
                
                
            }
        } 
}
