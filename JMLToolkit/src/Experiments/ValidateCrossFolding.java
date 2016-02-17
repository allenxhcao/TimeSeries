/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Experiments;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import Utilities.Logging;
import java.io.File;
import java.util.List;

/**
 *
 * @author Josif Grabocka
 */
public class ValidateCrossFolding 
{
    public static void main( String [] args )
    {
        if( args.length == 0)
            args = new String[]{"F:\\ucr_ts_data\\"};
        
        File dir = new File(args[0]);
        
        File [] dsFolders = dir.listFiles();
        
        String s = File.separator;
        
        for(File dsFolder : dsFolders)
        {
            
            String dsName = dsFolder.getName();
            
            Logging.println("Check folder: " + dsName, Logging.LogLevel.DEBUGGING_LOG);
            
            String defaultTrain= dir + s + dsName + s + dsName +"_TRAIN",
                    defaultTest= dir + s + dsName + s + dsName +"_TEST";
            
            //Logging.println("Validate default sets: " + dsName, Logging.LogLevel.DEBUGGING_LOG);
            //ValidateDefaultSets(defaultTrain, defaultTest); 
            
            for(int fold = 1; fold <=5; fold++)
            {
                String foldNo = String.valueOf(fold);  
            
                Logging.println("Checking dataset: " + dsName + " fold: " + foldNo, Logging.LogLevel.DEBUGGING_LOG);
                
                String trainSplit= dir + s + dsName + s + "folds"+ s + foldNo + s + dsName + "_TRAIN" ;
                String validationSplit=dir + s + dsName + s + "folds"+ s + foldNo + s + dsName + "_VALIDATION";
                String testSplit= dir + s + dsName + s + "folds"+ s + foldNo + s + dsName + "_TEST";
                
               ValidateSplit(trainSplit, validationSplit, testSplit, defaultTrain, defaultTest);
            }
        }
    }
    
    public static void ValidateSplit(
            String train, String validation, String test, 
            String defaultTrain, String defaultTest)
    {
        DataSet trainSetSplit = new DataSet();
        trainSetSplit.LoadDataSetFile( new File(train) );
        DataSet testSetSplit = new DataSet();
        testSetSplit.LoadDataSetFile( new File(test) );
        DataSet validationSetSplit = new DataSet();
        validationSetSplit.LoadDataSetFile( new File(validation) );
        
        DataSet defaultTrainSet = new DataSet();
        defaultTrainSet.LoadDataSetFile( new File(defaultTrain) );
        DataSet defaultTestSet = new DataSet();
        defaultTestSet.LoadDataSetFile( new File(defaultTest) );    

        for(int i = 0; i < defaultTrainSet.instances.size() + defaultTestSet.instances.size(); i++)
        {
            DataInstance ins = null;
            String defaultSetName = "";
            int lineNo = 0;
            
            if( i < defaultTrainSet.instances.size())
            {
                defaultSetName = "Train";
                lineNo = i+1;
                ins = defaultTrainSet.instances.get(i); 
            }
            else 
            {
                defaultSetName = "Test";
                lineNo = i-defaultTrainSet.instances.size()+1;
                ins = defaultTestSet.instances.get(i-defaultTrainSet.instances.size()); 
            }
            
            // search the instance in the train test and validation splits
            List<Integer> trainMatchMatches = trainSetSplit.SearchMultiMatch( ins );
            List<Integer> testMatchMatches = testSetSplit.SearchMultiMatch( ins );
            List<Integer> validationMatchMatches= validationSetSplit.SearchMultiMatch( ins );
            
            if( trainMatchMatches.size() >= 2)
            {
                Logging.print( "Default " + defaultSetName + " line " + lineNo + " found more than 1 time in train split: ", Logging.LogLevel.ERROR_LOG);
                
                for(Integer match : trainMatchMatches)
                    Logging.print( match + " ", Logging.LogLevel.ERROR_LOG);
                
                Logging.println("", Logging.LogLevel.ERROR_LOG);
            }
            if( testMatchMatches.size() >= 2)
            {
                Logging.print( "Default " + defaultSetName + " line " + lineNo + " found more than 1 time in test split: ", Logging.LogLevel.ERROR_LOG);
                
                for(Integer match : testMatchMatches)
                    Logging.print( match + " ", Logging.LogLevel.ERROR_LOG);
                
                Logging.println("", Logging.LogLevel.ERROR_LOG);
            }
            if(validationMatchMatches.size() >= 2)
            {
                Logging.print( "Default " + defaultSetName + " line " + lineNo + " found more than 1 time in validation split: ", Logging.LogLevel.ERROR_LOG);
                
                for(Integer match : validationMatchMatches)
                    Logging.print( match + " ", Logging.LogLevel.ERROR_LOG);
                
                Logging.println("", Logging.LogLevel.ERROR_LOG);
            }
            
            int trainMatchIndex = trainMatchMatches.size() > 0 ? trainMatchMatches.get(0) : -1;
            int testMatchIndex = testMatchMatches.size() > 0 ? testMatchMatches.get(0) : -1;
            int validationMatchIndex = validationMatchMatches.size() > 0 ? validationMatchMatches.get(0) : -1;
            
            int foundInTrain = trainMatchIndex >= 0 ? 1 : 0;
            int foundInTest = testMatchIndex >= 0 ? 1 : 0;
            int foundInValidation = validationMatchIndex >= 0 ? 1 : 0;
            
            int sum = foundInTrain + foundInTest + foundInValidation;
            
            if( sum != 1)
            {
                if( sum == 0)
                {
                    Logging.println("No match of train line " +  lineNo, Logging.LogLevel.ERROR_LOG);
                }
                else
                {
                    Logging.print(defaultSetName + " line " + lineNo + " found in both " , Logging.LogLevel.ERROR_LOG);

                    if( trainMatchIndex > 0 )
                        Logging.print(" : train split at line: " +  trainMatchIndex + ", file: " + train, Logging.LogLevel.ERROR_LOG);
                    if( testMatchIndex > 0 )
                        Logging.print(" : test split at line: " +  testMatchIndex + ", file: " + test, Logging.LogLevel.ERROR_LOG);
                    if( validationMatchIndex > 0 )
                        Logging.print(" : validation split at line: " + validationMatchIndex + ", file: " + validation, Logging.LogLevel.ERROR_LOG);
                    
                    Logging.println("", Logging.LogLevel.ERROR_LOG);
                } 
            }      
        }
        
    }
    
    
    public static void ValidateDefaultSets(String defaultTrain, String defaultTest)
    {
        DataSet defaultTrainSet = new DataSet();
        defaultTrainSet.LoadDataSetFile( new File(defaultTrain) );
        DataSet defaultTestSet = new DataSet();
        defaultTestSet.LoadDataSetFile( new File(defaultTest) ); 
        
        for(int i = 0; i < defaultTrainSet.instances.size() + defaultTestSet.instances.size(); i++)
        {
            DataInstance ins = null;
            String defaultSetName = "";
            int lineNo = 0;
            
            if( i < defaultTrainSet.instances.size())
            {
                defaultSetName = "Train";
                lineNo = i+1;
                ins = defaultTrainSet.instances.get(i);
            }
            else 
            {
                defaultSetName = "Test";
                lineNo = i-defaultTrainSet.instances.size() + 1;
                ins = defaultTestSet.instances.get(i-defaultTrainSet.instances.size()); 
            }
            
            List<Integer> trainMatches = defaultTrainSet.SearchMultiMatch( ins );
            List<Integer> testMatches = defaultTestSet.SearchMultiMatch( ins ); 
            
            // check for duplicates among the default train and test sets
            if( defaultSetName.compareTo("Train") == 0 )
            {
                if( trainMatches.isEmpty() )
                {
                    Logging.println("Train line: " +  lineNo + " not matched in itself!", Logging.LogLevel.ERROR_LOG);
                    
                }
                else if( trainMatches.size() > 1 )
                {
                    Logging.print("Default train line: " +  lineNo + " double matched in itself: ", Logging.LogLevel.ERROR_LOG);
                    
                    for(Integer match : trainMatches)
                        Logging.print( (match+1) + " ", Logging.LogLevel.ERROR_LOG);
                
                    Logging.println("", Logging.LogLevel.ERROR_LOG);
                }
                
                if( testMatches.size() > 0 )
                {
                    Logging.print("Default train line: " +  lineNo + " matched in default test: ", Logging.LogLevel.ERROR_LOG);
                    
                    for(Integer match : testMatches) 
                        Logging.print( (match+1) + " ", Logging.LogLevel.ERROR_LOG);
                
                    Logging.println("", Logging.LogLevel.ERROR_LOG);
                }
            }
            else if( defaultSetName.compareTo("Test") == 0 )
            {
                if( testMatches.isEmpty() )
                {
                    Logging.println("Test line: " +  lineNo + "not matched in itself!", Logging.LogLevel.ERROR_LOG);
                    
                }
                else if( testMatches.size() > 1 )
                {
                    Logging.print("Default test line: " +  lineNo + " double matched in itself: ", Logging.LogLevel.ERROR_LOG);
                    
                    for(Integer match : testMatches)
                        Logging.print( (match+1) + " ", Logging.LogLevel.ERROR_LOG);
                
                    Logging.println("", Logging.LogLevel.ERROR_LOG);
                }
                
                if( trainMatches.size() > 0 )
                {
                    Logging.print("Default test line: " +  lineNo + " matched in default train: ", Logging.LogLevel.ERROR_LOG);
                    
                    for(Integer match : trainMatches) 
                        Logging.print( (match+1) + " ", Logging.LogLevel.ERROR_LOG);
                
                    Logging.println("", Logging.LogLevel.ERROR_LOG); 
                }
            }
            
        }
    }
}
