package Classification;

import java.util.ArrayList;
import java.util.Random;

import DataStructures.DataSet;
import DataStructures.Matrix;
import Utilities.IOUtils;
import Utilities.Logging;
import Utilities.Logging.LogLevel;

import java.util.List;
import java.io.File;
import weka.core.Instances;
import weka.core.converters.CSVSaver;

/*
 * An implementation of the KFold technique to divide a dataset into k folds
 * and then use 1 for testing and k-1 for training. Each k subsets are used as 
 * test in row.
 *  
 * Example code usage:
 * 
 * Dataset dataSet = new Dataset();
 * 
 * int noFolds = 10;
 * 
 * KFoldGenerator foldGen = new KFoldGenerator(dataSet, noFolds);
 * 
 * while (fold.HasNextFold())
 * {
 * 		fold.NextFold();
 * 
 * 		DataSet trainingSubSet = fold.GetTrainingFoldDataSet();
 * 		DataSet testingSubSet = fold.GetTestingFoldDataSet();
 * }
 * 
 */


public class KFoldGenerator 
{
	// the dataset to be folded
	DataSet dataSet;
        // weka representation of the dataset
        Instances dataSetWeka;
        Random rand;
	
	// no Folds
	int noFolds;
	// the instances allocated to each fold
	//public List<List<Integer>> folds;
	
	// the current fold index used when iterating over folds
	int currentFoldIndex;
	DataSet trainingFold;
	DataSet testingFold;
	
	public boolean isRegression;
	
	/*
	 * Constructors 
	 */
	public KFoldGenerator()
	{
		dataSet = null;
		noFolds = 0;
		currentFoldIndex = -1;
                
        rand = new Random();
        
        isRegression = true;
	}
	
        /*
         * Create a fold generator, the type can be:
         * null or "default" to create complete random folds
         * "stratified" to create random folds where the global ratio of labels 
         * is preserved in every fold, i.e. it there are 6 label A instances in the dataset
         * and we want to have 2 folds, then each fold should have 3 label A instances
         */
	public KFoldGenerator(DataSet ds, int numberOfFolds)
	{                
                noFolds = numberOfFolds;
                currentFoldIndex = 0;
                
                rand = new Random();
                
                Instances dataSetTemp = ds.ToWekaInstances();
                dataSetWeka = new Instances(dataSetTemp);
                // randomize the instances
                dataSetWeka.randomize(rand);
                                
                // split the dataset into stratified folds
                dataSetWeka.stratify(noFolds);
	}
	
	// another constructor for regression based data
	public KFoldGenerator(Matrix ds, int numberOfFolds)
	{                
                noFolds = numberOfFolds;
                currentFoldIndex = 0;
                
                rand = new Random();
                
                Instances dataSetTemp = ds.ToWekaInstances();
                dataSetWeka = new Instances(dataSetTemp);
                // randomize the instances
                dataSetWeka.randomize(rand);
                                
                // split the dataset into stratified folds
                dataSetWeka.stratify(noFolds);
	}
     
	/*
	 * Checks if there is any fold to be read
	 * until index points to the last element
	 */
	public boolean HasNextFold() 
	{
		if( currentFoldIndex < 0 )
			return false;
		else
			return currentFoldIndex < noFolds;
	}
	
	/*
	 * Generate the current fold and move index forward
	 */
	public void NextFold()
	{
		if( HasNextFold() )
		{
			if( isRegression)
			{
				
			}
			else
			{
                Instances train = dataSetWeka.trainCV(noFolds, currentFoldIndex);
                Instances test = dataSetWeka.testCV(noFolds, currentFoldIndex);
                
                trainingFold = new DataSet();
                trainingFold.LoadWekaInstances(train);
                
                testingFold = new DataSet();
                testingFold.LoadWekaInstances(test);
			}
			
            currentFoldIndex++;
		}
	}
	
	public DataSet GetTrainingFoldDataSet()
	{
		if( trainingFold != null )
		{
			return trainingFold; 
		}
		else 
		{
			return null;
		}
	}
	
	
	public DataSet GetTestingFoldDataSet()
	{
		if( testingFold != null )
		{
			return testingFold; 
		}
		else 
		{
			return null;
		}
	}	
        
		// set the dataset target variables to be binary
		public void SetDatasetBinary(DataSet ds)
		{
			ds.ReadNominalTargets();
			List<Double> labels = new ArrayList<Double>(ds.nominalLabels);
			
			double halfLabelVal = (double)labels.size() / 2.0;
			
			for(int i = 0; i < ds.instances.size(); i++)
			{
				double oldLabel = ds.instances.get(i).target;
				double newLabel = oldLabel <= halfLabelVal ? 0 : 1;
				
				System.out.println("Converted " + oldLabel + " to " + newLabel);
					
				ds.instances.get(i).target = newLabel; 
			}
			
		}
        
        // take the references of the ucr collection and create stratified folds
        public void GenerateFoldedDataSets(String urcDir)
        {
            int numFolds = 5;
            File dir = new File(urcDir);
        
            File [] dsFolders = dir.listFiles();

            String s = File.separator;

            for(File dsFolder : dsFolders)
            {
                String dsName = dsFolder.getName();
                
                // load the dataset 
                dataSet = new DataSet();
                dataSet.LoadDataSetFolder(dsFolder);
                
                Logging.println("Loaded: " + dsName + ", " + dataSet.instances.size() + " instances", 
                        Logging.LogLevel.DEBUGGING_LOG);
                
                int numInstancesBefore = dataSet.instances.size();
                
                //dataSet.RemoveDuplicateInstances();
                
                int numInstancesAfter = dataSet.instances.size();
                
                if( numInstancesAfter != numInstancesBefore)
                {
                    Logging.println("REMOVED: " + (numInstancesBefore-numInstancesAfter) + " duplicate instances", 
                        Logging.LogLevel.DEBUGGING_LOG);
                }
                
                // set the dataset binary
                //SetDatasetBinary(dataSet);
                
                // convert to weka instances format
                dataSetWeka = dataSet.ToWekaInstances();
                // randomize the instances
                dataSetWeka.randomize(rand);
                // split into stratified folds
                dataSetWeka.stratify(numFolds); 
                
                
                // create a folder to hold the splits
                File foldsDir = new File(dsFolder + File.separator + "folds");
                // delete the folder and all the contents
                IOUtils.Remove(foldsDir);
                // (re)create the folds folder
                foldsDir.mkdir();
                
                for(int fold = 0; fold < numFolds; fold++)
                {
                    // divide the dataset into a train and test set
                    Instances test = dataSetWeka.testCV(numFolds, fold);
                    Instances train = new Instances(dataSetWeka.trainCV(numFolds, fold));
                    
                    // stratify the train into #-1 folds and take one out as validation
                    train.randomize(rand);
                    train.stratify(numFolds-1);
                    Instances validation = train.testCV(numFolds-1, 0);
                    train = train.trainCV(numFolds-1, 0);
                    
                    File foldDir = new File(foldsDir + File.separator + String.valueOf(fold+1) ); 
                    // delete the fold folder and all the contents
                    IOUtils.Remove(foldDir);
                    // create a folder
                    foldDir.mkdir();
                    
                    // save the train,validation and test splits into the split folder
                    DataSet trainDS = new DataSet(train);
                    trainDS.SaveToFile(foldDir + File.separator + dsName + "_TRAIN");
                    DataSet validationDS = new DataSet(validation);
                    validationDS.SaveToFile(foldDir + File.separator + dsName + "_VALIDATION");
                    DataSet testDS = new DataSet(test);
                    testDS.SaveToFile(foldDir + File.separator + dsName + "_TEST");
                    
                    train = test = null;
                    trainDS = testDS = validationDS = null; 
                    
                    System.gc();
                }            
                
                Logging.println("Saved splits into TRAIN, VALIDATION, TEST", Logging.LogLevel.DEBUGGING_LOG);
            }            
        }
        
        
        public void GenerateFoldedRegressionDataSets(String dsDir)
        {
            int numFolds = 5;
            File dir = new File(dsDir);
        
            File [] dsFolders = dir.listFiles();

            String s = File.separator;

            for(File dsFolder : dsFolders)
            {
                String dsName = dsFolder.getName();
                
                // load the dataset 
                List<List<Double>> data = IOUtils.LoadFile(dsDir + s + dsName + s + dsName + ".csv");
                
                Matrix ds = new Matrix();
                ds.LoadRegressionData(data);
                
                Logging.println("Loaded: " + dsName + ", " + ds.getDimRows() + " instances", 
                        Logging.LogLevel.DEBUGGING_LOG); 
                
                // convert to weka instances format
                dataSetWeka = ds.ToWekaInstances();
                // randomize the instances
                dataSetWeka.randomize(rand);
                // split into stratified folds
                dataSetWeka.stratify(numFolds); 
                
                
                // create a folder to hold the splits
                File foldsDir = new File(dsFolder + File.separator + "folds");
                // delete the folder and all the contents
                IOUtils.Remove(foldsDir);
                // (re)create the folds folder
                foldsDir.mkdir();
                
                for(int fold = 0; fold < numFolds; fold++)
                {
                    // divide the dataset into a train and test set
                    Instances test = dataSetWeka.testCV(numFolds, fold);
                    Instances train = new Instances(dataSetWeka.trainCV(numFolds, fold));
                    
                    // stratify the train into #-1 folds and take one out as validation
                    train.randomize(rand);
                    train.stratify(numFolds-1);
                    Instances validation = train.testCV(numFolds-1, 0);
                    train = train.trainCV(numFolds-1, 0);
                    
                    File foldDir = new File(foldsDir + File.separator + String.valueOf(fold+1) ); 
                    // delete the fold folder and all the contents
                    IOUtils.Remove(foldDir);
                    // create a folder
                    foldDir.mkdir();
                    
                    CSVSaver csvSaver = new CSVSaver();
                    csvSaver.setDir(foldDir.getAbsolutePath());
                    
                    try
                    {
	                    csvSaver.setFile( new File( foldDir + File.separator + dsName + "_TRAIN" ) );
	                    csvSaver.setInstances(train);
                    	csvSaver.writeBatch();
                        csvSaver.resetWriter();
                        
                        csvSaver.setFile( new File( foldDir + File.separator + dsName + "_VALIDATION" ) );
	                    csvSaver.setInstances(validation);                    
                    	csvSaver.writeBatch();
                    	csvSaver.resetWriter();
                    	
                    	csvSaver.setFile( new File( foldDir + File.separator + dsName + "_TEST" ) );
	                    csvSaver.setInstances(test);                    
                    	csvSaver.writeBatch();
                    	csvSaver.resetWriter();
                    }
                    catch(Exception exc)
                    {
                    	Logging.println(exc.getMessage(), LogLevel.ERROR_LOG);
                    }
 
                    
                    try
                    {
	                    
                    }
                    catch(Exception exc)
                    {
                    	Logging.println(exc.getMessage(), LogLevel.ERROR_LOG);
                    }
                    csvSaver.resetWriter();
                    
                    // write the test set
                    try
                    {
	                    csvSaver.setFile( new File( foldDir + File.separator + dsName + "_TEST" ) );
	                    
	                    csvSaver.setInstances(test);
                    
                    	csvSaver.writeBatch();
                    }
                    catch(Exception exc)
                    {
                    	Logging.println(exc.getMessage(), LogLevel.ERROR_LOG);
                    }
                    csvSaver.resetWriter();
                    
                    
                }            
                
                Logging.println("Saved splits into TRAIN, VALIDATION, TEST", Logging.LogLevel.DEBUGGING_LOG);
            }            
        }
        // the main function does 
        public static void main(String [] args)
        {
            if( args.length == 0 )
            	args = new String[]{"/mnt/vartheta/Data/regression/"}; 
                        	
            Logging.currentLogLevel = Logging.LogLevel.DEBUGGING_LOG; 
             
            KFoldGenerator kFoldGenerator = new KFoldGenerator();
            //kFoldGenerator.GenerateFoldedDataSets(args[0]);
            kFoldGenerator.GenerateFoldedRegressionDataSets(args[0]);
            
        }
	
	
}
