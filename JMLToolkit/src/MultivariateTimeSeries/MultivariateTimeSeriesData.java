package MultivariateTimeSeries;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import Utilities.Logging;

public class MultivariateTimeSeriesData 
{
	// the number of train instances and total instances
	public int NTrain, N;
	// number of channels
	public int C;
	// the stored data
	public double [][][] T;
	// names of the series
	public String [] seriesNames; 
	
	// number of segments per series
	public int[] J;
	// the length of a segment
	int L;
	// the start offset increment
	int delta;
	
	public double[] trainLabels; 
	public double[] testLabels; 
	
	// the number of series
	public int getNumSeries()
	{
		return N;
	}
	
	public int getNumTrainSeries()
	{
		return NTrain;
	}
	
	public int getNumChannels()
	{
		return C;
	}
	
	// get a time series value: for the i-th instance, c-th channel, j-th segment at relative index l
	public double getValue(int i, int c, int j, int l)
	{
		return T[i][c][j*delta + l]; 
	}
	
	// return the number of segments of the i-th time series
	public int getNumSegments(int i)
	{
		return J[i];
	}

	// load the data in the folder, containing train and test subfolders
	public int loadFolder(String folder, int startGlobalIdx, boolean onlyReadNumInstance)
	{
		// load train data
		File dir = new File(folder); 
		
		Logging.println("Data folder: " + dir.getAbsolutePath()); 
		
		int globalInstanceIdx = startGlobalIdx;
		
		File [] files = dir.listFiles();
		for (int i = 0; i < files.length; i++)
		{
	        if (files[i].isFile())
	        { 
	        	String fileNameWithOutExt = files[i].getName().replaceFirst("[.][^.]+$", "");
	        	
	        	Logging.println("Loading: " + files[i].getAbsolutePath() + "...");
	        	
	        	// create a buffered reader and read the file
	        	try
	        	{
	        		BufferedReader br = new BufferedReader(new FileReader(files[i].getAbsoluteFile()));
	        		
	        		String line = null;
	        		int currLineIndex=-1;
	        		int feedbackEventNumber=0;
	        		int startFeedbackIndex = -1;
	        		
	        		// accumulate the data of one series
	        		List<String []> fieldsAccumulator = new ArrayList<String []>();
	        		
	        		// ommit header line
	        		line = br.readLine();
	        		// read all lines
	        		while ((line = br.readLine()) != null) 
	        		{
	        	        // use comma as separator
		    			String[] fields = line.split(",");
		    			
		    			// if a feedback flag is seen 
		    			if( Integer.parseInt(fields[fields.length-1]) == 1 )
		    			{
		    				// if not the first feedback flag than store the feedback lines
		    				// as an instance
		    				if( feedbackEventNumber > 0)
		    				{
		    					String instanceName = fileNameWithOutExt + "_FB" + String.format("%03d", feedbackEventNumber);
		    				
		    					Logging.println(globalInstanceIdx + ": " + instanceName + 
		    							", Lines: "  + startFeedbackIndex + " - " + currLineIndex);
		    					
		    					// initialize instance
		    					if(!onlyReadNumInstance)
		    						InitializeInstance(globalInstanceIdx, instanceName, fieldsAccumulator);
		    					// flush the accumulator
		    					fieldsAccumulator.clear();
		    					// increment the global instance ids
		    					globalInstanceIdx++;
		    				}
		    				
		    				// clear any accumulated lines prior to the first feedback flag
		    				if(feedbackEventNumber == 0)
		    					fieldsAccumulator.clear();
		    				
		    				feedbackEventNumber++;
		    				startFeedbackIndex = currLineIndex;
		    			}
		    			
		    			fieldsAccumulator.add(fields);
		    			currLineIndex++;
	        		}
	        		
	        		// the last feedback has occurred
	        		if( !fieldsAccumulator.isEmpty() )
	        		{
	        			String instanceName = fileNameWithOutExt + "_FB" + String.format("%03d", feedbackEventNumber);
	        			
	        			Logging.println(globalInstanceIdx + ": " + instanceName + 
    							", Lines: "  + startFeedbackIndex + " - " + currLineIndex);
    					
	        			// initialize instance
	        			if(!onlyReadNumInstance)
	        				InitializeInstance(globalInstanceIdx, instanceName, fieldsAccumulator);
    					// flush the accumulator
    					fieldsAccumulator.clear();
	        			// increment the global instance index
    					globalInstanceIdx++;
	        		}
	        		
	        		br.close();
	        	}
	        	catch(Exception exc)
	        	{
	        		exc.printStackTrace();
	        	}
	        }
	    }
		
		return globalInstanceIdx;
	}
	
	// initialize a series index 
	public void InitializeInstance(int i, String instanceName, List<String []> fieldsAccumulator)
	{
		for(int c=0; c<C; c++)
		{
			T[i][c] = new double[fieldsAccumulator.size()]; 
			
			for(int m=0; m<fieldsAccumulator.size(); m++)
			{
				T[i][c][m] = Double.parseDouble( fieldsAccumulator.get(m)[c+1] ); 
			}
			
			//T[i][c] = Utilities.StatisticalUtilities.Normalize(T[i][c]); 
		}
		
		seriesNames[i] = instanceName;
	}
	
	public void computeNumberOfSegments(int segmentLength)
	{
		L = segmentLength;
		delta = segmentLength/2;
		
		J = new int[N];
		
		for(int i=0; i < N; i++)
		{
			J[i] = 0;
			
			for(int m=0; m < T[i][0].length - L - 1; m+=delta)
				J[i]++;
			
			System.out.println("i=" + i + ", name="+ seriesNames[i] + ", M_i=" + T[i][0].length + ", J_i=" + J[i]); 
		}
		
	}
	
	public void loadData(String dataFolder)
	{
		// first only read the number of instances
		int lastTrainSeriesIdx  = loadFolder(dataFolder + File.separator + "train" + File.separator, 0, true);
		int lastSeriesIdx  = loadFolder(dataFolder + File.separator + "test" + File.separator, lastTrainSeriesIdx, true);
		
		N = lastSeriesIdx;
		NTrain = lastTrainSeriesIdx;
		C = 57; 
		
		Logging.println("NumTrainSeries=" + NTrain + ", NumTotalSeries=" + N + ", NumTestSeries=" + (N-NTrain));

		// initialize the data storage
		T = new double[N][C][];
		// initialize the vector of series names
		seriesNames = new String[N];
		
		// then load the data
		loadFolder(dataFolder + File.separator + "train" + File.separator, 0, false);
		loadFolder(dataFolder + File.separator + "test" + File.separator, lastTrainSeriesIdx, false);
		
	}
	
	public void loadLabels( String labelsFile )
	{
		trainLabels = new double[NTrain];
		
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(labelsFile));
			
			// ommit header line
			String line = br.readLine();
			// read all lines
			int i = 0;
			 
			while ((line = br.readLine()) != null) 
			{
		        // use comma as separator
				String[] fields = line.split(",");
				// read the instance name 
				String instanceName =  fields[0];
				trainLabels[i] = Double.parseDouble( fields[1] );  
				
				if( ("Data_" + instanceName).compareTo( seriesNames[i] ) != 0 ) 
				{
					Logging.println(instanceName + " != " + seriesNames[i] ); 
				}
				
				i++;
			} 
		}
		catch(Exception exc)
		{
			exc.printStackTrace();
		}
	}
}
