package FeatureRich;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.SortedSet;
import java.util.StringTokenizer;
import java.util.TreeSet;

import Utilities.Logging;
import Utilities.StatisticalUtilities;

public class FeatureRichDataset 
{
	public double [][][][] S;
	public double [][] Y;
	
	// number of instance
	public int I;
	// number of classes
	public int C;
	
	// number of scales
	public int R;
		
	// length of pattern at different scales
	public int [] L; 
	
	public void LoadDataset(String datasetFilePath)
	{
		try
		{
			// first count the number of instances and the number of classes
			BufferedReader br1 = new BufferedReader(new FileReader( datasetFilePath ));
			
			String delimiters = "\t ,;";
            String line;
            
            SortedSet<Integer> instanceIdsSet = new TreeSet<Integer>();
            SortedSet<Integer> classIdsSet = new TreeSet<Integer>();
            SortedSet<Integer> scaleIdsSet = new TreeSet<Integer>(); 
            
			while( (line=br1.readLine()) != null)
			{
				StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
	            // parse the instance id, class id and scale id fields
				int instanceId = (int) Double.parseDouble( tokenizer.nextToken() );
				int classId = (int) Double.parseDouble( tokenizer.nextToken() );
				int scaleId = (int) Double.parseDouble( tokenizer.nextToken() );
				
				// add them to the set
				instanceIdsSet.add(instanceId);
				classIdsSet.add(classId);
				scaleIdsSet.add(scaleId);				
			}
			
			// initialize the datasets
			I = instanceIdsSet.size();
			C = classIdsSet.size(); 
			R = scaleIdsSet.size();
			
			// initialize the storage arrays
			S = new double[I][R][][];
			Y = new double[I][C];
			L = new int[R];
			
			// set all labels to 0
			for(int i=0; i<I; i++)
				for(int c=0; c<C; c++)
					Y[i][c] = 0;
			
			br1.close(); 
			
			Logging.println("numInstances="+I+", numScales="+R+", numClasses=" + C); 
			
			// first count the number of instances and the number of classes
			BufferedReader br2 = new BufferedReader(new FileReader( datasetFilePath ));
			
			while( (line=br2.readLine()) != null)
			{
				StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
				
				int i = (int) Double.parseDouble( tokenizer.nextToken() );
				int c = (int) Double.parseDouble( tokenizer.nextToken() );
				int r = (int) Double.parseDouble( tokenizer.nextToken() );
				int J_ir = (int) Double.parseDouble( tokenizer.nextToken() );
				L[r] = (int) Double.parseDouble( tokenizer.nextToken() );
				
				// set the c-th index label to one
				Y[i][c] = 1.0;
				
				// initialize the segments of the r-th scale of the i-th instance
				S[i][r] = new double[J_ir][ L[r] ];  
				
				for(int j=0; j<J_ir; j++)
				{
					for(int l=0; l< L[r]; l++)
						S[i][r][j][l] = Double.parseDouble( tokenizer.nextToken() );
					
					//S[i][r][j] = StatisticalUtilities.Normalize(S[i][r][j]);
				} 
			}
			
			br2.close();
			
			Logging.println("L:");
			Logging.println(L); 
			
			Logging.println("Finished initializing.");  
		}
		catch(Exception exc)
		{
			exc.printStackTrace();
		}
	}
	
}
