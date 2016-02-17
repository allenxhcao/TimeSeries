package TimeSeries;

import java.io.File;
import java.util.ArrayList;
import java.util.List; 

public class UCRTimeSeriesCollection 
{
	public List<UCRTimeSeriesInfo> timeSeriesDataSetsFolders;
	
	public UCRTimeSeriesCollection(String ucrFolder)
	{
		timeSeriesDataSetsFolders = new ArrayList<UCRTimeSeriesInfo>();
		
                String s = File.separator;
                
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "50words" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Adiac" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Beef" + s), 2));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "CBF" + s), 16));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "ChlorineConcentration" + s), 9));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "CinC_ECG_torso" + s), 30));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Coffee" + s), 2));
                //timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/Cricket_X/"), 2));
                //timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/Cricket_Y/"), 2));
                //timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/Cricket_Z/"), 2));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "DiatomSizeReduction" + s), 10));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "ECG200" + s), 5));                
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "ECGFiveDays" + s), 26));  
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "FaceFour" + s), 5));
                //timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/FaceAll/"), 5));                         
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "FacesUCR" + s), 11));	
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Fish" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Gun_Point" + s), 5));                       	       	
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Haptics" + s), 5));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "InlineSkate" + s), 6));		
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "ItalyPowerDemand" + s), 8));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Lighting2" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Lighting7" + s), 2));                                
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "MALLAT" + s), 20));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "MedicalImages" + s), 5));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "MoteStrain" + s), 24));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "OliveOil" + s), 2));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "OSULeaf" + s), 5));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "SonyAIBORobot_Surface" + s), 16));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "SonyAIBORobot_SurfaceII" + s), 12)); 
                // timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/StarLightCurves/"), 9));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "SwedishLeaf" + s), 5));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Symbols" + s), 30));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "synthetic_control" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Trace" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "Two_Patterns" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "TwoLeadECG" + s), 25));
                //timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/uWaveGestureLibrary_X/"), 5));
                //timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/uWaveGestureLibrary_Y/"), 5));
                //timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + "/uWaveGestureLibrary_Z/"), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "wafer" + s), 7));
                timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "WordsSynonyms" + s), 5));
		timeSeriesDataSetsFolders.add( new UCRTimeSeriesInfo( new File( ucrFolder + s + "yoga" + s), 11));
                
	}
	
	
	// get the advised fold number
	public int GetAdvisedFoldNumber(String dataSetName)
	{
		int foldNo = 0;
		
		for( UCRTimeSeriesInfo tsInfo : timeSeriesDataSetsFolders )
		{
			if( tsInfo.timeSeriesLocation.getName().compareTo(dataSetName) == 0)
			{
				foldNo = tsInfo.advisedCrossFold; 
			}
		}
			
		return foldNo;
	}
	
}
