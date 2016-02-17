package Experiments;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint.PointStatus;
import Utilities.Logging;
import java.util.Random;


/*
 * The class is eclipsing a collection of time series
 * 
 */

public class MissingPointsGenerator 
{
    public boolean avoidExtremes = false;
    
    boolean [] selectedTrajectoriesFlags;

    public MissingPointsGenerator()
    {
            selectedTrajectoriesFlags = null;
            avoidExtremes = false;
    }

    public void GenerateMissingPoints( DataSet dataSet, double totalMissingRatio, double gapRatio)
    {
        // iterate through all the selected trajectories and then 
        for(int i = 0; i < dataSet.instances.size(); i++)
        {
            GenerateMissingPoints(dataSet.instances.get(i), totalMissingRatio, gapRatio);
        }
    }

    /*
     * Generate the missing points of an instance
     */
    public void GenerateMissingPoints(DataInstance instance, double missingRatio, double gapRatio)
    {
        // get a list of point indexes to label as missing ommiting extremes
        boolean [] missingPointIndexes = null;

        int noTrajectoryPoints = instance.features.size();
        int noPointsMissing =  (int) Math.ceil( noTrajectoryPoints * missingRatio );

        // if the chunk size ratio is zero then create random index list
        if( gapRatio == 0.0 )
        {
            missingPointIndexes = CreateRandomIndexList(noTrajectoryPoints, noPointsMissing, true);
        }
        // otherwise create random index ranges equivalent to multiples of a chunk size
        else
        {
            missingPointIndexes = CreateRandomIndexListByChunks(noTrajectoryPoints, missingRatio, gapRatio);
        }

        for(int j = 0; j < noTrajectoryPoints; j++)
        {
            if( missingPointIndexes[j] == true )
            {
                instance.features.get(j).status = PointStatus.MISSING;
            }
        }

    }

    /*
     * The method returns a boolean array that stores flags about selected indexes 
     * from a total number of elements. For example assume we have 10 elements and we have to 
     * choose 3 random indexes. Say we randomly chose indexes 0, 4, 8 the resulting array will be 
     * [ true, false, false, false, true, false, false, false, true, false, false ] 
     * 
     */
    public static boolean[] CreateRandomIndexList( int noTotalElements, int noIndexesToPick, boolean ommitExtremes )
    {
        boolean [] pickedIndexes = new boolean[noTotalElements];

        Random generator = new Random();

        // initialize the flags to zero
        for(int i = 0; i < noTotalElements; i++) 
                pickedIndexes[i] = false;

        int noSelected = 0;

        // select the trajectories to be affected randomly
        // until the predefined no of trajectories to be affected is reached
        while( noSelected < noIndexesToPick )
        {
            // pick a random trajectory index
            int index = generator.nextInt(noTotalElements);

            // if the trajectory was not already selected then select it 
            // otherwise loop again for another one
            if( pickedIndexes[index] == false )
            {
                    // ommit extreme indexes if the parameter flag is true
                    if( !(ommitExtremes == true && (index == 0 || index == noTotalElements-1) ))
                    {
                            pickedIndexes[index] = true;
                            noSelected++;
                    }
            }
        } 

        return pickedIndexes;
    }

    /*
     * Create a random index list by deleting whole chunks of data proportional to the 
     * parameter ratio gap ratio size, i.e. a gap ratio of 0.1 means each gap will be by 10% of length
     */
    public boolean[] CreateRandomIndexListByChunks( 
                    int noTotalElements, 
                    double missingPointsRatio, 
                    double gapSizeRatio )
    {
        boolean [] pickedIndexes = new boolean[noTotalElements];
        // initialize the flags to zero
        for(int i = 0; i < noTotalElements; i++) 
                pickedIndexes[i] = false;

        Random generator = new Random();

        // the number of chunks is calculated as the ratio of the trajectory
        int noChunksToPick = (int) Math.ceil(missingPointsRatio / gapSizeRatio);
        // the total possible chunks is the inverse of the gap ratio size,
        // i.e. if gap ration is 10% meaning 0.1 than there are 1 / 0.1 = 10 total gap chunks
        int noTotalChunks = (int) (1.0 / gapSizeRatio);

        // the total number of points in a chunk is the length times the gap ratio size
        int chunkSize = (int) (noTotalElements * gapSizeRatio);

        // an array of flags denoting flags of picked chunks
        boolean [] pickedChunkIndexes = new boolean[noTotalChunks];
        // initialize the flags to zero
        for(int i = 0; i < noTotalChunks; i++) 
                pickedChunkIndexes[i] = false;

        
        
        
        Logging.println("noTotalPoints=" + noTotalElements +
                        ", missingPointsRatio=" + missingPointsRatio +
                        ", gapSizeRatio=" + gapSizeRatio +
                        ", noTotalChunks=" + noTotalChunks + 
                        ", noChunksToPick=" + noChunksToPick + 
                        ", chunkSize=" + chunkSize, Logging.LogLevel.DEBUGGING_LOG); 

        int noSelectedChunks = 0;

        // select the trajectories to be affected randomly
        // until the predefined no of trajectories to be affected is reached
        while( noSelectedChunks < noChunksToPick )
        {
            // pick a random trajectory index
            int index = generator.nextInt( noTotalChunks );

            // check whether to ommit the point
            boolean isOmmited = false;
            
            // flag to avoid extremes if set so
            if( avoidExtremes == true && (index == 0 || index == noTotalChunks-1) )
            {
               isOmmited = true;
            }
            
            // if the trajectory was not already selected then select it 
            // otherwise loop again for another one
            if( pickedChunkIndexes[index] == false && isOmmited == false )
            {
                // set the range of the chunk on the time vim series points scale
                int startMissingPointsIndex = index * chunkSize;
                int endMissingPointsIndex = startMissingPointsIndex + chunkSize;

                // make sure the end index doesn't exceed the last point
                if( endMissingPointsIndex > noTotalElements)
                        endMissingPointsIndex = noTotalElements;

                // if the no of points are not a multiple of the chunk size there will be
                // some remainder points in the end, and in case we picked the last chunk
                // then we should include those points as well
                if( noTotalElements - endMissingPointsIndex < chunkSize )
                {
                        endMissingPointsIndex = noTotalElements;
                }

                for( int i = startMissingPointsIndex; i < endMissingPointsIndex; i++)
                {
                        pickedIndexes[i] = true;
                }	

                Logging.println("Picked points interval " + startMissingPointsIndex + " to "
                                + endMissingPointsIndex, Logging.LogLevel.DEBUGGING_LOG );

                pickedChunkIndexes[index] = true;

                noSelectedChunks++;

            }
        }

        return pickedIndexes;
    }
	

	
}
