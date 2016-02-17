/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Experiments;

import DataStructures.DataInstance;
import TimeSeries.MLS;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author josif
 */
public class ExampleMLS 
{
    
    public static void run(DataInstance ins)
    {
       
        int numFeatures = ins.features.size();
        
        int [] transformationField = new int[]
        {  +20, -10};
        
        List<Integer> oldCP = new ArrayList<Integer>();
        oldCP.add(0);
        oldCP.add(numFeatures/3);
        oldCP.add((2*numFeatures)/3);
        oldCP.add(numFeatures-1);
        
        List<Integer> newCP = new ArrayList<Integer>();
        newCP.add(0);
        newCP.add(numFeatures/3 + transformationField[0]);
        newCP.add((2*numFeatures)/3 + transformationField[1]);
        newCP.add(numFeatures-1);
        
        DataInstance transformed = MLS.getInstance().Transform(
                            ins, oldCP, newCP);
        
        
        System.out.println("cp at: " + numFeatures/3  + " " + (2*numFeatures)/3);
        
        System.out.print("original=[");
        for(int i = 0; i < ins.features.size(); i++)
        {
            System.out.print(i + " " + ins.features.get(i).value + "; ");
        }
        
        System.out.println("];");
        
        
        System.out.print("transformed=[");
        
        for(int i = 0; i < transformed.features.size(); i++)
        {
            System.out.print(i + " " + transformed.features.get(i).value + "; ");
        }
        
        System.out.println("];");
        
        
    }
}
