/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ToolkitGui;

import DataStructures.DataSet;
import TimeSeries.Distorsion;
import Utilities.Logging;
import java.io.File;
import javax.swing.DefaultListModel;
import javax.swing.JFileChooser;
import javax.swing.JFrame;

/**
 * The data that toolkit is using
 * @author josif
 */
public class ToolkitData 
{
    // singleton implementations
    private static ToolkitData instance=null;
    
    private ToolkitData()
    {
    }
    
    public static ToolkitData getInstance()
    {
        if(instance == null)
            instance = new ToolkitData();
        
        return instance;
    }
    
    // the currently opened dataset
    DataSet ds;
    
    public void LoadDataSet(JFrame parentFrame)
    {
        JFileChooser chooser = new JFileChooser(); 
        chooser.setCurrentDirectory(new java.io.File("~"));
        chooser.setDialogTitle("Select dataset dolder");
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        //
        // disable the "All files" option.
        //
        chooser.setAcceptAllFileFilterUsed(false);

        File dataSetDirectory = null;
        
        //    
        if (chooser.showOpenDialog(parentFrame) == JFileChooser.APPROVE_OPTION) 
        {

            dataSetDirectory = chooser.getSelectedFile();
            
            Logging.println("Selected dataset: " + dataSetDirectory.getAbsolutePath(), Logging.LogLevel.INFORMATIVE_LOG);
        }
        else
        {
            Logging.println("No dataset selected.", Logging.LogLevel.WARNING_LOG);
        }
	    
        String dataSetDirectoryStr = dataSetDirectory.getAbsolutePath();
	    

        // then load the data by the cm manager
        try
        {
            ds = new DataSet();
            ds.LoadDataSetFolder(dataSetDirectory);

            ds.NormalizeDatasetInstances();
        }
        catch(Exception exc)
        {
            Logging.println(exc.getMessage(), Logging.LogLevel.ERROR_LOG);
        }
    }
}
