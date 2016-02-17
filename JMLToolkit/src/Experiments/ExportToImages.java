/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Experiments;

import DataStructures.DataSet;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/**
 *
 * @author josif
 */
public class ExportToImages 
{
    // singleton
    private static ExportToImages instance = null;
    
    private ExportToImages()
    {
    }
    
    public static ExportToImages getInstance()
    {
        if(instance == null)
            instance = new ExportToImages();
        
        return instance;
    }
    
    
    public void Export(DataSet ds, String imagesFolder)
    {
        try
        {
            double dsMax = ds.GetOverallMaximum(),
                    dsMin = ds.GetOverallMinimum();
            
            ds.ReadNominalTargets();
            
            File dsFolder = new File(imagesFolder + "/" + ds.name);
            if(dsFolder.exists()) dsFolder.delete();            
            dsFolder.mkdir();
            
            for( int l = 0; l < ds.nominalLabels.size(); l++)
            {
                double label = ds.nominalLabels.get(l);
                File dslFolder = new File(imagesFolder + "/" + ds.name + "/" + label);
                if(dslFolder.exists()) dslFolder.delete();            
                dslFolder.mkdir();
            }
                
            for(int i = 0; i < ds.instances.size(); i++)
            {
                double label = ds.instances.get(i).target;
                String imageName = imagesFolder + "/" + ds.name + "/" + label + "/" + i;
                
                BufferedImage seriesImage = ds.instances.get(i).ConvertToImage(imageName, dsMax, dsMin);
                System.out.println(imageName);
                // Store the image using the PNG format.
                ImageIO.write(seriesImage,"PNG",new File( imageName + ".png"));
            }

        }
        catch(Exception exc)
        {
            exc.printStackTrace();
        }
    }
}
