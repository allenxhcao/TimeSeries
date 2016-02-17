/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

import java.util.List;

/**
 *
 * @author Josif Grabocka
 */
public class HyperParameterSearch 
{
    double[] parameters;
    double[] stepLength;
    double[] min;
    double[] max;
    int number;
    
    // the model to be tuned
    Classifier model;
    
    public HyperParameterSearch(Classifier model) 
    {	
            number = model.hyperParameterDefinitions.size();
            parameters = new double[number];
            stepLength = new double[number];
            min = new double[number];
            max = new double[number];
    }
}
