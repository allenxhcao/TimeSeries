/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Classification;

/**
 * 
 * A hyperparameter definition 
 * 
 * @author Josif Grabocka
 */
public class HyperParameter 
{
    String name;
    double minValue;
    double maxValue;
    
    double currentValue;
    
    // how to increment the value of the parameter 
    // either as addition or multiplication
    public enum IncrementType { MULTIPLY, ADD } ;
    
    public IncrementType incrementType;
    public double incrementStep;
    
    
    public HyperParameter(
            String paramName, 
            double paramMinValue, double paramMaxValue,
            IncrementType paramIncrementType)
    {
        name = paramName;
        minValue = paramMinValue;
        maxValue = paramMaxValue;
        
        incrementType = paramIncrementType;
        
        currentValue = minValue;
        incrementStep = 0;
    }
    
    /*
     * set the increment step 
     */
    public void SetIncrementStep(double paramIncrementStep)
    {
        incrementStep = paramIncrementStep;
    }
    
    public boolean validateValue(double paramValue)
    {
        return paramValue <= minValue && paramValue >= maxValue;
    }
    
    /*
     * Set the range of a hyperparameter
     */
    public void SetRange(double paramMinValue, double paramMaxValue)
    {
        minValue = paramMinValue;
        maxValue = paramMaxValue;
    }
    
    /*
     * Get the next value
     */
    
    public double NextValue()
    {
        if( incrementType == IncrementType.ADD )
            currentValue = currentValue + incrementStep;
        if( incrementType == IncrementType.MULTIPLY )
            currentValue = currentValue * incrementStep;
        
        return currentValue;
    }
    
}
