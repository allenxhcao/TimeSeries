package weka.classifiers.functions.supportVector;
 
import TimeSeries.DTW;
import TimeSeries.EuclideanDistance;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
 
import java.util.Enumeration;
import java.util.Vector;
 
/**
 <!-- globalinfo-start -->
 * The polynomial kernel : K(x, y) = <x, y>^p or K(x, y) = (<x, y>+1)^p
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D_i
 *  Enables debugging output (if available) to be printed.
 *  (default: off)</pre>
 * 
 * <pre> -no-checks
 *  Turns off all checks - use with caution!
 *  (default: checks on)</pre>
 * 
 * <pre> -C <num>
 *  The size of the cache (a prime number), 0 for full cache and 
 *  -1 to turn it off.
 *  (default: 250007)</pre>
 * 
 * <pre> -E_i <num>
 *  The Exponent to use.
 *  (default: 1.0)</pre>
 * 
 * <pre> -L
 *  Use lower-order terms.
 *  (default: no)</pre>
 * 
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Shane Legg (shane@intelligenesis.net) (sparse vector code)
 * @author Stuart Inglis (stuart@reeltwo.com) (sparse vector code)
 * @version $Revision: 5807 $
 */
public class DTWKernel 
  extends CachedKernel {
 
  /** for serialization */
  static final long serialVersionUID = -321831645846363201L;
   
  /** Use lower-order terms? */
  protected boolean m_lowerOrder = false;
 
  /**
   * default constructor - does nothing.
   */
  public DTWKernel() 
  {
    super();
  }
 
  /**
   * Frees the cache used by the kernel.
   */
  public void clean() {
    super.clean();
  }
   
  /**
   * Creates a new <code>DTWKernel</code> instance.
   * 
   * @param data    the training dataset used.
   * @param cacheSize   the size of the cache (a prime number)
   * @param lowerOrder  whether to use lower-order terms
   * @throws Exception  if something goes wrong
   */
  public DTWKernel(Instances data, int cacheSize,
            boolean lowerOrder) throws Exception {
         
    super();
     
    setCacheSize(cacheSize);
 
    buildKernel(data);
  }
   
  /**
   * Returns a string describing the kernel
   * 
   * @return a description suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return 
        "The polynomial kernel : K(x, y) = <x, y>^p or K(x, y) = (<x, y>+1)^p";
  }
   
  /**
   * Returns an enumeration describing the available options.
   *
   * @return        an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector      result;
    Enumeration     en;
     
    result = new Vector();
 
    en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());
 
    result.addElement(new Option(
    "\tThe Exponent to use.\n"
    + "\t(default: 1.0)",
    "E_i", 1, "-E_i <num>"));
 
    result.addElement(new Option(
    "\tUse lower-order terms.\n"
    + "\t(default: no)",
    "L", 0, "-L"));
 
    return result.elements();
  }
 
  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -D_i
   *  Enables debugging output (if available) to be printed.
   *  (default: off)</pre>
   * 
   * <pre> -no-checks
   *  Turns off all checks - use with caution!
   *  (default: checks on)</pre>
   * 
   * <pre> -C <num>
   *  The size of the cache (a prime number), 0 for full cache and 
   *  -1 to turn it off.
   *  (default: 250007)</pre>
   * 
   * 
   <!-- options-end -->
   * 
   * @param options     the list of options as an array of strings
   * @throws Exception  if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
     
    super.setOptions(options); 
  }
 
  /**
   * Gets the current settings of the Kernel.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
     return new String[]{""};       
  }
 
  /**
   * 
   * @param id1     the index of instance 1
   * @param id2     the index of instance 2
   * @param inst1   the instance 1 object
   * @return        the dot product
   * @throws Exception  if something goes wrong
   */
  protected double evaluate(int id1, int id2, Instance inst1)
    throws Exception {
         
    double result;
    
    if (id1 == id2) 
    {
      result = 0;
    } 
    else 
    {
      result = - DTW.getInstance().CalculateDistance(inst1, m_data.instance(id2));
      //result = - EuclideanDistance.getInstance().CalculateDistance(inst1, m_data.instance(id2));
    }

    return result;
  }
 
  /** 
   * Returns the Capabilities of this kernel.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();
     
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enableAllClasses();
    result.enable(Capability.MISSING_CLASS_VALUES);
     
    return result;
  }
 
  /**
   * Returns the tip text for this property
   * 
   * @return        tip text for this property suitable for
   *            displaying in the explorer/experimenter gui
   */
  public String useLowerOrderTipText() {
    return "Whether to use lower-order terms.";
  }
   
  /**
   * returns a string representation for the Kernel
   * 
   * @return        a string representaiton of the kernel
   */
  public String toString() {
    return "";
  }
   
  /**
   * Returns the revision string.
   * 
   * @return        the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1 $");
  }
}