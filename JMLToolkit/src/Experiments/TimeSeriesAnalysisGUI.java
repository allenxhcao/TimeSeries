package Experiments;

import Classification.MFSVM;
import Classification.ModelBasedImputation;
import DataStructures.DataInstance;
import DataStructures.DataSet;
import DataStructures.FeaturePoint;
import DataStructures.FeaturePoint.PointStatus;
import DataStructures.Matrix;
import MatrixFactorization.SupervisedMatrixFactorization;
import MatrixFactorization.CollaborativeImputation;
import MatrixFactorization.MatrixFactorizationModel;
import TimeSeries.*;
import Utilities.ExpectationMaximization;
import Utilities.GlobalValues;
import info.monitorenter.gui.chart.Chart2D;
import info.monitorenter.gui.chart.IRangePolicy;
import info.monitorenter.gui.chart.IAxis.AxisTitle;
import info.monitorenter.gui.chart.ITrace2D;
import info.monitorenter.gui.chart.rangepolicies.RangePolicyFixedViewport;
import info.monitorenter.gui.chart.traces.Trace2DSimple;
import info.monitorenter.util.Range;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.text.DecimalFormat;
import java.util.*;

import javax.swing.*;
import javax.swing.border.BevelBorder;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import weka.core.matrix.EigenvalueDecomposition;

public class TimeSeriesAnalysisGUI {

	public JFrame frmTimeSeries;
	private JTextField missingRatioTextField;
	private JTextField gapSizeRatioTextField;
	private final JList timeSeriesList = new JList();
	private JScrollPane timeSeriesScrollPane;
	JPanel chartPanel; 
	
	File dataSetDirectory;
	DataSet dataSet; 
    DataSet sparseDataSet; 
    CollaborativeImputation ci;
    ExpectationMaximization em;
    ModelBasedImputation mbi;
        //String imputationType = "linear";
        //String imputationType = "bspline"; 
	Chart2D tsChart;
	ITrace2D originalTimeSeriesTrace;
	ITrace2D interpolatedTimeSeriesTrace;
        
    ITrace2D currentEigenSeries;
    int currentEigenSeriesIndex;
    MatrixFactorizationModel factorizationModel;

	DataInstance originalTimeSeries;
	DataInstance manipulatedTimeSeries;
        
	private JTextField labelTextField;
	private JTextField datasetTextField;
	private JTextField noPointsTextField;
        private JTextField numEigenSeries;
        private JComboBox interpolationTechnique;
	private JCheckBox imputeCheckBox;
	private JCheckBox distortDataSetCheckBox;
	private JCheckBox enableSupervisionCheckBox;
	private JCheckBox enableWarpingCheckBox;
	
	/**
	 * Create the application.
	 */
	public TimeSeriesAnalysisGUI() 
        {
            initialize();
            manipulatedTimeSeries = null;
            ci = null;
            em = null;
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() 
        {
            frmTimeSeries = new JFrame();
            frmTimeSeries.setTitle("Time Series Analysis");
            frmTimeSeries.setBounds(100, 100, 1104, 757);
            frmTimeSeries.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frmTimeSeries.getContentPane().setLayout(null);

            missingRatioTextField = new JTextField();
            missingRatioTextField.setText("0.3");
            missingRatioTextField.setBounds(328, 12, 114, 19);
            frmTimeSeries.getContentPane().add(missingRatioTextField);
            missingRatioTextField.setColumns(10);

            JLabel lblMissingRatio = new JLabel("Missing Ratio");
            lblMissingRatio.setBounds(209, 16, 101, 15);
            frmTimeSeries.getContentPane().add(lblMissingRatio);

            JLabel lblGapSizeRatio = new JLabel("Gap Size Ratio");
            lblGapSizeRatio.setBounds(209, 47, 122, 15);
            frmTimeSeries.getContentPane().add(lblGapSizeRatio);

            JLabel lblInterpolation = new JLabel("Interpolation:");
            lblInterpolation.setBounds(459, 16, 122, 15);
            frmTimeSeries.getContentPane().add(lblInterpolation);
            
            String[] interpolationTechniques = { 
                "linear", 
                "cubicspline", 
                "bspline", 
                "collaborative",
                "em",
                "mbi"}; 
            
            interpolationTechnique = new JComboBox(interpolationTechniques);
            //interpolationTechnique.setBounds();
            interpolationTechnique.setBounds(559, 16, 142, 25);
            frmTimeSeries.getContentPane().add(interpolationTechnique);
            
            
            JLabel eigenSeriesInterpolation = new JLabel("Num.EigenS.");
            eigenSeriesInterpolation.setBounds(459, 47, 122, 15);
            frmTimeSeries.getContentPane().add(eigenSeriesInterpolation);
            
            numEigenSeries = new JTextField();
            numEigenSeries.setBounds(559, 47, 142, 25);            
            frmTimeSeries.getContentPane().add(numEigenSeries);
            
            
            imputeCheckBox = new JCheckBox("Impute", false);
            imputeCheckBox.setBounds(720, 16, 80, 15);
            frmTimeSeries.getContentPane().add(imputeCheckBox);
            
            distortDataSetCheckBox = new JCheckBox("Distort", false);
            distortDataSetCheckBox.setBounds(720, 47, 80, 15);
            frmTimeSeries.getContentPane().add(distortDataSetCheckBox);
            
            enableSupervisionCheckBox = new JCheckBox("Supervised", false);
            enableSupervisionCheckBox.setBounds(820, 16, 80, 15);
            frmTimeSeries.getContentPane().add(enableSupervisionCheckBox);
            
            enableWarpingCheckBox = new JCheckBox("Warp", true);
            enableWarpingCheckBox.setBounds(820, 47, 80, 15);
            frmTimeSeries.getContentPane().add(enableWarpingCheckBox);
            
            gapSizeRatioTextField = new JTextField();
            gapSizeRatioTextField.setText("0.1");
            gapSizeRatioTextField.setColumns(10);
            gapSizeRatioTextField.setBounds(328, 43, 114, 19);
            frmTimeSeries.getContentPane().add(gapSizeRatioTextField);
            timeSeriesList.addListSelectionListener(new ListSelectionListener() {
                    public void valueChanged(ListSelectionEvent arg0) {
                            ChangeSelectedTrajectory();
                    }
            });
            timeSeriesList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
            timeSeriesList.setModel(new AbstractListModel() {
                    String[] values = new String[] {};
                    public int getSize() {
                            return values.length;
                    }
                    public Object getElementAt(int index) {
                            return values[index];
                    }
            });
            timeSeriesList.setBorder(new EtchedBorder(EtchedBorder.LOWERED, null, null));
            timeSeriesList.setBounds(12, 84, 130, 613);

            timeSeriesScrollPane = new JScrollPane(timeSeriesList);
            timeSeriesScrollPane.setBounds( timeSeriesList.getBounds() );

            //frmTimeSeries.getContentPane().add(timeSeriesList);
            frmTimeSeries.getContentPane().add(timeSeriesScrollPane);

            //timeSeriesScrollPane.getViewport().add(timeSeriesList);

            JLabel lblTimeseries = new JLabel("TimeSeries:");
            lblTimeseries.setBounds(12, 61, 83, 15);
            frmTimeSeries.getContentPane().add(lblTimeseries);

            JButton btnLoadDataset = new JButton("Load Dataset");
            btnLoadDataset.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent arg0) {
                            LoadDataSet();
                    }
            });
            btnLoadDataset.setBounds(12, 9, 130, 25);
            frmTimeSeries.getContentPane().add(btnLoadDataset);

            JButton generateButton = new JButton("Generate");
            generateButton.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent arg0) {
                            GenerateManipulatedTimeSeries();
                    }
            });
            generateButton.setBounds(550, 83, 150, 25);
            frmTimeSeries.getContentPane().add(generateButton);
            
            JButton factorizeButton = new JButton("Find EigenS");
            factorizeButton.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent arg0) {
                            FindEigenSeries();
                    }
            });
            factorizeButton.setBounds(700, 83, 150, 25);
            frmTimeSeries.getContentPane().add(factorizeButton);

            
            JButton bcwdButton = new JButton("<");
            bcwdButton.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent arg0) {
                            PreviousEigenSeries();
                    }
            });
            bcwdButton.setBounds(850, 83, 50, 25);
            frmTimeSeries.getContentPane().add(bcwdButton);
            
            JButton fwdButton = new JButton(">");
            fwdButton.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent arg0) {
                            NextEigenSeries();
                    }
            });
            fwdButton.setBounds(900, 83, 50, 25);
            frmTimeSeries.getContentPane().add(fwdButton);
            
            
            
            chartPanel = new JPanel();
            chartPanel.setBorder(new TitledBorder(new BevelBorder(BevelBorder.RAISED, null, null, null, null), "Time Series Chart", TitledBorder.LEADING, TitledBorder.TOP, null, null));
            chartPanel.setBounds(171, 119, 905, 451);
            frmTimeSeries.getContentPane().add(chartPanel);

            labelTextField = new JTextField();
            labelTextField.setEditable(false);
            labelTextField.setBounds(196, 634, 114, 19);
            frmTimeSeries.getContentPane().add(labelTextField);
            labelTextField.setColumns(10);

            datasetTextField = new JTextField();
            datasetTextField.setEditable(false);
            datasetTextField.setBounds(209, 86, 295, 19);
            frmTimeSeries.getContentPane().add(datasetTextField);
            datasetTextField.setColumns(10);

            noPointsTextField = new JTextField();
            noPointsTextField.setEditable(false);
            noPointsTextField.setBounds(196, 608, 114, 19);
            frmTimeSeries.getContentPane().add(noPointsTextField);
            noPointsTextField.setColumns(10);

            noPointsTextField.setVisible(false);
            datasetTextField.setVisible(false);
            labelTextField.setVisible(false);
	}
	
	public void LoadDataSet()
	{
	    JFileChooser chooser = new JFileChooser(); 
	    chooser.setCurrentDirectory(new java.io.File("/mnt/vartheta/Data/classification/"));
	    chooser.setDialogTitle("Pick a time series dataset Folder");
	    chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
	    //
	    // disable the "All files" option.
	    //
	    chooser.setAcceptAllFileFilterUsed(false);
	    
	    //    
	    if (chooser.showOpenDialog(frmTimeSeries) == JFileChooser.APPROVE_OPTION) 
            { 
	      
	    	dataSetDirectory = chooser.getSelectedFile();
	    	System.out.println(" Selected dataset directory:" + dataSetDirectory.getAbsolutePath() );
	    }
	    else 
            {
	      System.out.println("No Selection ");
	    }
	    
	    String dataSetDirectoryStr = dataSetDirectory.getAbsolutePath();
	    
	    datasetTextField.setVisible(true);
	    datasetTextField.setText("Dataset: " + dataSetDirectoryStr);
	    
	    // then load the data by the cm manager
	    try
	    {
            dataSet = new DataSet();
            dataSet.LoadDataSetFolder(dataSetDirectory);
            
            dataSet.NormalizeDatasetInstances();
            /*  
            // create a bag of patterns representation of the dataset and 
            // generate the histogram of pattenrs
            BagOfPatterns bop = new BagOfPatterns();
            int slidingWindowSize = 32;
            int innerDimension = 4;
            int alphabetSize = 8;
            
            Matrix ds = new Matrix();
            ds.LoadDatasetFeatures(dataSet, false);
            
            Matrix H = bop.CreateHistogramMatrix(ds, slidingWindowSize, innerDimension, alphabetSize);
            */
            
            if( distortDataSetCheckBox.isSelected() )
            {
                String tfFolder ="/home/josif/Documents/transformation_fields";

                TransformationFieldsGenerator.getInstance().transformationScale = 0.5;
                dataSet = Distorsion.getInstance().DistortTransformationField(dataSet, tfFolder);

                
                // dataSet.SaveToFile("/home/josif/"+dataSet.name+"_transformed.txt");
                
                //dataSet = Distorsion.getInstance().Distort(dataSet, 0.05);
                //dataSet = Distorsion.getInstance().DistortMLS2(dataSet, 0.05);
            } 
			   
            sparseDataSet = null;
            ci = null;
                
                numEigenSeries.setText( String.valueOf(dataSet.nominalLabels.size()) ); 
	    }
	    catch(Exception exc)
	    {
	    	exc.printStackTrace();
	    }
	    
	    // set the no of points text field from any of the trajectories
            noPointsTextField.setVisible(true);
            noPointsTextField.setText("No points: " + dataSet.numFeatures );
	    
	    // populate the list with the time series trajectory indexes
	    
	    DefaultListModel listModel = new DefaultListModel();
	    for(int i = 0; i < dataSet.instances.size(); i++)
	    {
	    	listModel.addElement(String.valueOf(i));
	    }
	    timeSeriesList.setModel(listModel);
	    
	    
	    
	    System.out.println("Loaded " + dataSet.instances.size() + " time series trajectories.");
	    
	    // in the end create the chart
	    CreateChart();
	    
	    manipulatedTimeSeries = null;
	    originalTimeSeries = null;
	    
	    // set the first element as selected
	    if(timeSeriesList.getModel().getSize() > 0)
	    {
	    	timeSeriesList.setSelectedIndex(0);
	    }
	    
	    
	}
	
	public void ChangeSelectedTrajectory()
	{
            try
            {
                int selectedTrajectoryIndex = Integer.parseInt( timeSeriesList.getSelectedValue().toString() );
                originalTimeSeries = dataSet.instances.get(selectedTrajectoryIndex);
                manipulatedTimeSeries = null;

                labelTextField.setVisible(true);
                labelTextField.setText("Label: " + dataSet.instances.get(selectedTrajectoryIndex).target);

                if( imputeCheckBox.isSelected() )
                {
                	GenerateManipulatedTimeSeries();
                }

                RedrawTimeSeriesChart();
            }
            catch( Exception exc )
            {
                exc.printStackTrace();
            }
	}
	
	public void CreateChart()
	{
		tsChart = new Chart2D();
		
		originalTimeSeriesTrace = new Trace2DSimple("Original");
		interpolatedTimeSeriesTrace = new Trace2DSimple("Interpolated");
		
		originalTimeSeriesTrace.setColor(Color.BLUE);
		interpolatedTimeSeriesTrace.setColor(Color.RED);
		
	    // Add the trace to the chart. This has to be done before adding points (deadlock prevention): 
	    tsChart.addTrace(interpolatedTimeSeriesTrace);
	    tsChart.addTrace(originalTimeSeriesTrace);
	    
	    chartPanel.add(tsChart);
	    
	    tsChart.setSize( chartPanel.getSize() );
	    
	    tsChart.getAxisX().setAxisTitle(new AxisTitle("Time"));
	    tsChart.getAxisX().setRange( new Range(-1, dataSet.numFeatures+1) );
	    tsChart.getAxisX().setRangePolicy( new RangePolicyFixedViewport(new Range(-1, dataSet.numFeatures+1)));
	    
	    
	    tsChart.getAxisY().setAxisTitle(new AxisTitle("Value")); 
	}
	
	// redraw the chart of the time series
	public void RedrawTimeSeriesChart()
	{
            originalTimeSeriesTrace.removeAllPoints();
            interpolatedTimeSeriesTrace.removeAllPoints();

            if( manipulatedTimeSeries != null)
            {
                for(int i = 0; i < manipulatedTimeSeries.features.size(); i++)
                {
                    FeaturePoint p = manipulatedTimeSeries.features.get(i);
                    if( p.status == PointStatus.PRESENT )
                    	interpolatedTimeSeriesTrace.addPoint(i, p.value); 
                }
            }

            if( originalTimeSeries != null )
            {
            	int numPoints = originalTimeSeries.features.size();
                for(int i = 0; i < numPoints; i++)
                {
                    FeaturePoint p = originalTimeSeries.features.get(i);
                    if( p.status == PointStatus.PRESENT && p.value != GlobalValues.MISSING_VALUE )
                    	originalTimeSeriesTrace.addPoint(i, p.value);
                }
                
                /*
	                // print the sax representation
	    			SymbolicRepresentation sr = new SymbolicRepresentation();
					int slidingWindowSize = 12;
					int innerDimension = 4;
					int alphabetSize = 4;
					
					DataInstance ins = originalTimeSeries;
					List<String> insSax = sr.ExtractBagOfPatterns(ins, slidingWindowSize, innerDimension, alphabetSize);
				*/
                
            }
	}
	
	// generate the manipulated time series
	public void GenerateManipulatedTimeSeries()
	{
            if(originalTimeSeries != null)
            {
                
                String imputationType = interpolationTechnique.getSelectedItem().toString();

                try 
                {
                    manipulatedTimeSeries = new DataInstance( originalTimeSeries );
                } 
                catch (Exception e) 
                {
                    e.printStackTrace();
                }

                double missingRatio = Double.parseDouble( missingRatioTextField.getText() );
                double gapRatio = Double.parseDouble( gapSizeRatioTextField.getText() );

                 
                if(imputationType.compareTo("collaborative") == 0) 
                {
                    if( ci == null)
                    {
                        sparseDataSet = new DataSet(dataSet);

                        MissingPointsGenerator mpg = new MissingPointsGenerator();
                        mpg.GenerateMissingPoints(sparseDataSet, missingRatio, gapRatio); 

                        ci = new CollaborativeImputation();
                        ci.Impute(sparseDataSet);
                    }

                    int selectedTrajectoryIndex = Integer.parseInt( timeSeriesList.getSelectedValue().toString() );
                    manipulatedTimeSeries = sparseDataSet.instances.get(selectedTrajectoryIndex);
                }
                else if(imputationType.compareTo("em") == 0) 
                {
                    if( em == null)
                    {
                        sparseDataSet = new DataSet(dataSet);

                        MissingPointsGenerator mpg = new MissingPointsGenerator();
                        mpg.GenerateMissingPoints(sparseDataSet, missingRatio, gapRatio);  

                        em = new ExpectationMaximization();
                        em.ImputeMissing(sparseDataSet);
                    }

                    int selectedTrajectoryIndex = Integer.parseInt( timeSeriesList.getSelectedValue().toString() );
                    manipulatedTimeSeries = sparseDataSet.instances.get(selectedTrajectoryIndex);
                }
                else if(imputationType.compareTo("linear") == 0)
                {
                    MissingPointsGenerator mpg = new MissingPointsGenerator();
                    mpg.avoidExtremes = true;
                    mpg.GenerateMissingPoints(manipulatedTimeSeries, missingRatio, gapRatio); 
                    
                    LinearInterpolation li = new LinearInterpolation(gapRatio);
                    li.Interpolate(manipulatedTimeSeries);
                }
                else if(imputationType.compareTo("cubicspline") == 0)
                {
                    MissingPointsGenerator mpg = new MissingPointsGenerator();
                    mpg.GenerateMissingPoints(manipulatedTimeSeries, missingRatio, gapRatio); 
                    
                    SplineInterpolation si = new SplineInterpolation(gapRatio);
                    si.CubicSplineInterpolation(manipulatedTimeSeries); 
                }
                else if(imputationType.compareTo("bspline") == 0)
                {
                    MissingPointsGenerator mpg = new MissingPointsGenerator();
                    mpg.GenerateMissingPoints(manipulatedTimeSeries, missingRatio, gapRatio); 
                    
                    SplineInterpolation si = new SplineInterpolation(gapRatio);
                    si.BSplineInterpolation(manipulatedTimeSeries); 
                }
                else if(imputationType.compareTo("mbi") == 0)
                {
                    if( mbi == null)
                    {
                        sparseDataSet = new DataSet(dataSet);

                        MissingPointsGenerator mpg = new MissingPointsGenerator();
                        mpg.GenerateMissingPoints(sparseDataSet, missingRatio, gapRatio);  

                        mbi = new ModelBasedImputation();
                        mbi.Impute(sparseDataSet);
                    }

                    int selectedTrajectoryIndex = Integer.parseInt( timeSeriesList.getSelectedValue().toString() );
                    manipulatedTimeSeries = sparseDataSet.instances.get(selectedTrajectoryIndex);

                }
                
                
                
                
                double distanceTrajectories = DTW.getInstance().CalculateDistance(originalTimeSeries, manipulatedTimeSeries);

                // round to 6 precision number
                try
                {
                    DecimalFormat formatDecimal = new DecimalFormat("#.######");
                    distanceTrajectories = Double.valueOf(formatDecimal.format(distanceTrajectories));
                }
                catch(Exception exc){}

                RedrawTimeSeriesChart();
            }
	}
        
        public void FindEigenSeries()
        {
            int numberEigenSeries = Integer.parseInt( numEigenSeries.getText() ); 
            
            MFSVM model = new MFSVM();
            model.enableTimeWarping = enableWarpingCheckBox.isSelected();
            model.learningRate = 0.0001;
            model.lambda = 0.001;
            model.alpha = 0.3;
            
            model.latentDimensions = numberEigenSeries;//(int)( (double)dataSet.numFeatures * 0.2);
            factorizationModel = model.FactorizeOnly(dataSet);
            
            
            tsChart = new Chart2D(); 
                
            Color [] colorArray = { Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW,
                Color.PINK, Color.ORANGE, Color.MAGENTA}; 
            

            currentEigenSeries = new Trace2DSimple("EigenSeries"); 
            currentEigenSeries.setColor(Color.blue); 
            
            tsChart.addTrace(currentEigenSeries); 

            NextEigenSeries();
  
	    chartPanel.add(tsChart);
	    
	    tsChart.setSize( chartPanel.getSize() );
	    
	    tsChart.getAxisX().setAxisTitle(new AxisTitle("Time"));
	    tsChart.getAxisY().setAxisTitle(new AxisTitle("Value"));
            
        }
        
        public void NextEigenSeries()
        {
            int numberEigenSeries = Integer.parseInt( numEigenSeries.getText() );
            
            if(currentEigenSeriesIndex < numberEigenSeries)
            {

                currentEigenSeries.removeAllPoints();

                for(int j = 0; j < dataSet.numFeatures; j++)
                {
                        double value = factorizationModel.getV().get(currentEigenSeriesIndex, j);
                        currentEigenSeries.addPoint(j, value);
                }

                if( currentEigenSeriesIndex < numberEigenSeries -1 )
                currentEigenSeriesIndex++;
            }
        }
        
        public void PreviousEigenSeries()
        {
            if(currentEigenSeriesIndex >= 0)
            {
                if(currentEigenSeriesIndex > 0)
                {
                    currentEigenSeriesIndex--;
                }
                 
               currentEigenSeries.removeAllPoints();

                for(int j = 0; j < dataSet.numFeatures; j++)
                {
                        double value = factorizationModel.getV().get(currentEigenSeriesIndex, j);
                        currentEigenSeries.addPoint(j, value);
                }

                
            }
        }
        /*
         * A main function
         */
        public static void main( String [] args )
        {
            TimeSeriesAnalysisGUI win = new TimeSeriesAnalysisGUI();
            win.frmTimeSeries.setVisible(true);
        }
	
}
