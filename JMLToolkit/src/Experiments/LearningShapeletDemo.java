package Experiments;

import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JMenuBar;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import java.awt.BorderLayout;
import javax.swing.JTextField;
import javax.swing.JLabel;

public class LearningShapeletDemo {

	private JFrame frmLearningShapelets;
	private JTextField KTextField;
	private JTextField LTextField;
	private JTextField etaTextField;
	private JTextField maxIterTextField;
	private JTextField lambdaWTextField;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					LearningShapeletDemo window = new LearningShapeletDemo();
					window.frmLearningShapelets.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public LearningShapeletDemo() {
		initialize();
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frmLearningShapelets = new JFrame();
		frmLearningShapelets.setTitle("Learning Shapelets - Demonstration");
		frmLearningShapelets.setBounds(100, 100, 983, 651);
		frmLearningShapelets.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		JMenuBar menuBar = new JMenuBar();
		frmLearningShapelets.setJMenuBar(menuBar);
		
		JMenu mnOpen = new JMenu("Open");
		menuBar.add(mnOpen);
		
		JMenuItem mntmOpenDataset = new JMenuItem("Open Dataset");
		mnOpen.add(mntmOpenDataset);
		
		JMenu mnRun = new JMenu("Run");
		menuBar.add(mnRun);
		
		JMenuItem mntmRunToCompletion = new JMenuItem("Run To Completion");
		mnRun.add(mntmRunToCompletion);
		
		JMenuItem mntmRunForOne = new JMenuItem("Run For One Iteration");
		mnRun.add(mntmRunForOne);
		
		JPanel panel = new JPanel();
		frmLearningShapelets.getContentPane().add(panel, BorderLayout.NORTH);
		
		JLabel lblK = new JLabel("K");
		panel.add(lblK);
		
		KTextField = new JTextField();
		KTextField.setText("0.15");
		panel.add(KTextField);
		KTextField.setColumns(10);
		
		JLabel lblL = new JLabel("L");
		panel.add(lblL);
		
		LTextField = new JTextField();
		LTextField.setText("0.125");
		panel.add(LTextField);
		LTextField.setColumns(10);
		
		JLabel lblEta = new JLabel("eta");
		panel.add(lblEta);
		
		etaTextField = new JTextField();
		etaTextField.setText("0.01");
		panel.add(etaTextField);
		etaTextField.setColumns(10);
		
		JLabel lblMaxiter = new JLabel("maxIter");
		panel.add(lblMaxiter);
		
		maxIterTextField = new JTextField();
		maxIterTextField.setText("2000");
		panel.add(maxIterTextField);
		maxIterTextField.setColumns(10);
		
		JLabel lblLambdaw = new JLabel("lambdaW");
		panel.add(lblLambdaw);
		
		lambdaWTextField = new JTextField();
		panel.add(lambdaWTextField);
		lambdaWTextField.setColumns(10);
	}

}
