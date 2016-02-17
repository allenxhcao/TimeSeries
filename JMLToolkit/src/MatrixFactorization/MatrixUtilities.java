package MatrixFactorization;

import Utilities.GlobalValues;
import DataStructures.Matrix;

/*
 * Some matrix utilities
 */
public class MatrixUtilities 
{
	public static double getRowByColumnProduct( Matrix U, int i, Matrix V, int j )
	{
		double val = GlobalValues.MISSING_VALUE;
		
		if( U.getDimColumns() == V.getDimRows() )
		{
			int K = U.getDimColumns();
			
			val = 0;
			
			for(int k = 0; k < K; k++)
			{
				val += U.get(i, k) * V.get(k, j);
			}
		}
		else
		{
			System.err.println( "Trying to multiply a row and a column of two matrixes having" +
									" different dimensions. U_col : " + U.getDimColumns() + 
									", V_row : " + V.getDimRows());
		}
		
		return val;
	}

       

}

