package MatrixFactorization;

import java.util.ArrayList;
import java.util.List;

import TimeSeries.DTW;

// factorizations of type X = UU' meaning we use the baseline
// factorization X = UV', by enforcing U=Psi_i' 

public class SymmetricMF //extends BirelationalFactorization 
{
/*
	public SymmetricMF(int factorsDim) {
		super(factorsDim);
		// TODO Auto-generated constructor stub
	}
	
	@Override
    public void TrainReconstructionLoss(int i, int j)
    {
		//System.out.println("SymmetricMF::TrainReconstructionLoss");
		
		  double [] newUiRow = new double[latentDim];
          double [] newVjColumn = new double[latentDim];
          
          for(int k = 0; k < latentDim; k++)
          {
              double u_ik =  U.get(i, k), 
                      v_kj = Psi_i.get(k, j);

              double grad_u_ik = alpha * lossX.Gradient_A(i, j, k) + alpha*2*lambdaU*u_ik,
                  grad_v_kj = alpha * lossX.Gradient_B(i, j, k) + alpha*2*lambdaV* v_kj;

              double new_u_ik = u_ik - learningRate * grad_u_ik,
                      new_v_kj = v_kj - learningRate * grad_v_kj;

              newUiRow[k] = new_u_ik;
              newVjColumn[k] = new_v_kj;
          }
          
          // flush the changes
          for(int k = 0; k < latentDim; k++)
          {
        	  // set both updates on U
              U.set(i, k, newUiRow[k]);
              U.set(j, k, newVjColumn[k]);
              // copy the same updates on Psi_i
              Psi_i.set(k, i, newUiRow[k]);
              Psi_i.set(k, j, newVjColumn[k]);        
              
          }
    }
	*/
}
