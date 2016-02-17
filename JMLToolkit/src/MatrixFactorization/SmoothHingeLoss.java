package MatrixFactorization;
import Utilities.Sigmoid;

public class SmoothHingeLoss extends LossFunction {

	@Override
	public double Gradient_A(int i, int j, int k) 
        {
	    double v = kernel.Value(i, j), 
                   y = C.get(i, j);
            
            double z = y*v;
            
            double gradHinge = 0; 
            
            if(z <= 0) 
                gradHinge = -1;
            else if ( z > 0 && z < 1) 
                gradHinge = z-1;
            else if( z >= 1)
                gradHinge = 0;                
            
            
            return gradHinge * y * kernel.Gradient_A(i, j, k);
	}

	@Override
	public double Gradient_B(int i, int j, int k) 
        {
            double v = kernel.Value(i, j), 
                   y = C.get(i, j);
            
            double z = y*v;
            
            double gradHinge = 0;
            
            if(z <= 0) 
                gradHinge = -1;
            else if ( z > 0 && z < 1) 
                gradHinge = z-1;
            else if( z >= 1)
                gradHinge = 0;                
            
            
            return gradHinge * y * kernel.Gradient_B(i, j, k); 
            
	}

	@Override
	public double Loss(int i, int j) 
        {
            double v = kernel.Value(i, j), 
                   y = C.get(i, j);
            
            double z = y*v;
            double loss = 0;
            
            if(z <= 0) 
                loss = 0.5 - z;
            else if ( z > 0 && z < 1) 
                loss = 0.5 * (1-z)*(1-z);
            else if( z >= 1)
                loss = 0;                
            
            return loss;
	}

}
