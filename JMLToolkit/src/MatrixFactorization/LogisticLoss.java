package MatrixFactorization;

import Utilities.Sigmoid;

public class LogisticLoss extends LossFunction {

	@Override
	public double Gradient_A(int i, int j, int k) {
		// TODO Auto-generated method stub
            double v = kernel.Value(i, j); 
            double sig = Sigmoid.Calculate(v);
            double y = C.get(i, j);
            
            return -( y - sig ) * kernel.Gradient_A(i, j, k);
	}

	@Override
	public double Gradient_B(int i, int j, int k) 
        {
            double v = kernel.Value(i, j);
            double sig = Sigmoid.Calculate(v);
            double y = C.get(i, j);
            
            return -( y - sig ) * kernel.Gradient_B(i, j, k);
            
	}

	@Override
	public double Loss(int i, int j) 
        {
            double v = kernel.Value(i, j);
            double sig = Sigmoid.Calculate(v);            
            double y = C.get(i, j);
            
            return -y*Math.log( sig ) - (1-y)*Math.log(1-sig); 
	}

}
