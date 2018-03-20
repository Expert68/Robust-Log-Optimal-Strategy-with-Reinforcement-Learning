"""
Robust Log-Optimal Strategy with Reinforcement Learning(RLOSRL)
Authors: XingYu Fu; YiFeng Guo; YuYan Shi; MingWen Liu;
Institution: Sun Yat_sen University
Contact: fuxy28@mail2.sysu.edu.cn
All Rights Reserved
"""


"""Import Modules"""
# Numerical Computation
import numpy as np
# math
from math import sqrt
from math import log
# Optimization
from scipy.optimize import minimize


class Agent:
    
    def __init__(self, M, NN, ro):
        # Number of assets under management
        self.M = M
        # Maximal length of history under consideration when making PM decision(for robust log-optimal strategy)
        self.NN = NN
        # Threshold for Similarity
        self.ro = ro
        
    
    """Decision Making"""
    def predict(self, history):
        # Record different Portfolio Vectors and their corresponding Weights
        b_set = []
        w_set = []
        # Consider different history length
        for n in range(2, self.NN+1):
            # Construct the background for the coming trading period
            target_back = history[:,-n:]
            # Select similar trading period
            selected = []
            for i in range(2*self.NN-n, 2*self.NN):
                i_back = history[:,i-n:i]
                r = self._pearson( target_back, i_back)
                if r > self.ro:
                    selected.append(i)
                else:
                    continue
                   
            if len( selected ) <= 1:
                continue # The number of samples is too small
            else: # Parameter Estimation && Optimization
                history_selected = history[:,selected]
                u_n = self._mean(history_selected) # Estimated Expectation
                sigma_n = self._covariance(history_selected, u_n) # Estimated Covariance Matrix
                b_opt_n = self._optimal(u_n, sigma_n) # Optimal Portfolio for history of length n
                w_n = self._weight( b_opt_n, history_selected) # Weight asssigned to b_opt_n, reflecting its portfitability
                b_set.append( b_opt_n )
                w_set.append( w_n )
        # Calculate final portfolio        
        if len( w_set ) == 0:
            return np.ones(self.M) / self.M
        else:
            s = sum(w_set)
            b_opt = np.zeros(self.M)
            for i in range( len( w_set ) ):
                b_opt += w_set[i]*b_set[i]
            b_opt /= s
            return b_opt
            

        
    """Calculation of Pearson Coefficient of two matrixes"""
    def _pearson(self, A, B):
        # Check shape
        if A.shape != B.shape:
            raise Exception("A and B doesn't have the same shape!")
        # calculate mean value of two matrix
        mean_A = A.mean()
        mean_B = B.mean()
        # calculate the pearson correlation coefficient
        part_1 = 0
        part_2 = 0
        part_3 = 0
        for m in range(A.shape[0]):
            for n in range(A.shape[1]):
                diff_A = A[m][n]-mean_A
                diff_B = B[m][n]-mean_B
                part_1 += diff_A * diff_B
                part_2 += diff_A**2
                part_3 += diff_B**2
        if part_2==0 or part_3==0:
            r=-100
        else:
            r = part_1/sqrt(part_2*part_3)
        return r


    """Estimate the expection of price fluctuation vector"""
    def _mean(self, history_selected):
        expectation = np.zeros( history_selected.shape[0] )
        for j in range(history_selected.shape[1]):
            expectation += history_selected[:,j]
        expectation /= history_selected.shape[1]
        return expectation
    
    
    """"Estimate the covariance of price fluctuation vector"""
    def _covariance(self, history_selected, u_n):
        covariance_matrix = np.zeros( ( history_selected.shape[0], history_selected.shape[0]) )
        for i in range(history_selected.shape[0]):
            for j in range(history_selected.shape[0]):
                for k in range( history_selected.shape[1] ):
                    covariance_matrix[i][j] += history_selected[i][k]*history_selected[j][k]
                covariance_matrix[i][j] /= history_selected.shape[1]
                covariance_matrix[i][j] -= u_n[i]*u_n[j]
        return covariance_matrix
        
    
    """Optimize the Objective Function"""
    def _optimal(self, u_n, sigma_n):
        
        """Allocation Utility(Objective Function)"""
        def objective(b, sign = -1):
            part0 = np.dot( u_n, b)
            part1 = log( part0 )
            part2 = 1/( 2*( part0 )**2 )
            part3 = 0
            for i in range( sigma_n.shape[0] ):
                for j in range( sigma_n.shape[1] ):
                    part3 +=  b[i]*sigma_n[i][j]*b[j]
            return sign * (part1 - part2*part3)
        
        """The gradient of objective function"""
        def _gradient(b, sign = -1):
            gradient = np.zeros( len(u_n) )
            part0 = np.dot( b, u_n)
            part1 = 0
            for i in range( sigma_n.shape[0] ):
                for j in range( sigma_n.shape[1] ):
                    part1 +=  b[i]*sigma_n[i][j]*b[j]
            for i in range( len(u_n) ):
                part3 = sum( [ b[j]*sigma_n[i][j] for j in range(len(u_n))] )
                gradient[i] = ( u_n[i]/part0 ) + (u_n[i]*part1)/(part0**3) - part3/(part0**2)
            return gradient*sign
            
        """Equality Constraint"""
        cons = [ { 'type':'eq', 'fun':lambda b: np.array([sum(b)-1]), 'jac':lambda b: np.ones(self.M)} ]
        
        """Inequality Constraint"""
        for i in range( self.M ):
            cons.append( { 'type':'ineq', 'fun':lambda b: np.array([ b[i] ]), 'jac':lambda b: np.array([ j==i for j in range(self.M)])} )
        
        cons = tuple(cons)
        bnd=tuple([ (0,1) for i in range(self.M)])
        
        res = minimize(objective, np.ones(self.M)/self.M, bounds=bnd, jac = _gradient, constraints=cons, method = "SLSQP")
        return res.x
    
    """Weight for b_opt_n"""
    def _weight(self, b_opt_n, history_selected):
        log_return = 1
        for j in range( history_selected.shape[1] ):
            log_return *= np.dot( b_opt_n, history_selected[:,j])
        log_return = log(log_return)
        return log_return