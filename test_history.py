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
# Math Calculation
from math import log
from math import sqrt
# Randomness
import random
# Plotting
import matplotlib.pyplot as plt
# Different Trading Strategies
import strategy_rlosrl
import strategy_rlos
import strategy_average
import strategy_winner
import strategy_loser
    

"""Use History Stock Data to test the performance of Trading Strategy"""
def Back_Test( Database_all, Mode):
    
    """*********************************Hyperparameter*********************************"""
    M = 15 + 1 # the number of assets under management (Note that the additional asset is currency)
    N = 20 # the length of history under consideration when making portfolio management decision(for Reinforcement Learning)
    Feature = 3 # how many features each tensor contains
    ro = 0 # Threshold for Similarity
    NN = 10 # the maximal length of history under consideration when making PM decision(for robust log-optimal strategy)
    L = 1000 # the duration(number of periods) of back test
    Learning_Percentage = 0.3 # The possibility we
    Initial_Investment = 1e6 # total money that we invest into the stock market initially
    Starting_Date_train = '2010-02-22' # the starting point of the investment (train mode)
    Starting_Date_test = '2010-02-22' # the starting point of the investment (test mode)
    Training_Episode = 100 # determines how many rounds we carry out the trading in training mode
    Training_Round = 5 # determines how many rounds we train the model after a long period of trading
    Testing_Episode = 3 # determines how many rounds we carry out the testing in testing mode
    #  Determine how many rounds we carry out the investment and when to start based on the mode 
    if Mode == 1:
        Epoch = Training_Episode # Training
        Starting_Date = Starting_Date_train
    else:
        Epoch = Testing_Episode # Testing
        Starting_Date = Starting_Date_test
  
    # Construct Trading agents
    agent_rlosrl = strategy_rlosrl.Agent( M, N, Feature)
    agent_rlos = strategy_rlos.Agent( M, NN, ro)
    agent_average = strategy_average.Agent()
    agent_winner = strategy_winner.Agent()
    agent_loser = strategy_loser.Agent()
    
    """Trading and Training"""
    for e in range(Epoch):
        # Sampling of Stocks
        stocks_num = len(Database_all) - 1
        index_all = [ i for i in range(stocks_num)]
        index_sample = random.sample( index_all, M-1)
        Database = []
        for index in index_sample:
            Database.append( Database_all[index] )
        Database.append( Database_all[-1] )
        
        # Convert Starting_date into Starting_index
        Starting_index = ( Database[-1].index(Starting_Date) )
        
        """*************************Preparation*************************"""
        # Initial investment
        p_start_t_rlosrl = Initial_Investment
        p_start_t_rlos = Initial_Investment
        p_start_t_average = Initial_Investment 
        p_start_t_winner = Initial_Investment
        p_start_t_loser = Initial_Investment
        # Portfolio Vector Memory(PVM)
        pvm_rlosrl = []
        pvm_rlos = []
        pvm_average = []
        pvm_winner = []
        pvm_loser = []
        # Portfolio Value Record(PVR)
        pvr_rlosrl = []
        pvr_rlos = []
        pvr_average = []
        pvr_winner = []
        pvr_loser = []
        # Training Material, each term in the train_material is of the form: (x_t, v_t, w_target, r_true)
        if Mode == 1:
            train_material = []
        else:
            pass
        
        """*************************Trading*************************"""
        for t in range(L): # Trade for consecutive L periods
            """First: we construct a tensor reflecting the history background of the t_th trading period and the price fluctuation vector of yesterday"""
            """Besides, we need to construct a stack of price fluctuation vectors reflecting 2NN days history"""
            # Building tensor
            V_High  = np.array( [np.ones(N)] )
            V_Low   = np.array( [np.ones(N)] )
            V_Close = np.array( [np.ones(N)] )
            for i in range(M-1):
                V_High = np.vstack( (V_High, Database[i][t+Starting_index-N:t+Starting_index,1]) ) 
                V_Low = np.vstack( (V_Low, Database[i][t+Starting_index-N:t+Starting_index,2]) )
                V_Close = np.vstack( (V_Close, Database[i][t+Starting_index-N:t+Starting_index,3]) )
            x_t = np.stack( ( V_High, V_Low, V_Close), axis = 2 ) # Note: the dimension of x_t is M*N*Feature
            x_t = x_t.reshape( 1, M, N, Feature) # for the input requirement of tensorflow 
            # Building price fluctuation vector of yesterday
            y_yesterday = [1]
            for i in range(M-1):
                y_yesterday.append( Database[i][Starting_index+t-1][3]/Database[i][Starting_index+t-1][0] )
            y_yesterday = np.array(y_yesterday)
            # Building a stack of price fluctuation vectors(2NN)
            history = []
            for forward in range(2*NN):
                y_history = [1]
                for i in range(M-1):
                    y_history.append( Database[i][Starting_index+t+forward-2*NN][3]/Database[i][Starting_index+t+forward-2*NN][0] )
                y_history = np.array(y_history)
                history.append(y_history)
            history = np.array(history)
            history = np.transpose(history)
                
            """Second: let different trading agents make portfolio changing decision based on the history background of the t_th trading period"""
            w_rlos = agent_rlos.predict(history)
            w_rlosrl, r_predict_rlosrl = agent_rlosrl.predict(x_t, w_rlos)
            w_average = agent_average.predict(y_yesterday) 
            w_winner = agent_winner.predict(y_yesterday) 
            w_loser = agent_loser.predict(y_yesterday) 
        
            """Third: The market commences! we need to know the change of assets and the immediate reward(r_t) after the t_th trading period"""
            # Calculate Price fluctuation Vector yt:= v_t ./ v_t-1
            y_t = [1]
            for i in range(M-1):
                y_t.append( Database[i][Starting_index+t][3]/Database[i][Starting_index+t][0] )
            y_t = np.array(y_t)
            # Calculate total money after the t_th trading period for each agent
            p_end_t_rlosrl = p_start_t_rlosrl*np.dot(y_t,w_rlosrl)
            p_end_t_rlos = p_start_t_rlos*np.dot(y_t,w_rlos)
            p_end_t_average = p_start_t_average*np.dot(y_t,w_average)
            p_end_t_winner = p_start_t_winner*np.dot(y_t,w_winner)
            p_end_t_loser = p_start_t_loser*np.dot(y_t,w_loser)

            
            """Forth: Write the trading information into memory for later use(training)"""
            # Record portfolio vector
            pvm_rlosrl.append(w_rlosrl)
            pvm_rlos.append(w_rlos)
            pvm_average.append(w_average)
            pvm_winner.append(w_winner)
            pvm_loser.append(w_loser)
            # Record portfolio value(total money)
            pvr_rlosrl.append( (p_start_t_rlosrl, p_end_t_rlosrl) )
            pvr_rlos.append( (p_start_t_rlos, p_end_t_rlos) )
            pvr_average.append( (p_start_t_average, p_end_t_average) )
            pvr_winner.append( (p_start_t_winner, p_end_t_winner) )
            pvr_loser.append( (p_start_t_loser, p_end_t_loser) )


            """Fifth: Preparing train_materia"""
            if Mode == 1:
                # Calculate the Optimal Portfolio Weight
                w_target = np.zeros(M)
                w_target[ np.argmax( y_t ) ] = 1
                # Calculate the Log rate of return
                r_t_rlosrl = log( p_end_t_rlosrl/p_start_t_rlosrl )
                train_material.append( (x_t, w_rlos, w_target, r_t_rlosrl) )
            else:
                pass
            
            
            """Sixth: update asset information for next trading period"""
            p_start_t_rlosrl = p_end_t_rlosrl
            p_start_t_rlos = p_end_t_rlos
            p_start_t_average = p_end_t_average
            p_start_t_winner = p_end_t_winner
            p_start_t_loser = p_end_t_loser
            
            """Print the current round"""
            print( "Epoch:" + str(e) +" || period:" + str(t) )
            
                     
        """*************************Training*************************"""
        if Mode == 1: # Training mode
            for epoch in range(Training_Round):
                print( "Training Round: " + str(epoch+1) )
                # Sample some training material
                train_material_sample = random.sample( train_material, int(Learning_Percentage*L) )
                # SGD Training
                agent_rlosrl.train( train_material_sample, M , N, Feature)
        else:# Test mode; Skip training process
            pass
            
        """*************************Performance Measure*************************"""
        # Seperate Evaluation
        _Evaluate( pvr_rlosrl, L,"RLOSRL", Starting_Date)
        _Evaluate( pvr_rlos, L,"RLOS", Starting_Date)
        _Evaluate( pvr_average, L, "Average", Starting_Date)
        _Evaluate( pvr_winner, L,"Winner", Starting_Date)
        _Evaluate( pvr_loser, L,"Loser", Starting_Date)
        # Combination Evaluation
        performance_all = [ (pvr_rlosrl, "RLOSRL"), (pvr_rlos, "RLOS"), (pvr_average, "Average"), (pvr_winner, "Winner"), (pvr_loser, "Loser") ]
        _Evaluate_all( performance_all, L, Starting_Date)

            
        
"""Calculation of Sharpe Ratio"""
def _Sharpe(pvr,L):
    ro = [ (record[1]/record[0])-1 for record in pvr]
    mean_ro = sum(ro) / L
    variance_ro = sum( [ (term-mean_ro)**2 for term in ro ] ) / L
    if variance_ro != 0:
        Sharpe_Ratio = mean_ro/sqrt(variance_ro)
    else:
        Sharpe_Ratio = np.inf
    return Sharpe_Ratio


"""Calculation of Maximum_Drawdown"""
def _MDD(pvr,L):
    p_start = [ record[0] for record in pvr ]
    DD = [ 1-min( p_start[t+1:] )/p_start[t] for t in range(L-1)]
    Maximum_Drawdown = max(DD)
    return Maximum_Drawdown        


"""Evaluate the performance of a strategy"""
def _Evaluate( pvr, L, name, Starting_Date):
    # Calculation of Sharpe Ratio
    Sharpe_Ratio = _Sharpe(pvr,L)
    # Calculation of Maximum Drawdown
    Maximum_Drawdown = _MDD(pvr,L)
    print("---------------------------------------------------------------")
    print("Note: This is the performance measure of "+ name +" strategy")
    print("Starting Date:")
    print(Starting_Date)
    print("Trading Length:")
    print(L)
    print("Sharpe Ratio:")
    print(Sharpe_Ratio)
    print("Maximum Drawdown:")
    print(Maximum_Drawdown)
    print("Initial Portfolio Value:")
    print(str(pvr[0][0]) + " RMB")
    print("Final Portfolio Value:")
    print(str(pvr[-1][-1]) + " RMB")
    print("Value plot:")
    plt.plot([record[0] for record in pvr])
    plt.show()
    print("Logarithmic rate of return plot:")
    plt.plot( [ log(record[1]/record[0]) for record in pvr] )
    plt.show()

def _Evaluate_all( performance_all, L, Starting_Date):
    print("---------------------------------------------------------------")
    print("Note: This is the performance comparison of all stategies")
    print("Starting Date:")
    print(Starting_Date)
    print("Trading Length:")
    print(L)
    figure = (plt.figure()).add_subplot(111)
    for strategy in performance_all:
        figure.plot( [record[0] for record in strategy[0]] , label = strategy[1] )
    plt.legend(loc = "lower left")
    plt.show()