***
# Robust-Log-Optimal-Strategy-with-Reinforcement-Learning

Contributions
====
- There are 4 contributors to this work: ***YiFeng Guo***; ***XingYu Fu***; ***YuYan Shi***; ***MingWen Liu***.<br>
- This Research is sponsored by ***Gradient Trading***.<br>
- For details of our research, you can visit: ***http://arxiv.org/abs/1805.00205*** <br>
- It is free to use the code for NON-COMMERCIAL purposes.<br>
- For further contact, you can send email to ***fuxy28@mail2.sysu.edu.cn***<br>

Sketch
====
We proposed a new Portfolio Management method termed as ***Robust Log-Optimal Strategy*** (***RLOS***), which ameliorates the ***General Log-Optimal Strategy*** (***GLOS***) by approximating the traditional objective function with quadratic Taylor expansion. It avoids GLOS’s complex CDF estimation process, hence resists the ”Butterfly Effect” caused by estimation error. Besides, RLOS retains GLOS’s profitability and the optimization problem involved in RLOS is computationally far more practical compared to GLOS. Further, we combine RLOS with ***Reinforcement Learning*** (***RL***) and propose the so-called ***Robust Log-Optimal Strategy with Reinforcement Learning*** (***RLOSRL***), where the RL agent receives the analyzed results from RLOS and observes the trading environment to make comprehensive investment decisions. The RLOSRL’s performance is compared to some traditional strategies on several back tests, where we randomly choose a selection of constituent stocks of the CSI300 index as assets under management and the test results validate its profitability and stability.<br>

Back Test Result
====
![image](https://github.com/fxy96/Robust-Log-Optimal-Strategy-with-Reinforcement-Learning/blob/master/1500.png)

Set up
====
Python Version:
------- 
- 3.5

Modules needed:
------- 
- numpy
- scipy
- matplotlib
- random
- math
- os
- sklearn
- tensorflow

Order of Running Code: 
------- 
- First, unzip **database_model.zip** .
- Second,run **run.py** to back test the PM strategy.                                               
