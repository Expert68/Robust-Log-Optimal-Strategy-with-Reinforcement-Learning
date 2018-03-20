"""
Robust Log-Optimal Strategy with Reinforcement Learning(RLOSRL)
Authors: XingYu Fu; YiFeng Guo; YuYan Shi; MingWen Liu;
Institution: Sun Yat_sen University
Contact: fuxy28@mail2.sysu.edu.cn
All Rights Reserved
"""


if __name__ == "__main__":
    # Step0: Import Necessary Packages
    import load_data 
    import test_history
    
    # Step1: Load Stock Data from Disk
    print("Loading data from disk.")
    Database_all = load_data.Load_Stock_Data()
    
    # Step2: Let user specify the mode(We have 2 available modes)
    Mode = int( input("Choose your mode(1 for Training; 2 for Testing;):") )
    
    # Step3: Run the Back Test
    test_history.Back_Test( Database_all, Mode)
else:
    raise Exception("Error: You can not quote run.py from outside file!")