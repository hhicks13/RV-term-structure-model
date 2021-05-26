from mibian import BS
import pandas as pd

class OptionPool():
    
    def __init__(self,_strike,_type,_exp,_init_Prem,_init_time,_init_spot,_init_iv):
        self.expiry = _exp
        self.strike = _strike
        self.otype = _type
        self.Prem_i = _init_Prem
        self.i = _init_time
        self.spot_i = _init_spot
        self.iv = _init_iv
        self.TotalA_i = 250
        self.TotalB_i = 250*_init_Prem
        self.DeamortizedA_i = 0
        self.DeamortizedB_i = 0
        self.user2BalanceA = {}
        self.user2BalanceB = {}
        self.user2fImp = {}
        self.lock = False

    def set_initialA(self,A_0):
        self.TotalA_i = A_0

    def set_initialB(self,B_0):
        self.TotalB_i = B_0
        
    def P_i(self):
        return self.Prem_i

    def get_current_pool_factor(self):
        return 0

    def get_user_pool_factor(self,user):
        return self.user2fImp[user]

    def get_user_balance_A(self,user):
        return self.user2BalanceA[user]

    def get_uder_balance_B(self,user):
        return self.user2BalanceB[user]

    def compute_mAA(self,user):
        Fvi = self.get_current_pool_Factor()
        d_b_A  = self.DeamortizedA_i
        total_A = self.TotalA_i 
        return min(Fvi*d_b_A,total_A)/d_b_A

    def compute_mBB(self,user):
        Fvi = self.get_current_pool_Factor()
        d_b_B  = self.DeamortizedB_i
        total_B = self.TotalB_i 
        return min(Fvi*d_b_B,total_B)/d_b_B

    def compute_mAB(self,user):
        return (self.TotalB_i - self.compute_mBB*self.DeamortizedB_i)/self.DeamortizedA_i

    def compute_mBA(self,user):
        return (self.TotalA_i - self.compute_mAA*self.DeamortizedA_i)/self.DeamortizedB_i

    #external
    def ADD_LIQUID(self,user,A,B):
        return 0
    #external
    def READD_LIQUID(self,user,A,B):
        return 0
    #external
    def REMOVE_LIQUID(self,user,A,B):
        return 0
    
    def update_deamortized_A(self,A,B):
        return 0
    def update_deamortized_B(self,A,B):
        return 0
    def update_user_balance_A(self,user,A):
        return 0
    def update_user_balance_B(self,user,B):
        return 0
    def update_total_A(self,user,A):
        return 0
    def update_total_B(self,user,B):
        return 0

    # TRADE FUNCTION ( does not update DB or Fvi)
    def update_time(self,_i):
        self.i = _i

    def update_spot_i(self,_i,_spot):
        self.spot_i = _spot
        self.i = _i
        
    def update_P_i(self,_i,_spot):
        self.update_spot_i(_i,_spot)
        ttm = self.compute_ttm(_i)
        params = [self.spot_i,self.strike,0,ttm]
        iv = self.iv
        C_unit = BS(params,volatility=iv)
        if self.otype == 'put':
            self.Prem_i = C_unit.putPrice
        elif self.otype == 'call':
            self.Prem_i = C_unit.callPrice
        else:
            raise ValueError

    # get PoolAmountA
    def get_x(self):
        P_i = self.Prem_i
        x = min(self.TotalA_i,self.TotalB_i/P_i)
        return x

    # get PoolAmountB
    def get_y(self):
        P_i = self.Prem_i
        y = min(self.TotalB_i,self.TotalB_i*P_i)
        return y
    
    def get_k(self):
        x = self.get_x()
        y = self.get_y()
        k = x*y
        return k

    # trade function
    def g(self,A):
        print("    g~Calculated Unit Price: ",self.Prem_i)
        x = self.get_x()
        y = self.get_y()
        print("    g~X:   ",x)
        print("    g~Y:   ",y)
        k = x*y
        B = k/(x - A) - y
        P_target = (y - B)/(x - A)
        print("    g~B: ",B)
        print("    g~P_target:",P_target)
        return B,P_target
    
    def update_totals(self,A,B):
        self.TotalA_i += A
        self.TotalB_i += B
        if self.TotalA_i <=0.001 or self.TotalB_i <= 0.001:self.lock = True

    # ONLY CALL THIS AFTER UPDATE_PI
    def update_next_iv(self,P_target):
        ttm = self.compute_ttm(self.i)
        params = [self.spot_i,self.strike,0,ttm]
        if self.otype == "put":
            C_target = BS(params,putPrice=P_target)
            new_iv = C_target.impliedVolatility
        elif self.otype == 'call':
            C_target = BS(params,callPrice=P_target)
            new_iv = C_target.impliedVolatility
        else:
            raise ValueError
        self.iv = new_iv
        return new_iv

    def compute_ttm(self,current_time):
        dt_exp = pd.to_datetime(self.expiry)
        dt_time = pd.to_datetime(current_time)
        dt_T = dt_exp - dt_time
        days_to_maturity = dt_T.total_seconds()/(24*60*60)
        return days_to_maturity
        
    def TRADE(self,A,direction,current_time,current_spot):
        current_time *=1e3
        print(pd.to_datetime(current_time),":-------------------------------------------------","(A,B) = (",self.TotalA_i,self.TotalB_i,")")
        if self.lock:
            print(" NO TRADE!!!!!!!! ")
            return False

        print("0. Consult Market Data")

        print("    ","time:        ",pd.to_datetime(current_time))
        print("    ","spot price:  ",current_spot)
        print("    ","strike:      ",self.strike)
        print("    ","option type: ",self.otype)
        print("    ","expiration:  ",pd.to_datetime(self.expiry))
        print("1. Calculate Factors")
        print("1.1 Calculate Option Unit Theoretical Price")
        ttm = self.compute_ttm(current_time)
        params = [self.spot_i,self.strike,0,ttm]
        iv = self.iv
        C_unit = BS(params,volatility=iv)
        if self.otype == 'put':
            if C_unit.putPrice <= 0.001:
                print("NO TRADE!!")
                return False
        elif self.otype == 'call':
            if C_unit.callPrice <= 0.001:
                print("NO TRADE!!")
                return False
        self.update_P_i(current_time,current_spot)
        print("    ","premium:   ",self.Prem_i)
        print("    ","iv:        ",self.iv)
        if self.i != current_time:raise ValueError
        print("    ","ttm:       ",self.compute_ttm(self.i))
        print("1.2 Calculate total price based on the transaction amount")
        if direction == 'buy': 
            print("    ","purchase options (options leaving pool)")
            B,P_target = self.g(A)
            self.update_totals(-1*A,B)
        elif direction == 'sell':
            print("    ","sell options (options entering pool)")
            B,P_target = self.g(A)
            self.update_totals(A,-1*B)
        
        print("    ","Transaction Price (in terms of B):",B)
        print("    ","Target Unit Price                :",P_target)

        print("2. Update Pool Balances")
        print("    "," Total Amount A",self.TotalA_i)
        print("    "," Total Amoubt B",self.TotalB_i)
        if self.lock: return False
        else:
            iv = self.update_next_iv(P_target)
            return True
        
        
        
    
        
        
        
        
