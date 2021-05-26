import psycopg2
import sys
import datetime
from mibian import BS
import pandas as pd
sys.path.append(".")
from pool import OptionPool

def compute_target_price(row,R1,R2):
    print("hello world")

def black_scholes(row):
    otype = row[14]
    mark_iv = row[13]
    underlying_price = row[12]
    strike = row[9]
    
    _exp = row[10]*1e3
    _time = row[8]*1e3
    
    dt_exp = pd.to_datetime(_exp)
    dt_time = pd.to_datetime(_time)

    dt_T = dt_exp - dt_time
    days_to_maturity = dt_T.total_seconds()/(24*60*60)

    
    mark_price_oc = row[11]*underlying_price
    
    #calc premium from trade
    price_trade = row[5]
    amount_trade= row[6]
    mark_price_trade = price_trade*underlying_price
    params = [underlying_price,strike,0,days_to_maturity]
    if otype == "put":
        C_oc = BS(params,putPrice=mark_price_oc)
        sigma_oc = C_oc.impliedVolatility
        C_tr = BS(params,putPrice=mark_price_trade)
        sigma_tr = C_tr.impliedVolatility

        C_oc2 = BS(params,volatility=sigma_oc)
        Pbs_oc = C_oc2.putPrice
        C_tr2 = BS(params,volatility=sigma_tr)
        Pbs_tr = C_tr2.putPrice

        BSmarkiv = BS(params,volatility=mark_iv)
        miv2Pbs = BSmarkiv.putPrice
    elif otype == 'call':
        C_oc = BS(params,callPrice=mark_price_oc)
        sigma_oc = C_oc.impliedVolatility
        C_tr = BS(params,callPrice=mark_price_trade)
        sigma_tr = C_tr.impliedVolatility

        C_oc2 = BS(params,volatility=sigma_oc)
        Pbs_oc = C_oc2.callPrice
        C_tr2 = BS(params,volatility=sigma_tr)
        Pbs_tr = C_tr2.callPrice

        BSmarkiv = BS(params,volatility=mark_iv)
        miv2Pbs = BSmarkiv.callPrice
    print(row[0]==row[15]," obs2sigma oc - tr: ",sigma_oc-sigma_tr,", sigma2calc oc - tr: ",Pbs_oc-Pbs_tr,", (obs) Market Price - (obs) Trade Price: ",mark_price_oc - mark_price_trade)
    
    

def main():
    # Connect to an existing database
    connection = psycopg2.connect(user="postgres",host="127.0.0.1",port="5432",database="options")

    # Create a cursor to perform database operations
    cursor = connection.cursor()
    # Print PostgreSQL details
    print("PostgreSQL server information")
    print(connection.get_dsn_parameters(), "\n")

    # Initialize Pools
    pool_query = "select * from a_o_btc where (symbol,timestamp) IN (select symbol,min(timestamp) from a_o_btc group by symbol) and strike_price > 7000 and strike_price < 9000 and type='call';"
    cursor.execute(pool_query)
    rows = cursor.fetchall()
    sym2Pool = {}
    sym2count = {}
    sym2accept_count = {}
    print(f"{'symbol':<30}{'Beginning timestamp':^30}{'Beginning Premium':^30}{'spot-strike':^30}{'ITM':>10}")
    for row in rows:
        time = row[1]*1e3
        expiration = row[5]*1e3
        pool = OptionPool(row[4],row[3],expiration,row[8]*row[6],time,row[8],row[7])
        sym2Pool[row[0]] = pool
        sym2count[row[0]] = 0
        sym2accept_count[row[0]] = 0
        print(f"{row[0]:<30}{str(pd.to_datetime(time)):^30}{row[8]*row[6]:^30.6f}{row[8]-row[4]:^30.6f}   {str(row[4] < row[8]):>10}")
        print("Expiration:",pd.to_datetime(pool.expiry*1e3))
        print("strike:",pool.strike)
        print("type:",pool.otype)
        print("premium",pool.Prem_i)
        print("time: ",pool.i)
        print("spot:",pool.spot_i)
        print("iv: ",pool.iv)
        print("totalA:",pool.TotalA_i)
        print("totalB:",pool.TotalB_i)


    # Executing a SQL query
    print("joining tables")

    get_initial_vals=""
    

    join_btc1 = "SELECT tr_btc.*, a_o_btc.timestamp,a_o_btc.strike_price, a_o_btc.expiration, a_o_btc.mark_price, a_o_btc.underlying_price, a_o_btc.mark_iv, a_o_btc.type, a_o_btc.symbol, a_o_btc.local_timestamp "
    join_btc2 = "FROM tr_btc LEFT JOIN a_o_btc ON a_o_btc.timestamp = tr_btc.timestamp ORDER BY tr_btc.timestamp "
    print(join_btc1+join_btc2)
    #join_eth = "SELECT tr_eth.*, a_o_eth.timestamp,a_o_eth.strike_price,a_o_eth.expiration, a_o_eth.mark_price, a_o_eth.underlying_price, a_o_eth.mark_iv, a_o_eth.type, a_o_eth.symbol a_o_eth.local_timestamp
    #            FROM tr_eth LEFT JOIN a_o_eth ON a_o_eth.local_timestamp = tr_eth.local_timestamp ORDER BY tr_eth.timestamp"
    #print(join_eth)
    #
    print("btc trades with simultaneous options chain data")
    cursor.execute(join_btc1+join_btc2)
    rows = cursor.fetchall()
    #
    # on deribit each option corresponds to 1 ETH/BTC. 
    #
    #
    # Y HAT
    # calculate iv using bisection method and price using black scholes
    # underlying_price row[13]
    # strike row[10]
    # expiration[11]
    # timestamp[9]
    # NEED target_price from Pods protocol HERE!!!
    # for now, compare
    # mark_price and premium_usd 
    #
    #
    # calculate market price using LP reserves (pods protocol)
    #
    # Y
    # calculate premium (usd) using row[6] price
    #                              ,row[7] amount
    #                               row[13] underlying_price
    #
    # price/amount * underlying_price
    mismatch_bad_data = 0
    mismatch_pool = 0
    
    for row in rows:
        sym = row[0]
        if sym != row[15]:
            mismatch_bad_data+=1
            continue
        elif sym not in sym2Pool.keys():
            mismatch_pool+=1
            continue
        #elif sym == 'BTC-6MAR20-8500-C':
        else:
            sym2count[sym] +=1
            direction = row[4]
            pool = sym2Pool[sym]
            amount = row[6]
            print(row[0],"   ",direction," ",amount)
            current_time = row[1]
            current_spot = row[12]
            if pd.to_datetime(current_time*1e3).day == pd.to_datetime(pool.expiry).day and pd.to_datetime(current_time*1e3).month == pd.to_datetime(pool.expiry).month: continue
            print(direction)
            accept = pool.TRADE(amount,direction,current_time,current_spot)
            if accept:
                sym2accept_count[sym] +=1
        #else:
        #    continue

    print(" # rows ",len(rows))
    print("bad data:",mismatch_bad_data)
    print("bad init query:",mismatch_pool)
    print("# of trades per pool")
    print(f"{'symbol':<30}{'# of trades':^30}{'# of trades accepted':^10}{'reserves A:^10'}{'reserves B':>10}")
    for sym in sym2count:
        print(f"{sym :<30}{sym2count[sym]:^3d}{sym2accept_count[sym]:^3d}{sym2Pool[sym].TotalA_i:^20.6f}{sym2Pool[sym].TotalB_i:>20.6f}")
    cursor.close()
    connection.close()
    print("Postrgresql connection is closed"+"\n")


if __name__ == "__main__":
    main()
