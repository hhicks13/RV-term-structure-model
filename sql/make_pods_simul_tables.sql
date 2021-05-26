DROP TABLE pods_btc1;
DROP TABLE pods_eth1;

CREATE TABLE pods_btc1 AS (SELECT tr_btc.symbol,tr_btc.id,tr_btc.timestamp,tr_btc.side,tr_btc.amount,
                                  a_o_btc.timestamp,a_o_btc.strike_price, a_o_btc.mark_price, a_o_btc.underlying_price,a_o_btc.expiration FROM tr_btc LEFT JOIN a_o_btc ON a_o_btc.timestamp = tr_btc.timestamp);
				  
CREATE TABLE pods_eth1 AS (SELECT tr_eth.symbol,tr_eth.id,tr_eth.timestamp,tr_eth.side,tr_eth.amount,
                                  a_o_eth.timestamp,a_o_eth.strike_price, a_o_eth.mark_price, a_o_eth.underlying_price,a_o_eth.expiration FROM tr_eth LEFT JOIN a_o_eth ON a_o_eth.timestamp = tr_eth.timestamp);
