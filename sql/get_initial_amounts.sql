select * from a_o_btc 
where (symbol,timestamp) IN (select symbol,min(timestamp) from a_o_btc group by symbol) and strike_price > 8000 and strike_price < 9000 and type='call';
