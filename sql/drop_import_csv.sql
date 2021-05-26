/*
DROP TABLE active_options;

DROP TABLE trades;
*/
DROP TABLE a_o_btc;
DROP TABLE a_o_eth;
DROP TABLE tr_btc;
DROP TABLE tr_eth;


/*active options import from csv

CREATE TABLE active_options (
    symbol varchar(50),
    timestamp BIGSERIAL,
    local_timestamp BIGSERIAL,
    type varchar(5),
    strike_price integer,
    expiration BIGSERIAL,
    mark_price float8, 
    mark_iv float8,
    underlying_price float8,
    CONSTRAINT "active_options_pkey"
    PRIMARY KEY (symbol,local_timestamp)
);

COPY active_options (symbol,timestamp,local_timestamp,type,strike_price,expiration,mark_price,mark_iv,underlying_price)
FROM PROGRAM 'cut -d "," -f 2,3,4,5,6,7,16,17,19 /Users/harrisonhicks/dreams/before/Deribit/deribit_options_chain_2020-03-01_OPTIONS.csv' WITH(FORMAT CSV, HEADER);
*/

/*trades that occured
*/
/*
CREATE TABLE trades (
    symbol varchar(50),
    timestamp BIGSERIAL,
    local_timestamp BIGSERIAL,
    id varchar(50),
    side varchar(5),
    price float8,
    amount float8,
    CONSTRAINT "trades_pkey"
    PRIMARY KEY (id)
);

COPY trades (symbol,timestamp,local_timestamp,id,side,price,amount)
FROM PROGRAM 'cut -d "," -f 2,3,4,5,6,7,8 /Users/harrisonhicks/dreams/before/Deribit/deribit_trades_2020-03-01_OPTIONS.csv' WITH(FORMAT CSV, HEADER);
*/

/*
add asset columns


ALTER TABLE active_options
ADD column asset varchar(5) ;

UPDATE active_options
SET asset = split_part(symbol,'-',1);


ALTER TABLE trades
ADD column asset varchar(5);

UPDATE trades
SET asset = split_part(symbol,'-',1);
*/




CREATE TABLE a_o_btc AS SELECT * FROM active_options WHERE asset='BTC';
CREATE TABLE a_o_eth AS SELECT * FROM active_options WHERE asset='ETH';


CREATE TABLE tr_btc AS SELECT * FROM trades WHERE asset='BTC';
CREATE TABLE tr_eth AS SELECT * FROM trades WHERE asset='ETH';


