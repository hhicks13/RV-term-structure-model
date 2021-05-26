SELECT DISTINCT symbol, asset, expiration, type
FROM active_options
WHERE asset <> 'ETH' AND strike_price > 9000 AND strike_price < 11000 AND type = 'call';
