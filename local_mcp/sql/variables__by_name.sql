-- One dictionary entry. $1 = name.
SELECT * FROM variables WHERE name = $1::VARCHAR
