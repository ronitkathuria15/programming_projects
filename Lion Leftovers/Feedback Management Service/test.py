import psycopg2
import pandas as pd
from credentials import PASSWORD, USER

connection = psycopg2.connect(
    host = 'db-cc.co4twflu4ebv.us-east-1.rds.amazonaws.com',
    port = 5432,
    user = USER,
    password = PASSWORD,
    database='lion_leftovers'
    )
cursor=connection.cursor()
connection.commit()

sql = """
SELECT * FROM Reviews; 
"""
print(pd.read_sql(sql, con=connection))