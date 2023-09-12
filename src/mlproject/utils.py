import os
import sys
import pandas as pd
import pymysql
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.Connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established", mydb)
        df = pd.read_sql_query("select * from students", mydb)
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException(ex)