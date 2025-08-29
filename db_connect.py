# db_connect.py
import os
import pymysql
from dotenv import load_dotenv
from dbutils.pooled_db import PooledDB

# โหลดตัวแปรจาก .env
load_dotenv()

# สร้าง Connection Pool
pool = PooledDB(
    creator=pymysql,                  
    maxconnections=10,               
    mincached=2,                      
    maxcached=5,                     
    blocking=True,                    
    ping=1,                          
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", 3306)),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    database=os.getenv("DB_NAME"),
    charset="utf8mb4"
)

def get_connection():
    """ ดึง connection จาก pool (อย่าลืม conn.close() = คืนเข้าพูล) """
    return pool.connection()
