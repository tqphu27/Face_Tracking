import json
import logging
from enum import Enum
from typing import List, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2 import pool

LOGGER = logging.getLogger("driver")


class DbConnectionSingleton:
    __instance = None

    @staticmethod
    def getInstance(host, port, database, username, password):
        """ Static access method. """
        if DbConnectionSingleton.__instance is None:
            DbConnectionSingleton(host, port, database, username, password)
        return DbConnectionSingleton.__instance

    def __init__(self, host, port, database, username, password):
        self.connect_pool_to_postgre(host, port, database, username, password)
        """ Virtually private constructor. """
        if DbConnectionSingleton.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DbConnectionSingleton.__instance = self

    def connect_pool_to_postgre(self, host, port, database, username, password):
        try:
            print('SQL connection pool is establishing...')
            self.threaded_pool = pool.ThreadedConnectionPool(1, 10,
                                                             user=username,
                                                             password=password,
                                                             host=host,
                                                             port=port,
                                                             database=database)
        except Exception as e:
            print("Error while connecting to PostgreSQL: ", e)

    def close_postgre_connection_pool(self, connection, cursor):
        try:
            cursor.close()
            self.threaded_pool.putconn(connection)
            self.threaded_pool.closeall()
            print("SQL connection pool is closed")
        except Exception as e:
            print(e)

class ExecuteType(Enum):
    QUERY = 0
    ONE = 1
    MANY = 2


class PostgresException(Exception):
    pass


class PostgresConnection:
    def __init__(self, user_name, password, host, port, database):
        self._user_name = user_name
        self._password = password
        self._host = host
        self._port = port
        self._database = database

        db_connection = DbConnectionSingleton.getInstance(host, port, database, user_name, password)
        self.pool = db_connection.threaded_pool
    
    def _execute(self, command: str, exe_type: ExecuteType = ExecuteType.QUERY, data: Union[List, Tuple] = None,
                 try_time: int = 0, max_time: int = 1):
        try:
            conn = self.pool.getconn()
            cursor = conn.cursor()

            try:
                if exe_type == ExecuteType.QUERY:
                    cursor.execute(command)
                    records = cursor.fetchall()
                    return records, cursor.description
                elif exe_type == ExecuteType.ONE:
                    cursor.execute(command, data)
                    conn.commit()
                    return cursor.rowcount
                elif exe_type == ExecuteType.MANY:
                    cursor.executemany(command, data)
                    conn.commit()
                    return cursor.rowcount

            except psycopg2.ProgrammingError as e:
                LOGGER.exception(e)
                conn.rollback()
            except psycopg2.OperationalError as e:
                LOGGER.exception(e)
                if try_time < max_time:
                    return self._execute(command, exe_type, data, try_time + 1, max_time)

        except Exception as e:
            LOGGER.exception(e)
            raise e
        finally:
            try:
                self.pool.putconn(conn)
            except:
                pass
            
    def create(self, tbl_name, data):
        # data_to_insert = [[0.17397864, -0.03958223, -0.02925141, -0.03081017, 0.12008465, 0.09588012]] 
        vector_data = "'{{ " + ','.join(str(value) for value in data[0]) + " }}'"
        sql_command = "INSERT INTO " + tbl_name + \
                      " (id, name, vector) VALUES " + \
                      "('5', 'Ronaldo', " + str(vector_data) +");" 
                      
        row_count = self._execute(command=sql_command, exe_type=ExecuteType.ONE, data=None, max_time=1)
        print(row_count)

    # def insert(self, tbl_name: str, col_names: list, values: list):
    #     col_query = "("
    #     value_query = "("
    #     for col in col_names:
    #         col_query += (col + ",")
    #         value_query += "%s,"
    #     col_query = col_query[:-1] + ")"
    #     value_query = value_query[:-1] + ")"

    #     sql_insert_query = "INSERT INTO " + tbl_name + " " + col_query + " VALUES " + value_query
    #     row_count = self._execute(sql_insert_query, ExecuteType.ONE, values, max_time=1)
    #     # print(row_count, "Record inserted successfully into " + tbl_name)
    #     LOGGER.info(f"{row_count} Record inserted successfully into {tbl_name}")
    #     return row_count > 0
               
if __name__ == "__main__":
    import sys
    import cv2
    sys.path.append("/home/tima/detec_and_tracking/Face-Mask-Detection")
    from run_face_mask import *
    img_path = "/home/tima/detec_and_tracking/Face-Mask-Detection/images/1.jpg"
    image = cv2.imread(img_path)

   
    face_mask = face_mask_end2end()
    
 
    f1, f2 = face_mask.extract_feature(image)

    mysql = PostgresConnection(user_name="postgres", password="tranphu123", host="localhost", port="5432", database="postgres-local")
    mysql.create("vector", f1)