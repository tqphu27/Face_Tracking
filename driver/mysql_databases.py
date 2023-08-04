import mysql.connector.pooling
import logging
from enum import Enum
from typing import List, Tuple, Union

import json
import pandas as pd

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
        self.connect_pool_to_mysql(host, port, database, username, password)
        """ Virtually private constructor. """
        if DbConnectionSingleton.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DbConnectionSingleton.__instance = self

    def connect_pool_to_mysql(self, host, port, database, username, password):
        try:
            print('MySQL connection pool is establishing...')
            self.threaded_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=10,
                host=host,
                port=port,
                user=username,
                password=password,
                database=database
            )
        except Exception as e:
            print("Error while connecting to MySQL: ", e)

    def close_mysql_connection_pool(self, connection, cursor):
        try:
            cursor.close()
            self.threaded_pool.putconn(connection)
            self.threaded_pool.closeall()
            print("MySQL connection pool is closed")
        except Exception as e:
            print(e)

class ExecuteType(Enum):
    QUERY = 0
    ONE = 1
    MANY = 2
    
class MySQLException(Exception):
    pass

class MySQLConnection:
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
            conn = self.pool.get_connection()
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

            except mysql.connector.Error as e:
                LOGGER.exception(e)
                conn.rollback()
            except Exception as e:
                LOGGER.exception(e)
                raise e

        except Exception as e:
            LOGGER.exception(e)
            raise e
        finally:
            try:
                if conn.is_connected():
                    cursor.close()
                    conn.close()
            except:
                pass
            
    def get_all_col_name(self, tbl_name):
        sql_command = "SELECT column_name " \
                      "FROM information_schema.columns " \
                      "WHERE table_name   = '" + tbl_name + "'"
                      
        col_names = []
        records,_ = self._execute(command=sql_command, exe_type=ExecuteType.QUERY, data=None, max_time=1)
        for row in records:
            col_names.append(row[0])
        return col_names
    
    def get_all(self, tbl_name, condition=None, order_by=None):
        sql_command = "SELECT * from " + tbl_name
        if condition is not None:
            sql_command += (" WHERE " + condition)
        if order_by is not None:
            sql_command += (" ORDER BY " + order_by)

        return self.get_with_query(sql_command)

    def get_with_query(self, sql_command):
        records, desc = self._execute(sql_command, exe_type=ExecuteType.QUERY, data=None, max_time=1)

        col_names = []
        for elt in desc:
            col_names.append(elt[0])
        return pd.DataFrame(records, columns=col_names)

    def insert_many(self, tbl_name, list_records):
        col_names = "("
        values = "("
        for col in list_records.columns:
            col_names += (col + ",")
            values += ("%s,")

        col_names = col_names[:-1] + ")"
        values = values[:-1] + ")"

        sql_insert_query = "INSERT INTO " + tbl_name + " " + col_names + " VALUES " + values
        records_insert = [tuple(x) for x in list_records.values]
        row_count = self._execute(sql_insert_query, ExecuteType.MANY, records_insert, max_time=1)
        print(row_count, "Record inserted successfully into " + tbl_name)

    def insert(self, tbl_name: str, col_names: list, values: list):
        col_query = "("
        value_query = "("
        for col in col_names:
            col_query += (col + ",")
            value_query += "%s,"
        col_query = col_query[:-1] + ")"
        value_query = value_query[:-1] + ")"

        sql_insert_query = "INSERT INTO " + tbl_name + " " + col_query + " VALUES " + value_query
        row_count = self._execute(sql_insert_query, ExecuteType.ONE, values, max_time=1)
        # print(row_count, "Record inserted successfully into " + tbl_name)
        LOGGER.info(f"{row_count} Record inserted successfully into {tbl_name}")
        return row_count > 0

    def insert_dict(self, tbl_name: str, data: dict):
        values = []
        for x in data.values():
            if isinstance(x, dict) or isinstance(x, list):
                values.append(json.dumps(x, ensure_ascii=False))
            else:
                values.append(x)
        return self.insert(tbl_name, col_names=list(data.keys()), values=values)

    def update(self, tbl_name, col_name, value, col_condition, value_condition):
        sql = "UPDATE " + tbl_name + " SET " + col_name + " = %s WHERE " + col_condition + " = %s"
        row_count = self._execute(sql, exe_type=ExecuteType.ONE, data=(value, value_condition), max_time=1)
        return row_count > 0
                        

    
if __name__ == "__main__":
    
    mysql = MySQLConnection(user_name="root", password="tranphu123@", host="localhost", port="3306", database="es_db")
    # mysql._execute()