import pymysql
from typing import Optional, Dict, Union, Tuple, List
#from pprint import pprint as print

class sql_querier:
    def __init__(self, **kwargs):
        self.connection = pymysql.connect(**kwargs)

    def get_cursor(self):
        cursor = self.connection.cursor(cursor=pymysql.cursors.DictCursor)
        return cursor

    def close(self):
        self.connection.close()
    
    def connect(self):
        self.connection.connect()
        
    def read_from_query(self, sql:str, args:Union[Tuple, List, Dict] = ()):
        if sql.split(" ")[0] != "SELECT":
            raise ValueError("Somente operações de leitura são suportadas!")
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql, args)
                return cursor.fetchall()
        except Exception as e:
            print("query error:", e)


def __test():
    import connection_data
    q = sql_querier(**connection_data.connection_data)
    r = q.read_from_query("SELECT DAC_ID, DAC_APPL FROM DEVACS WHERE DAC_COMIS = 1 AND FLUID_TYPE = 'r22';")
    assert isinstance(r, list)
    assert isinstance(r[0], dict)
    #print(r)


if __name__ == "__main__":
    __test()