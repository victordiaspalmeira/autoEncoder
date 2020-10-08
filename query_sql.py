import pandas as pd
import mysql_querier
from typing import List

def mariadb_select(attr:List,table:str,fltr:str,fltr_id:str):
    """
    ### Consulta o MariaDB no api-server
    """

    import connection_data
    attr = ",".join(attr)

    try:
        q = mysql_querier.sql_querier(**connection_data.connection_data)
    except:
        raise ConnectionError("Error ao conectar no MySQL.")
    
    sql_command = "SELECT %s FROM %s WHERE %s = '%s';" % (attr,table,fltr,fltr_id)
    dac_info = q.read_from_query(sql_command)

    dac_info_df = pd.DataFrame(dac_info)

    return dac_info_df

if __name__ == "__main__":
    info = mariadb_select(attr=["DAC_ID"],
                          table="DEVACS",
                          fltr="DAC_APPL",
                          fltr_id="camara-fria")
    print(info)
