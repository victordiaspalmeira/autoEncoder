import boto3
import pandas
from numpy import mean, median

class dynamo_querier:
    dynamo = boto3.client('dynamodb')
    
    @classmethod
    def query_single_table(cls, table_name, **kwargs):
        paginator = cls.dynamo.get_paginator('query')

        result = []
        i = 1
        for page in paginator.paginate(TableName=table_name, **kwargs):
            result.append(page["Items"])
            i += 1
        #print('')
        return result

s
    @classmethod
    def query_all_proc(cls,kwargs):

        #dac_id = kwargs["ExpressionAttributeValues"][":dac_id"]["S"]
        #dac_id_version = dac_id[0:8]

        tables = []
        results = pandas.DataFrame()
        paginator = cls.dynamo.get_paginator("list_tables")
        for page in paginator.paginate():
            for table_name in page["TableNames"]:
                tables.append(table_name)

        for table in tables:
            if table in ['placeholder']:
                continue
            if table in ['DAC21019XXXX_RAW','DAC20719XXXX_RAW']:
                temp_result = cls.query_single_table(table, **kwargs[1])
                temp_result = flatten_list(temp_result)
                result = parse_result_list(temp_result)
            else:
                temp_result = cls.query_single_table(table, **kwargs[0])
                temp_result = flatten_list(temp_result)
                result = parse_result_array(temp_result)
                    
            if len(result) > 0:
                data = pandas.DataFrame.from_records(result, index="timestamp")
                results = results.append(data)
                
        return results  
        

def flatten_list(deep_list):
    flattened = []
    def flatten_list_aux(level):
        if isinstance(level, list):
            for next_level in level:
                flatten_list_aux(next_level)
        else:
            flattened.append(level)
    flatten_list_aux(deep_list)
    return flattened


def parse_result_list(in_list):
    
    d = {
                "N":float,
                "S":str,
                "BOOL":bool,
    }
    result_list = []
    for in_dict in in_list:
        result_dict = {} 
        for var, var_value in in_dict['telemetry']['M'].items():
            for data_type, data_value in var_value.items():
                if data_type != "NULL":
                    result_dict[var] = d[data_type](data_value)
        result_list.append(result_dict)
    return result_list


def parse_result_array(in_list):
    
    d = {
                "N":float,
                "S":str,
                "BOOL":bool,
                "L":list
    }
    result_list = []
    for in_dict in in_list:
        result_dict = {} 
        for var, var_value in in_dict.items():
            for data_type, data_value in var_value.items():
                if data_type != "NULL":
                    if data_type == "L": 
                        l = ([float(dic['N']) for dic in data_value])
                        if var == "L1":
                            result_dict[var] = median(l)
                        else:     
                            result_dict[var] = mean(l)
                    else:
                        result_dict[var] = d[data_type](data_value)

        result_list.append(result_dict)
    return result_list


def get_query_data(dev_id, start_time, end_time):

    return [{
        "ConsistentRead":False,
        #"ProjectionExpression":'dac_id, #tmstp, telemetry',
        #"FilterExpression":'#tmstp BETWEEN :start_time AND :end_time',
        "KeyConditionExpression":'dev_id =:dev_id AND #tmstp BETWEEN :start_time AND :end_time',
        "ExpressionAttributeNames": {
            '#tmstp': 'timestamp',
        },
        "ExpressionAttributeValues": {
            ":dev_id":{"S": dev_id},
            ":start_time":{"S": start_time},
            ":end_time":{"S": end_time},
        },
    },
    {
        "ConsistentRead":False,
        #"ProjectionExpression":'dac_id, #tmstp, telemetry',
        #"FilterExpression":'#tmstp BETWEEN :start_time AND :end_time',
        "KeyConditionExpression":'dac_id =:dac_id AND #tmstp BETWEEN :start_time AND :end_time',
        "ExpressionAttributeNames": {
            '#tmstp': 'timestamp',
        },
        "ExpressionAttributeValues": {
            ":dac_id":{"S": dev_id},
            ":start_time":{"S": start_time},
            ":end_time":{"S": end_time},
        },
    }]


def __test():
    import datetime

    #end_time = (datetime.datetime.now() - datetime.timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    end_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = (end_time - datetime.timedelta(hours=10)).isoformat()
    end_time = end_time.isoformat()
    
    r = ['DAC210191010']
    results = []
    for dac_id in r:
        query_data = get_query_data(dac_id, start_time, end_time)
        results.append(dynamo_querier.query_all_proc(query_data))

    print(results)


if __name__ == "__main__":
    __test()