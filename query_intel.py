# -*- coding: utf-8 -*-
import datetime
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from CoolProp import CoolProp

import dynamo_querier
import mysql_querier

ewma = pd.Series.ewm




class Thermo:
    @classmethod
    def superheating(cls,Tsuc,Psuc,fluid):
        """Função para cálculo do superaquecimento
        Tsuc: Temperatura de sucção [°C]
        Psuc: Pressão de sucção [bar]
        fluid: nome do fluido de trabalho
        """
        fluid = cls.check_mixture(fluid)
        try:
            Te = CoolProp.PropsSI('T','P',Psuc*1e5+101325,'Q',1,fluid) # convertendo de bar para Pa
            Tsh = (Tsuc + 273.15) - Te # convertendo de °C para K
            return Tsh
        except ValueError as e:
            print(f"Não foi possível calcular Te com Psuc = {Psuc}!", file=sys.stderr)
            return None
        

    @classmethod
    def subcooling(cls,Tliq,Pliq,fluid):
        """Função para cálculo do subrresfriamento
        Tliq: Temperatura de líquido [°C]
        Pliq: Pressão de líquido [bar]
        fluid: nome do fluido de trabalho
        """
        fluid = cls.check_mixture(fluid)
        try:
            Tc = CoolProp.PropsSI('T','P',Pliq*1e5+101325,'Q',0,fluid) # convertendo de bar para Pa
            Tsc = Tc - (Tliq + 273.15) # convertendo de °C para K
            return Tsc
        except ValueError as e:
            print(f"Não foi possivel calcular Tc para Pliq = {Pliq}!", file=sys.stderr)
            return None


    @staticmethod
    def check_mixture(fluid):

        mixture = fluid
        # para acrescentar mais misturas, utilizar a estrutura abaixo
        if fluid == "R402B":
            CoolProp.set_config_bool(CoolProp.OVERWRITE_BINARY_INTERACTION, True)
            CoolProp.apply_simple_mixing_rule('R125', 'R22','linear')
            mixture = 'R22[0.62]&R125[0.38]'
        
        return mixture

        
class AWSdatabase():

    @classmethod
    def mariadb_select(cls,attr,table,fltr,fltr_id):
        """ Função que consulta o RDS para saber os dac id comissionados e seus fluidos"""
        import connection_data
        attr = ",".join(attr)

        try:
            q = mysql_querier.sql_querier(**connection_data.connection_data)
        except:
            raise ConnectionError("Error ao conectar no MySQL.")
        
        sql_command = "SELECT %s FROM %s WHERE %s = '%s';" % (attr,table,fltr,fltr_id)

        dac_info = q.read_from_query(sql_command)

        if len(dac_info) != 1:
            raise ValueError('Não existe este DAC ID')
        
        return dac_info[0]


    @classmethod
    def dynamodb_query(cls,dac_id : str, start_time : Union[str, datetime.datetime], end_time : Union[str, datetime.datetime]):
        """Realiza um query no DynamoDB na tabela RAW e retorna um dataframe."""


        if isinstance(start_time, str):
            cls.start_time = start_time
        else:
            cls.start_time = start_time.isoformat()
        
        if isinstance(end_time, str):
            cls.end_time = end_time
        else:
            cls.end_time = end_time.isoformat()

        query_data = dynamo_querier.get_query_data(dac_id, cls.start_time, cls.end_time)

        return dynamo_querier.dynamo_querier.query_all_proc(query_data).reset_index()



class Query(AWSdatabase):
    DAC=0
    DUT=1

    def __init__(self,dac_id,start_time,end_time):
        self.dev_type = Query.DAC if dac_id[0:3]=='DAC' else Query.DUT if dac_id[0:3]=='DUT' else None
        self.dac_id = dac_id
        self.start_time = start_time
        self.end_time = end_time

    def adc_to_bar(self,sensor,sensor_position,sensor_info):
        if self.dac_info['FLUID_TYPE'] is None:
            return None
        if sensor_position == 'Psuc':
            self.dataset['Psuc'] = float(sensor_info["MULT"])*self.raw_dataset[sensor].values + float(sensor_info["OFST"])
            Tsh = pd.Series(index=self.dataset.index, data=len(self.dataset)*[np.nan])
            if len(self.idx_on) != 0:
                tsh_series = Thermo.superheating(self.dataset.loc[self.idx_on,'Tsuc'].values,
                                                        self.dataset.loc[self.idx_on,'Psuc'].values,
                                                        str(self.dac_info['FLUID_TYPE']).upper()) # HERE!!!!!
                if tsh_series is not None:
                    Tsh.loc[self.idx_on] = tsh_series 
            self.dataset['Tsh'] = Tsh

        elif sensor_position == 'Pliq':
            self.dataset['Pliq'] = float(sensor_info["MULT"])*self.raw_dataset[sensor].values + float(sensor_info["OFST"])
                
            Tsc = pd.Series(index=self.dataset.index, data=len(self.dataset)*[np.nan])

            if len(self.idx_on) != 0:
                tsc_series = Thermo.subcooling(self.dataset.loc[self.idx_on,'Tliq'].values,
                                                        self.dataset.loc[self.idx_on,'Pliq'].values,
                                                        str(self.dac_info['FLUID_TYPE']).upper())
                if tsc_series is not None:
                    Tsc.loc[self.idx_on] = tsc_series
            self.dataset['Tsc'] = Tsc

    def fetch_dac_data(self):
        return AWSdatabase.mariadb_select(
                            ["FLUID_TYPE","P0_SENSOR","P0_POSITN","P1_SENSOR","P1_POSITN"],
                            "DEVACS",
                            "dac_id",
                            self.dac_id)

    def fetch_dataset(self):
        self.raw_dataset : pd.DataFrame = AWSdatabase.dynamodb_query(self.dac_id,self.start_time,self.end_time)
        if len(self.raw_dataset) == 0:
            raise Exception('Não há dados')  

        self.raw_dataset.timestamp = self.raw_dataset.timestamp.astype('datetime64[ns]')
        return self.raw_dataset

    def process_dataset(self):
        if self.dev_type == Query.DAC:
            self.dac_info = self.fetch_dac_data()
            self.fetch_dataset()
            self.clean_dac_dataset()
            dataset = self.steady_state()
        elif self.dev_type == Query.DUT:
            dataset = self.fetch_dataset()
            dataset.set_index("timestamp", inplace=True)
            if "Humidity" in dataset.columns:
                dataset = dataset[["Temperature", "Humidity"]]
            else:
                dataset = dataset[["Temperature"]]
        return dataset

    def clean_dac_dataset(self):
        self.dataset = pd.DataFrame()
        # transformando L1 de bool para int
        self.dataset['L1'] = self.raw_dataset['L1'].astype(int)
        
        
        self.dataset.loc[:, "Tamb"] = self.raw_dataset.T0.copy()
        self.dataset.loc[:,"Tamb"] = self.dataset.loc[:,"Tamb"].mask(np.isclose(self.dataset.loc[:, "Tamb"].to_numpy(), 85))
        self.dataset.loc[:,"Tamb"] = self.dataset.loc[:,"Tamb"].mask(np.isclose(self.dataset.loc[:, "Tamb"].to_numpy(), -99.9))
        self.dataset.loc[:, "Tsuc"] = self.raw_dataset.T1.copy()
        self.dataset.loc[:,"Tsuc"] = self.dataset.loc[:,"Tsuc"].mask(np.isclose(self.dataset.loc[:, "Tsuc"].to_numpy(), 85))
        self.dataset.loc[:,"Tsuc"] = self.dataset.loc[:,"Tsuc"].mask(np.isclose(self.dataset.loc[:, "Tsuc"].to_numpy(), -99.9))
        self.dataset.loc[:, "Tliq"] = self.raw_dataset.T2.copy()
        self.dataset.loc[:,"Tliq"] = self.dataset.loc[:,"Tliq"].mask(np.isclose(self.dataset.loc[:, "Tliq"].to_numpy(), 85))
        self.dataset.loc[:,"Tliq"] = self.dataset.loc[:,"Tliq"].mask(np.isclose(self.dataset.loc[:, "Tliq"].to_numpy(), -99.9))

        self.idx_on = self.dataset.loc[self.dataset['L1']==1].index.values
        
        if self.dac_info['P0_POSITN']:
            sensor_info = AWSdatabase.mariadb_select(["MULT","OFST"],"SENSORS","SENSOR_ID",self.dac_info['P0_SENSOR'])
            self.adc_to_bar('P0',self.dac_info['P0_POSITN'],sensor_info)
        if self.dac_info['P1_POSITN']:
            sensor_info = AWSdatabase.mariadb_select(["MULT","OFST"],"SENSORS","SENSOR_ID",self.dac_info['P1_SENSOR'])
            self.adc_to_bar('P1',self.dac_info['P1_POSITN'],sensor_info)

        # a mensagem dentro do telemetry talvez esteja com timestamp errado
        self.dataset['timestamp'] =  pd.to_datetime(self.raw_dataset['timestamp'])             
        self.dataset.drop(index = self.dataset.loc[self.dataset['timestamp'] < self.start_time].index,inplace = True)
        self.dataset.set_index('timestamp',inplace = True)

        # existe um erro nas mensagens e alguns timestamps estão duplicados
        self.dataset = self.dataset[~self.dataset.index.duplicated()]
        self.dataset = self.dataset.sort_index()

        return self.dataset


    def steady_state(self,window_size = 120):
        """ Aplica um filtro de média móvel exponencial e cria um nova coluna com os momento em regime permanente.
        """
        dataset_filtered = self.dataset

        for column in dataset_filtered:
            if column == 'L1':
                continue
            else:
                # EWMA filter
                # dropna() para ignorar os valores NaN de Tsh e Tsc
                dataset_filtered[column] = self.dataset[column].ewm(alpha = 0.01).mean()
    
        # steady state calculation
        max_std_dict = {"Tliq":0.2,
                        "Tsuc":0.2,
                        "Pliq":0.1,
                        "Psuc":0.1,
                        "Tsc":0.7,
                        "Tsh":0.7}

        #(df_std.index[0:window_size])
        df_std = dataset_filtered.drop(columns=['L1','Tamb']).rolling('240s').std() # 2T equivale a 2min
        df_ssd = np.all(df_std.apply(lambda x: x.replace(np.nan,0) < max_std_dict[x.name]) == True, axis = 1)
        dataset_filtered['SSD'] = df_ssd.astype(int)
        dataset_filtered['SSD'].replace(np.nan,0,inplace=True)

        # if self.dac_info['CLIENT_ID'] == 26:
        #     import tandem
        #     td = tandem.pred(dataset_filtered.loc[((dataset_filtered['L1'] == 1) & (dataset_filtered['SSD'] == 1)),['Psuc']].values)
        #     filter_idx = dataset_filtered.loc[((dataset_filtered['L1'] == 1) & (dataset_filtered['SSD'] == 1))].index
        #     dataset_filtered.at[filter_idx,'tandem'] = td
        
        return dataset_filtered


    def plot(self, dataset : pd.DataFrame, save_fig = False):

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as ticker

        colors = {
            'Temperature' : 'green', # DUT
            'Humidity' : 'red',
            'Tamb':'green',
            'Tliq':'red',
            'Tsuc':'navy',
            'Pliq':'orange',
            'Psuc':'deepskyblue',
            'Tsc':'springgreen',
            'Tsh':'darkred',
            'L1':'black',
            'SSD':'darkviolet',
            'tandem':'blue'
        }

        plot_labels = {
            'Temperature' : 'Temperatura ambiente [°C]', # DUT
            'Humidity' : 'Umidade ambiente relativa [%]',
            'Tamb':"Temperatura ambiente [°C]",
            'Tliq':'Temperatura de líquido [°C]',
            'Tsuc':'Temperatura de sucção [°C]',
            'Pliq':'Pressão de líquido [bar]',
            'Psuc':'Pressão de sucção [bar]',
            'Tsc':'Subresfriamento',
            'Tsh':'Superaquecimento',
            'L1':'Sinal de comando',
            'SSD':'Regime permanente',
            'tandem':'Compressor 2 ON'
        }

        dataset = dataset[~dataset.index.duplicated()]
        dataset = dataset.sort_index()
        #dataset = dataset.asfreq(freq='1s')
        ax = plt.gca()
        labels = []
        for column in dataset:
            dataset[column].plot(color=colors[column], style=',', label=plot_labels[column],ax=ax)
    
        ax.grid(which='both',axis='both')
        #ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,20)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
        plt.xticks(rotation=25)

        plt.gcf().autofmt_xdate()
        plt.legend(shadow=False,loc='upper right').set_draggable(True)

        plt.xlabel('Tempo')
        plt.title(f'Histórico {self.dac_id} - {dataset.index[0].strftime("%d/%m/%Y")}')
        
        if save_fig:
            plt.savefig('query.png')

        return ax


def active_sensors(dataset):
        column_len = dataset.count()
        active = []
        for sensor in column_len.index:
            if column_len[sensor] == 0 or sensor == 'SSD':
                continue
            else:
                active.append(sensor)
        return active


def buttons(ax):
    from matplotlib.widgets import CheckButtons

    lines = ax.get_lines()
    rax = plt.axes([0.65,0.02, .25, .15])
    labels = [str(line.get_label()) for line in lines]
    visibility = [line.get_visible() for line in lines]
    check = CheckButtons(rax, labels, visibility)

    def func(label):
        lines[labels.index(label)].set_visible(not lines[labels.index(label)].get_visible())
        ymax = []
        ymin = []
        for line in lines:
            if line.get_visible():
                ymax.append(np.nanmax(line.get_ydata())+2)
                ymin.append(np.nanmin(line.get_ydata())-2)

        ax.set_ylim(min(ymin),max(ymax))
        plt.draw()

    check.on_clicked(func)
    plt.show()


def query_manual(check_button=True):

    import argparse
    parser = argparse.ArgumentParser(description="Ferramenta para visualização de dados de um DAC.")
    parser.add_argument('-d',action='append', dest='ids', default=[], help='um dos DAC IDs desejados')
    parser.add_argument('-s', '--start-time',type=str, dest='start_time', help='Timestamp de início')
    parser.add_argument('-e', '--end-time' ,type=str, dest='end_time', help='Timestamp de fim')
    parser.add_argument('-l', '--last-hours', type=int, dest='last_hours', help='Use para fazer o query das últimas horas')
    parser.add_argument('--save', default=False, action="store_true",help='salvar os datasets')
    args = parser.parse_args()

    for dev_id in args.ids:
        dev_type = dev_id[0:3].upper()
        if args.start_time is not None and args.end_time is not None:
            intelQuery = Query(dev_id,args.start_time,args.end_time) 
        elif args.last_hours is not None:
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=args.last_hours)
            intelQuery = Query(dev_id, start_time.isoformat(), end_time.isoformat())
        dataset = intelQuery.process_dataset()
        if args.save:
            dataset.to_csv("%s.csv" % (dev_id))
        ax = intelQuery.plot(dataset)
        if check_button:
            buttons(ax)
        plt.show()

def query(dev_id = None , start_time = None, end_time = None, last_hours = None, save = False):
    print(dev_id)
    dev_type = dev_id[0:3].upper()
    if start_time is not None and end_time is not None:
        intelQuery = Query(dev_id, start_time, end_time)
        print(intelQuery)
    elif last_hours is not None:
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=last_hours)
        intelQuery = Query(dev_id, start_time.isoformat(), end_time.isoformat())
        print(intelQuery)

    dataset = intelQuery.process_dataset()
    print('------------------------------------------------------')
    print(dataset)
    if save:
        dataset.to_csv("%s.csv" % (dev_id))

    return dataset

if __name__ == "__main__":
    query_manual()
