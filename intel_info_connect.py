import mysql.connector
import datetime
db = mysql.connector.connect(
    host="localhost",
    user="django",
    password="asdpoi",
    database="intel_models",
)
"""
    dev_id = models.CharField(max_length=30)
    date = models.DateTimeField(default='NONE')
    L1 = models.DecimalField(max_digits=5 , decimal_places=4)
    Tamb = models.DecimalField(max_digits=5 , decimal_places=4)
    Tliq = models.DecimalField(max_digits=5 , decimal_places=4)
    Tsuc = models.DecimalField(max_digits=5 , decimal_places=4)
    Psuc = models.DecimalField(max_digits=5 , decimal_places=4)
    Pliq = models.DecimalField(max_digits=5 , decimal_places=4)
    Tsh = models.DecimalField(max_digits=5 , decimal_places=4)
"""
#tags = ['lockdown']
cursor = db.cursor(buffered=True)

def create_table():
    cursor.execute(f"CREATE TABLE modelErrors (dev_id TEXT, date DATETIME, L1 DOUBLE, Tamb DOUBLE, Tsuc DOUBLE, Tliq DOUBLE, Psuc DOUBLE, Tsh DOUBLE, Pliq DOUBLE)")

def insert_model_info(dev_id, threshold_list):
    date_creation = datetime.datetime.now()
    #last_info_plot_L1 = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    command = "INSERT INTO modelInfo (dev_id,  date_creation, date_updated, threshold_L1, threshold_Tamb, threshold_Tliq, threshold_Tsuc, threshold_Psuc, threshold_Pliq, threshold_Tsh, last_info_plot_L1, last_info_plot_Tamb, last_info_plot_Tliq, last_info_plot_Tsuc, last_info_plot_Psuc, last_info_plot_Pliq, last_info_plot_Tsh) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    vals = (dev_id, date_creation, date_creation, threshold_list[0], threshold_list[1],threshold_list[2],threshold_list[3],threshold_list[4],threshold_list[5],threshold_list[6], 'static/graphGatherer/diel.png', 'static/graphGatherer/diel.png', 'static/graphGatherer/diel.png', 'static/graphGatherer/diel.png', 'static/graphGatherer/diel.png', 'static/graphGatherer/diel.png', 'static/graphGatherer/diel.png')

    cursor.execute(command, vals)
    db.commit()
    return

def insert_error_info(errorDict):
    #valores em %
    #L1, Tamb, Tliq, Tsuc, Psuc, Pliq, Tsh
    command = "INSERT INTO modelErrors (dev_id, date, L1, Tamb, Tliq, Tsuc, Psuc, Pliq, Tsh) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    vals = (errorDict['dev_id'][0],
            errorDict['date'],
            errorDict['L1'],
            errorDict['Tamb'],
            errorDict['Tliq'],
            errorDict['Tsuc'],
            errorDict['Psuc'],
            errorDict['Pliq'],
            errorDict['Tsh'],                        
    )
    print(vals)
    
    cursor.execute(command, vals)
    db.commit()
    print('success')
    return 

if __name__ == '__main__':
    create_table()
