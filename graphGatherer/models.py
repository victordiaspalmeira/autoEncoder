from django.db import models

# Create your models here.
class ModelInfo(models.Model):
    dev_id = models.CharField(max_length=30)
    date_creation = models.DateTimeField(default='NONE')
    date_updated = models.DateTimeField(default='NONE')
    threshold_L1 = models.DecimalField(max_digits=5 , decimal_places=4, default=0.0)
    threshold_Tamb = models.DecimalField(max_digits=5 , decimal_places=4, default=0.0)
    threshold_Tliq = models.DecimalField(max_digits=5 , decimal_places=4, default=0.0)
    threshold_Tsuc = models.DecimalField(max_digits=5 , decimal_places=4, default=0.0)
    threshold_Psuc = models.DecimalField(max_digits=5 , decimal_places=4, default=0.0)
    threshold_Pliq = models.DecimalField(max_digits=5 , decimal_places=4, default=0.0)
    threshold_Tsh = models.DecimalField(max_digits=5 , decimal_places=4, default=0.0)
    last_info_plot_L1 = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    last_info_plot_Tamb = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    last_info_plot_Tliq = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    last_info_plot_Tsuc = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    last_info_plot_Psuc = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    last_info_plot_Pliq = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    last_info_plot_Tsh = models.FileField(upload_to='static/graphGatherer', default='static/graphGatherer/diel.png')
    
    class Meta:
        db_table = 'modelInfo'

class ModelErrors(models.Model):
    dev_id = models.CharField(max_length=30)
    date = models.DateTimeField(default='NONE')
    L1 = models.DecimalField(max_digits=5 , decimal_places=4)
    Tamb = models.DecimalField(max_digits=5 , decimal_places=4)
    Tliq = models.DecimalField(max_digits=5 , decimal_places=4)
    Tsuc = models.DecimalField(max_digits=5 , decimal_places=4)
    Psuc = models.DecimalField(max_digits=5 , decimal_places=4)
    Pliq = models.DecimalField(max_digits=5 , decimal_places=4)
    Tsh = models.DecimalField(max_digits=5 , decimal_places=4)

    class Meta:
        db_table = 'modelErrors'