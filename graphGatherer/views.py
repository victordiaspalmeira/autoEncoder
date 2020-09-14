from django.shortcuts import render
from django.http import HttpResponse
from .models import ModelInfo, ModelErrors
def main_index(request):
    dev_list = ModelInfo.objects.order_by('dev_id')
    print(dev_list)
    context = {'dev_list': dev_list}
    return render(request, 'graphGatherer/index.html', context)

def thresholds(request, dev_id):
    query = ModelInfo.objects.filter(dev_id=dev_id)
    dev = query[0]
    context = {
        'dev_id': dev.dev_id,
        'threshold_L1': dev.threshold_L1,
        'threshold_Tamb': dev.threshold_Tamb,
        'threshold_Tliq': dev.threshold_Tliq,
        'threshold_Tsuc': dev.threshold_Tsuc,
        'threshold_Psuc': dev.threshold_Psuc,
        'threshold_Pliq': dev.threshold_Pliq,
        'threshold_Tsh': dev.threshold_Tsh
    }

    return render(request, 'graphGatherer/thresholds.html', context)

def errors(request, dev_id):
    dev_list = ModelErrors.objects.filter()
    print(dev_list)
    context = {
        'dev_list': dev_list,
    }

    return render(request, 'graphGatherer/errors.html', context)    

def display_graph(request, dev_id):
    query = ModelErrors.objects.filter(dev_id=dev_id)
    #dev = query[0]
    context = {
        'dev_id': dev_id,
    }
    
    return render(request, 'graphGatherer/graph.html', context)
