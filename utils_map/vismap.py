
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform
import seaborn as sns
from highcharts import Highchart
import pandas as pd
import numpy as np
import os


def plot_scatter(omicsMap, htmlpath = './', htmlname = None, radius = 3):

    title = '2D emmbedding of %s based on %s method' % (omicsMap.ftype, omicsMap.method)
    subtitle = 'number of %s: %s, metric method: %s' % (omicsMap.ftype, len(omicsMap.flist), omicsMap.metric)
    name = '5.%s_%s_%s_%s_%s' % (omicsMap.ftype,len(omicsMap.flist), omicsMap.metric, omicsMap.method, 'scatter')
    
    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)
    
    if htmlname:
        name = htmlname + '_' + name 
        
    filename = os.path.join(htmlpath, name)
        
    
    xy = omicsMap.embedded.embedding_
    colormaps = omicsMap.extract.colormaps
    
    df = pd.DataFrame(xy, columns = ['x', 'y'])
    
    bitsinfo = omicsMap.extract.bitsinfo.set_index('IDs')
    df = df.join(bitsinfo.reset_index())
    df['colors'] = df['Subtypes'].map(colormaps)
    df.loc[:,"y"] = -(df.loc[:,"y"].values)
    df.to_csv("./test.csv")
    




    H = Highchart(width=1000, height=900)
    H.set_options('chart', {'type': 'scatter', 'zoomType': 'xy'})    
    H.set_options('title', {'text': title})
    H.set_options('subtitle', {'text': subtitle})
    H.set_options('xAxis', {'title': {'enabled': True,'text': 'X', 'style':{'fontSize':20}},
                           'labels':{'style':{'fontSize':20}}, 
                           'gridLineWidth': 1,
                           'startOnTick': True,
                           'endOnTick': True,
                           'showLastLabel': True})
    
    H.set_options('yAxis', {'title': {'text': 'Y', 'style':{'fontSize':20}},
                            'labels':{'style':{'fontSize':20}}, 
                            'gridLineWidth': 1,})
    
#     H.set_options('legend', {'layout': 'horizontal','verticalAlign': 'top','align':'right','floating': False,
#                              'backgroundColor': "(Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'",
#                              'borderWidth': 1})
    
    
    H.set_options('legend', {'align': 'right', 'layout': 'vertical',
                             'margin': 1, 'verticalAlign': 'top', 'y':40,
                              'symbolHeight': 12, 'floating': False,})

    
    H.set_options('plotOptions', {'scatter': {'marker': {'radius': radius,
                                                         'states': {'hover': {'enabled': True,
                                                                              'lineColor': 'rgb(100,100,100)'}}},
                                              'states': {'hover': {'marker': {'enabled': False} }},
                                              'tooltip': {'headerFormat': '<b>{series.name}</b><br>',
                                                          'pointFormat': '{point.IDs}'}},
                                  'series': {'turboThreshold': 5000}})
    
    for subtype, color in colormaps.items():
        dfi = df[df['Subtypes'] == subtype]
        if len(dfi) == 0:
            continue
            
        data = dfi.to_dict('records')
        H.add_data_set(data, 'scatter', subtype, color=color)
    H.save_file(filename)
    print('   [MoInter] -- Info: save html file to %s' % filename)
    return df, H



def plot_grid(omicsMap, htmlpath = './', htmlname = None):

    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)    
    
    title = 'Assignment of %s by %s emmbedding result' % (omicsMap.ftype, omicsMap.method)
    subtitle = 'number of %s: %s, metric method: %s' % (omicsMap.ftype, len(omicsMap.flist), omicsMap.metric)    

    name = '5.%s_%s_%s_%s_%s' % (omicsMap.ftype,len(omicsMap.flist), omicsMap.metric, omicsMap.method, 'omicsMap')
    
    if htmlname:
        name = name = htmlname + '_' + name   
    
    filename = os.path.join(htmlpath, name)
    
    
    
    m,n = omicsMap.fmap_shape
    colormaps = omicsMap.extract.colormaps
    position = np.zeros(omicsMap.fmap_shape, dtype='O').reshape(m*n,)
    bitsinfo = omicsMap.extract.bitsinfo
    for i in range(len(bitsinfo)):
        bitsinfo.iloc[i,0] = bitsinfo.iloc[i,1] + '-' + bitsinfo.iloc[i,0]
    position[omicsMap._S.col_asses] = bitsinfo.iloc[:,0].values
    position = position.reshape(m, n)
    

    
    x = []
    for i in range(n):
        x.extend([i]*m)
        
    y = list(range(m))*n
        
        
    v = position.reshape(m*n, order = 'f')

    df = pd.DataFrame(list(zip(x,y, v)), columns = ['x', 'y', 'v'])
    subtypedict = bitsinfo.set_index('IDs')['Subtypes'].to_dict()
    subtypedict.update({0:'NaN'})
    df['Subtypes'] = df.v.map(subtypedict)
    df['colors'] = df['Subtypes'].map(colormaps) 

    
    H = Highchart(width=1000, height=900)
    H.set_options('chart', {'type': 'heatmap', 'zoomType': 'xy'})
    H.set_options('title', {'text': title})
    H.set_options('subtitle', {'text': subtitle})
    H.set_options('xAxis', {'title': None,                         
                            'min': 0, 'max': omicsMap.fmap_shape[1],
                            'startOnTick': False,
                            'endOnTick': False,    
                            'allowDecimals':False,
                            'labels':{'style':{'fontSize':20}}})

    
    H.set_options('yAxis', {'title': {'text': ' ', 'style':{'fontSize':20}}, 
                            'startOnTick': False,
                            'endOnTick': False,
                            'gridLineWidth': 0,
                            'reversed': True,
                            'min': 0, 'max': omicsMap.fmap_shape[0],
                            'allowDecimals':False,
                            'labels':{'style':{'fontSize':20}}})
    


    H.set_options('legend', {'align': 'right', 'layout': 'vertical',
                             'margin': 1, 'verticalAlign': 'top', 
                             'y': 60, 'symbolHeight': 12, 'floating': False,})

    
    H.set_options('tooltip', {'headerFormat': '<b>{series.name}</b><br>',
                              'pointFormat': '{point.v}'})

    
    H.set_options('plotOptions', {'series': {'turboThreshold': 5000}})
    
    for subtype, color in colormaps.items():
        dfi = df[df['Subtypes'] == subtype]
        if len(dfi) == 0:
            continue
        H.add_data_set(dfi.to_dict('records'), 'heatmap', 
                       name = subtype,
                       color = color,#dataLabels = {'enabled': True, 'color': '#000000'}
                      )
    H.save_file(filename)
    print('   [MoInter] -- Info: save html file to %s' % filename)
    
    return df, H




