"""
Author>>>Arman_Najari
Date>>>2019/09/10
Email>>>armannd092@gmial.com
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import geojson


def unique_rows(a):
    """
    :param a: is the arrays of data
    :return: the unique item in the array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def point_inside_polygon(x, y, poly):
    """
    check if the point is inside the polygon or not
    :param x: x value of the point
    :param y: y value of the point
    :param poly: array of the points
    :return: boolean
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    xinters = 0
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class dataMG:
    """
    dataMG is a class for preparing data of the bike sharing for ML algorithm
    """

    def __init__(self, fp):
        self.fp = fp
        global inpath, filepath
        inpath = os.getcwd()
        filepath = os.path.join(inpath, fp)

    def read(self):
        # read the csv file
        data = pd.read_csv(filepath, infer_datetime_format=True)
        return data

    def clean(data):
        """
        cleaning data from invalid input
        :return:
        """
        return data[data['birth year'].values > 1940]

    def read2learn(self, frac):
        """
            Fp is the data file path
            frac is the percentage of data that want to be use in training
        """
        out = []
        data = self.read()
        # reduce the data
        redData = data.sample(frac=frac, random_state=99)
        # reduce the size of the data
        data_prd_trn = redData.sample(frac=0.8, random_state=1354)
        data_prd_tst = redData.drop(data_prd_trn.index)
        out.append(data_prd_trn)
        out.append(data_prd_tst)
        # print(data_prd_trn.iloc[0])
        return out

    def time2seconds(self, cln):
        """
        transfer the time data of a column to the second
        :param cln: the key to the column
        :return: the pars to second
        """
        data = self.read()
        return (pd.to_datetime(data[cln])).dt.hour * 3600 + (pd.to_datetime(data[cln])).dt.minute * 60 + (
            pd.to_datetime(data[cln])).dt.second

    def date_gen(self):
        """
        Generating the date as variable from the file name
        :return: date string
        """
        fn = self.fp
        fn = fn.split('.')[0]
        fn = fn.split('/')[1]
        fn = fn.split('-')[0]
        year = fn[:4]
        month = fn[4:6]
        date = year + '-' + month + '-' + '01' + ' ' + '00:00:00'
        return date

    def exp_name(self, freq='30D', n=''):
        """
            generate a filename for saving the csv file
        :param freq: Frequency of the data
        :param n: number of file
        :return: File name string
        """
        fp = self.fp
        date = pd.date_range(self.date_gen(), periods=2, freq=freq)
        filename = (str(date[0]) + '-' + str(date[1]) + '-' + fp.split('-')[1:2][0] + '-' + n + fp.split('-')[2:3][
            0]).replace(
            ':', '_')
        return filename

    def add_hour(data):
        """
        Add a column to the data that extract hour from date
        :return: data frame
        """
        data['hours'] = (pd.to_datetime(data['starttime'])).dt.hour
        return data

    def add_dayofweek(data):
        """
        Add a column to the data that extract day of the week from date
        :return: data frame
        """
        data['dayofweek'] = (pd.to_datetime(data['starttime'])).dt.dayofweek
        return data
    def add_dayofmonth(data):
        """
        Add a column to the data that extract day of the week from date
        :return: data frame
        """
        data['dayofmonth'] = (pd.to_datetime(data['starttime'])).dt.day
        return data
    def sampler(data, frac=0.3):
        """
        select a random sample from the data set
        :param frac: the amount of the selected data between 0-1
        :return: data frame
        """
        return data.sample(frac=frac, random_state=45678)

    def data_inPeriod(self, data, frac, freq):
        """
        select a specific period of time from the data :param data: dataset :param frac: the amount of the selected
        data between 0-1 :param freq: frequency that data needed to be divided:(check
        :https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)
        :return:data frame
        """
        # make the starting time of the trip as index of data set
        data = data.set_index(pd.DatetimeIndex(data['starttime']))
        # adding the week days to the table
        # data['dayofweek']=(pd.to_datetime(data['starttime'])).dt.dayofweek
        data = dataMG.add_hour(data)
        # take 30% of the data for experimentation
        data_sample = data.sample(frac=frac, random_state=99)
        # generate the date that is going to split the data
        date = pd.date_range(self.date_gen(), periods=2, freq=freq)
        # split the data with the first period of time that had generated
        data_out = data_sample[str(date[0]):str(date[1])]
        return data_out

    def exportCSV(data, filename):
        """
        exporting the dataframe on a csv file in the current directory
        :param filename: the name of the file
        :return: None
        """
        # build the path for openning file
        expt = os.path.join(inpath, filename)
        # open the file to write on
        with open(expt, 'w', newline='') as tp:
            # write the data on csv
            data.to_csv(tp, index=False)
            tp.close()
            pass
        return print("done!")

    def dataToanalysis(data):
        """
        print a set of analysis about the tripe times
        :return: None
        """
        print(data.loc[:, 'hours'].var(), " ", data.loc[:, 'dayofweek'].var(), " ", data.loc[:, 'hours'].mean(), " ",
              data.loc[:, 'dayofweek'].mean(), " ", data.loc[:, 'hours'].median(), " ",
              data.loc[:, 'dayofweek'].median())

    def segmentDuration(self, key):
        """
        cluster the trip duration in 3 category of low, mid, high
        :param key: the label of the data that indicate duration
        :return: data frame
        """
        data = self.read()
        l0 = []
        l1 = []
        l2 = []
        for i in data[key]:
            if i < (5 * 60):
                l0.append(1)
                l1.append(0)
                l2.append(0)
                pass
            elif i < (8 * 60) and i >= (5 * 60):
                l0.append(0)
                l1.append(1)
                l2.append(0)
                pass
            else:
                l0.append(0)
                l1.append(0)
                l2.append(1)
                pass
        data['duration_low'] = pd.DataFrame(l0)
        data['duration_mid'] = pd.DataFrame(l1)
        data['duration_high'] = pd.DataFrame(l2)
        data.drop(columns=key, inplace=True)
        return data

    def unique_stations(data):
        """
        make a data with unique station from the trip data
        :return: data frame
        """
        startSt = data[['start station id', 'start station latitude', 'start station longitude']].copy()
        endSt = data[['end station id', 'end station latitude', 'end station longitude']].copy()
        ss_np = pd.DataFrame.to_numpy(startSt)
        es_np = pd.DataFrame.to_numpy(endSt)
        alls = np.concatenate((ss_np, es_np), axis=0)
        sts = unique_rows(alls)
        sts = pd.DataFrame(sts)
        pd.DataFrame.dropna(sts, inplace=True)
        return sts

    def kminClusterSt(data, k):
        """
        clustring the station with k-means method and add the result for the start and end trip station to the data
        :param k: number of clusters
        :return: data frame
        """
        sts = dataMG.unique_stations(data)
        sts = pd.DataFrame.drop(sts, axis=1, columns=[0])
        kmeans = KMeans(n_clusters=k, random_state=0, max_iter=200)
        ss_np = pd.DataFrame.to_numpy(data[['start station latitude', 'start station longitude']])
        es_np = pd.DataFrame.to_numpy(data[['end station latitude', 'end station longitude']])
        sts_np = pd.DataFrame.to_numpy(sts)
        clr = kmeans.fit(sts_np)
        data['start_st_cluster'] = kmeans.predict(ss_np)
        data['end_st_cluster'] = kmeans.predict(es_np)
        sts['cls'] = kmeans.predict(sts_np)
        # plt.scatter(x=sts[:,0], y=sts[:,1], data=endSt,c='blue')
        # plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='pink')
        # plt.show()
        return data

    def toBinary(data, key, nods):
        """
        explode the value of the given labels to its binary representative; adding each diget in seprate column to data
        :param key: labels to convert
        :param nods: list of nodes for each labels
        :return:
        """
        def split(str):
            return [char for char in str]

        def toInt(l_str):
            return [int(i) for i in l_str]

        def clmname(key, n):
            ks = []
            for i in range(n):
                nm = key + str(i + 1)
                ks.append(nm)
                pass
            return ks

        dt = data
        for k in key:
            for n in nods:
                data_key = dt[k].copy()
                l = []
                nms = clmname(k, n)
                for i in data_key:
                    if i != None:
                        i = int(i)
                    else:
                        i = -1
                    b = format(i, '08b')
                    sb = str(b)
                    sbl = split(sb)[3:]
                    ibl = toInt(sbl)
                    # print(ibl)
                    l.append(ibl)
                pass
            pass
            array = np.array(l)
            d = pd.DataFrame({nms[0]: array[:, 0],
                              nms[1]: array[:, 1],
                              nms[2]: array[:, 2],
                              nms[3]: array[:, 3],
                              nms[4]: array[:, 4]})
            dt = pd.DataFrame.join(dt, d)
        pass
        return dt

    def stationCLs_join(data, st_cls):
        '''
            joining the data with the clustered station based on station id
        :param st_cls: clustered station file path
        :return:
        '''
        cls = pd.read_csv(st_cls)
        cls = cls[['id', 'id_2']].copy()
        data_start = data.join(cls.set_index('id'), on='start station id')
        data_start.rename(columns={'id_2': 'start_st_cluster'}, inplace=True)
        data_start_end = data_start.join(cls.set_index('id'), on='end station id', lsuffix='_end_cls')
        data_start_end.rename(columns={'id_2': 'end_st_cluster'}, inplace=True)
        data_start_end['start_st_cluster'] = pd.to_numeric(data_start_end['start_st_cluster'], downcast='integer')
        data_start_end.dropna(inplace=True)
        return data_start_end

    def stationCls(data, st_cls):
        '''
           Looping in the data and add the station cluster number based on location of the station
        :param st_cls: clustered station data
        :return:
        '''
        ss = data[['start station latitude', 'start station longitude']]
        se = data[['end station latitude', 'end station longitude']]
        clss = []
        for i, j in ss.iloc[:].values:
            cls = st_cls.loc[i.item(), j.item()].values
            clss.append(cls)

        data['start_st_cluster'] = pd.DataFrame(clss)
        clss.clear()
        for i, j in se.iloc[:].values:
            cls = st_cls.loc[i.item(), j.item()].values
            clss.append(cls)
        data['end_st_cluster'] = pd.DataFrame(clss)
        return data

    def stationCls_id(data, st_cls):

        '''
            Looping in the data and add the station cluster number based on id of the station
        :param st_cls: clustered station data
        :return:
        '''
        ss = data[['start station id']]
        se = data[['end station id']]
        clss = []
        for i in ss.iloc[:].values:
            if str(i.item()) == 'nan':
                cls = -1
                pass
            else:
                cls = st_cls.loc[int(i.item())]['id_2']
                pass
            clss.append(int(cls))
        data['start_st_cluster'] = pd.DataFrame(clss)
        clss.clear()
        for i in se.iloc[:].values:
            if str(i.item()) == 'nan':
                cls = -1
                pass
            else:
                cls = st_cls.loc[int(i.item())]['id_2']
                pass
            clss.append(int(cls))
        data['end_st_cluster'] = pd.DataFrame(clss)
        return data

    def boundClustring(data, boundrys):
        """
        this is for the raw data of the neighborhood!
        :param boundrys:
        :return:
        """
        sts = dataMG.unique_stations(data)
        b = dataMG(boundrys)
        bo = b.read()['the_geom']
        bos = []
        clusterNM = []

        def clean(pts):
            for p in pts:
                if len(p) > 2:
                    del p[0]
            return pts

        def toInt(pts):
            newpts = []
            for p in pts:
                nm = []
                for j in p:
                    for c in j:
                        if c == ')' or c == '(':
                            j = j.replace(c, '')
                    nm.append(float(j))
                newpts.append(nm)
            return newpts
            # print(st_cls.loc[40.863000,-73.905000].values)

        for i in bo:
            st = i[16:-3]
            pt = st.split(',')
            pts = [p.split(' ') for p in pt]
            pts = clean(pts)
            pts = toInt(pts)
            bos.append(pts)
        n = len(bos)
        for i in range(n):
            bound = bos[i]
            for s in sts:
                if point_inside_polygon(s[1], s[0], bound):
                    clusterNM.append(i)
                    pass
                pass
            pass
        st_cls = pd.DataFrame(sts, columns=['x', 'y'])
        # print(st_cls)
        st_cls['cls'] = clusterNM
        st_cls.set_index(keys=['x', 'y'], inplace=True, drop=True)
        dataMG.stationCls(data, st_cls)
        return data

    def addClustr_Intersection(data, st_cls):
        cls = dataMG(st_cls).read()
        cls.set_index(keys=['id'], inplace=True)
        stCLS = dataMG.stationCls_id(data, cls)
        return stCLS

    def in_out_cls(data):
        l_in = []
        l_out = []
        for index, row in data.iterrows():
            if row['start_st_cluster'] == row['end_st_cluster']:
                l_out.append(0)
                l_in.append(1)
                pass
            else:
                l_in.append(0)
                l_out.append(1)
                pass
            pass
        data['in_cls'] = pd.DataFrame(l_in)
        data['out_cls'] = pd.DataFrame(l_out)
        return data

    def contextToCluster(data, st_cls, contexts):
        cls_ob = dataMG(st_cls)
        cls = cls_ob.read()
        cls = cls[['Shape_Area', 'id_2']].copy()
        d_all = dataMG.addClustr_Intersection(data, r'CSV/stationCLS.csv')

        def keyGen_forArea(context):
            path_con = context.split('/')[-1]
            con = path_con.split('.')[0]
            key = con.split('_')[0]
            return key

        def clean_context(context, keys):
            d = pd.read_csv(context)
            d = d[keys].copy()
            return d

        def att_to_Cls(cls, context):
            # min_max_sc = preprocessing.MinMaxScaler()
            d = clean_context(context, keys=['area', 'id_2'])
            cls = cls.join(d.set_index('id_2'), on='id_2')
            cls['area'] = (cls['area'] * 1000) / cls['Shape_Area']
            return cls['area']

        def att_to_contex(contexts):
            n = len(contexts)
            cls_area = pd.read_csv(contexts[0])['Shape_Area']
            main = clean_context(contexts[0], keys=['area', 'id_2'])
            main['area'] = (main['area'] * 1000) / cls_area
            for i in range(n):
                if i > 0:
                    context = clean_context(contexts[i], keys=['area', 'id_2'])
                    context['area'] = (context['area'] * 1000) / cls_area
                    context = main.join(context.set_index('id_2'), on='id_2',
                                        lsuffix='_' + keyGen_forArea(contexts[i]),
                                        how='outer')
                    pass
                else:
                    continue
                pass
            return context

        context_all = att_to_contex(contexts)
        d_all_start = d_all.join(context_all.set_index('id_2'), on='start_st_cluster', lsuffix='_start')
        out = d_all_start.fillna(value=0)
        return out

    def dir_tendency(data, filename='trip_line.geojson'):
        fn = os.path.join(os.getcwd(), filename)
        raw = data
        lines = []
        vec_l = []
        for index, row in raw.iterrows():
            start_pt = (row['start station latitude'], row['start station longitude'])
            end_pt = (row['end station latitude'], row['end station longitude'])
            line = geojson.geometry.LineString([start_pt, end_pt])
            lines.append(line)
            # print(line)
            trip_vec = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
            len_trip_vec = np.sqrt(np.square(end_pt[0] - start_pt[0]) + np.square(end_pt[1] - start_pt[1]))
            # vec.append(trip_vec)
            # vec_l.append(len_trip_vec)
            pass
        geocal = geojson.GeometryCollection(lines)
        with open(fn, 'w') as fl:
            fl.write(str(geocal))
            fl.close()
            pass
        # print(type(vec[0]))
        # raw['vec_X'] = pd.DataFrame()
        # raw['vec_l'] = pd.DataFrame(vec_l)
        return None

    def regression_plot(data):
        '''
        compare linear relation of data lables
        :return: None
        '''
        data_h = dataMG.add_hour(data)
        data_d = dataMG.add_dayofmonth(data_h)
        ax = plt.gca()
        pie = data_d.plot(kind='line', x='dayofmonth', y='hours', ax=ax)
        plt.show()
        fig = pie.get_figure()
        fig.savefig('test.pdf')

    def tripNum_time(data, saveName='TripNum_time.pdf'):
        '''
        this function make a datetime table base on the number of the trips
        :return: Series
        '''
        dt_h = dataMG.add_hour(data)
        dt_d = dataMG.add_dayofmonth(dt_h)
        dt_d =dataMG.clean(dt_d)
        tn = pd.Series(dt_d['birth year'], name='TripNum').value_counts(sort=False)
        tn_d = tn.to_frame()
        tn_d['birth year'] = tn_d.index
        ax = plt.gca()
        pie = tn_d.plot(kind='line', x='birth year', ax=ax)
        plt.show()
        fig = pie.get_figure()
        fig.savefig(saveName)
        print(pd.DataFrame.idxmax(tn_d))
        return tn_d

    def tripNum_St(data, saveName='TripNum_st.pdf', start=True):
        '''
        this function make a start station table base on the number of the trips
        :return: dataframe
        '''
        unq_st = pd.read_csv(filepath_or_buffer=r'NY_BikeStations.csv')
        if start:
            key = 'start station id'
            saveName = 'start_' + saveName
        else:
            key = 'end station id'
            saveName = 'end_' + saveName
            pass

        sn = pd.Series(data[key], name='TripNum').value_counts(sort=False)
        sn_d = sn.to_frame()
        sn_d[key] = sn_d.index
        st_tn_loc = sn_d.join(unq_st.set_index('id'), on=key)
        pd.DataFrame.dropna(st_tn_loc, inplace=True)
        ax = plt.gca()
        pie = sn_d.plot(kind='scatter', x=key, y='TripNum', ax=ax)
        plt.show()
        # fig = pie.get_figure()
        # fig.savefig(saveName)
        return st_tn_loc


'''
fp = r'CSV/201903-citibike-tripdata.csv'
st_cls = r'CSV/stationCLS.csv'
#fp = r'2019-03-01 00_00_00-2019-03-08 00_00_01-citibike-tripdata.csv'
data = dataMG(fp=fp).read()
#print(data.iloc[0])
d = dataMG.contextToCluster(data,st_cls,contexts=['CSV/Res_in_cls.csv','CSV/Com_in_cls.csv'])

#print(d.head(5))

#dataMG.exportCSV(d,'cls.csv')

#dataMG.addClustr_Intersection(data,st_cls)

#st = dataMG.unique_stations(data.read())
#st = pd.DataFrame(st,columns=['id','latitude','longitude'])

#dataMG.exportCSV(st,'NY_BikeStations.csv')
#s = dataMG.boundClustring(data.read(),r'CSV/nycdwi.csv')
#data.censusToCluster(census=r'CSV/nyc_census_tracts.csv',cn_loc=r'CSV/census_block_loc.csv')

#print(data.toBinary('end_st_cluster').iloc[0])
#data=data.kminClusterSt(9)
time = dataMG.data_inPeriod(data,frac=1,freq='30D')
#print(time[0].iloc[0])

dataMG.exportCSV(data=time[0],filename=time[1])
'''
