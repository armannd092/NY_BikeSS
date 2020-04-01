import DataMG

def main():
    fp = r'CSV/201903-citibike-tripdata.csv'
    st_cls = r'CSV/stationCLS.csv'
    contexts=['CSV/Res_in_cls.csv','CSV/Com_in_cls.csv']
    obj = DataMG.dataMG(fp)
    filename = DataMG.dataMG.exp_name(obj,freq='30D',n='01')
    data = obj.read()
    data = DataMG.dataMG.add_hour(data)
    data = DataMG.dataMG.add_dayofweek(data)
    data = DataMG.dataMG.stationCLs_join(data,st_cls)
    data = DataMG.dataMG.in_out_cls(data)
    data = DataMG.dataMG.toBinary(data,['start_st_cluster','end_st_cluster','dayofweek','hours'],[7,7,3,5])
    data = DataMG.dataMG.contextToCluster(data,st_cls,contexts=contexts)
    data = DataMG.dataMG.data_inPeriod(self=obj,data=data,frac=0.3,freq='30D')
    #filename=data[1]
    #data = data[0]
    DataMG.dataMG.exportCSV(data,filename)
def experiment_01():
    fp = r'CSV/201903-citibike-tripdata.csv'
    obj = DataMG.dataMG(fp)
    filename = DataMG.dataMG.exp_name(obj, freq='30D', n='Sample')
    data = obj.read()
    data =  DataMG.dataMG.sampler(data,frac=0.05)
    #DataMG.dataMG.exportCSV(data, filename)
    data = DataMG.dataMG.dir_tendency(data)
    #print(data.head())
def experiment_03():
    fps = [r'CSV/201903-citibike-tripdata.csv',r'CSV/201908-citibike-tripdata.csv']
    #fps = ["2019-citibike-tripdata.csv"]
    datas=[]
    for fp in fps:
        obj = DataMG.dataMG(fp)
        data = obj.read()
        datas.append(data)
    dt = DataMG.dataMG.tripNum_time_Comp(datas,saveName='TripNum__hour_Comp.pdf', key = "hours")
    print(dt)
    return None

def experiment_02():
    #fp = r'2019-citibike-tripdata-test03S30.csv'
    fps = [ r'CSV/201901-citibike-tripdata.csv',
            r'CSV/201902-citibike-tripdata.csv',
            r'CSV/201903-citibike-tripdata.csv',
             r'CSV/201904-citibike-tripdata.csv',
            r'CSV/201905-citibike-tripdata.csv',
            r'CSV/201906-citibike-tripdata.csv',
            r'CSV/201907-citibike-tripdata.csv',
            r'CSV/201908-citibike-tripdata.csv',
            r'CSV/201909-citibike-tripdata.csv',
            r'CSV/201910-citibike-tripdata.csv',
            r'CSV/201911-citibike-tripdata.csv',
            r'CSV/201912-citibike-tripdata.csv',
            ]
    #obj = DataMG.dataMG(fp)
    #data = obj.read()
    data = DataMG.dataMG.massRead(fps)
    dt = DataMG.dataMG.tripNum_age(data,saveName="tripAgegroup2019.pdf")
    #dt = DataMG.dataMG.tripNum_time(data,saveName='TripNum_age_2019.pdf', key = "age")
    #print(dt)
    return None
def experiment():
    fp = r'CSV/201908-citibike-tripdata.csv'
    obj = DataMG.dataMG(fp)
    data = obj.read()
    data_clean = DataMG.dataMG.clean(data)
    #data_cls = DataMG.dataMG.kminClusterSt(data_clean,k=22)
    #print(data_cls)
    #ds = DataMG.dataMG.tripNum_St(data_clean,start=False)
    dr = DataMG.dataMG.regression_plot(data_clean)
    #DataMG.dataMG.exportCSV(data_cls,'st_cls_kmean.csv')

def experiment_04():
    rest = [ r'CSV/201901-citibike-tripdata.csv',
            r'CSV/201902-citibike-tripdata.csv',
            r'CSV/201903-citibike-tripdata.csv',]
    fps = [ r'CSV/201904-citibike-tripdata.csv',
            r'CSV/201905-citibike-tripdata.csv',
            r'CSV/201906-citibike-tripdata.csv',]
    rest = [ r'CSV/201907-citibike-tripdata.csv',
            r'CSV/201908-citibike-tripdata.csv',
            r'CSV/201909-citibike-tripdata.csv',
            r'CSV/201910-citibike-tripdata.csv',
            r'CSV/201911-citibike-tripdata.csv',
            r'CSV/201912-citibike-tripdata.csv',]


    st_cls = r'CSV/stationCLS.csv'
    contexts = ['CSV/Res_in_cls.csv', 'CSV/Com_in_cls.csv']
    filename = "2019-citibike-tripdata-test04S30.csv"
    data = DataMG.dataMG.massRead(fps)
    data = DataMG.dataMG.sampler(data,frac=0.03)
    data = DataMG.dataMG.add_hour(data)
    data = DataMG.dataMG.add_dayofweek(data)
    data = DataMG.dataMG.stationCLs_join(data, st_cls)
    data = DataMG.dataMG.station_id(data)
    data = DataMG.dataMG.station_id(data,key="end station id")
    #data = DataMG.dataMG.in_out_cls(data)
    data = DataMG.dataMG.toBinary(data, ['start_st_cluster', 'end_st_cluster', 'dayofweek', 'hours', "st_id_start", "st_id_end"], [5, 5, 3, 5,10,10])
    #data = DataMG.dataMG.toBinary(data, ['start_st_cluster', 'end_st_cluster', 'hours', "st_id"],[7, 7, 5, 10])
    #data = DataMG.dataMG.toBinary(data, ['dayofweek'],[3])
    data = DataMG.dataMG.contextToCluster(data, st_cls, contexts=contexts)
    #print(data.iloc[0])
    # data = DataMG.dataMG.data_inPeriod(self=obj, data=data, frac=0.3, freq='30D')
    # filename=data[1]
    # data = data[0]
    DataMG.dataMG.exportCSV(data, filename)
def experiment_05():
    fp = r'CSV/201906-citibike-tripdata.csv'
    obj = DataMG.dataMG(fp)
    data = obj.read()
    data = DataMG.dataMG.station_id(data)
    data = DataMG.dataMG.toBinary(data, ["st_id"],[10])
    print(data.iloc[0])

if __name__ == '__main__':
    # main()
    experiment_02()