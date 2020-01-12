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
def experiment_02():
    fp = r'CSV/201903-citibike-tripdata.csv'
    obj = DataMG.dataMG(fp)
    data = obj.read()
    dt = DataMG.dataMG.tripNum_time(data,saveName='TripNum_age5.pdf')
    print(dt)
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
if __name__ == '__main__':
    # main()
    experiment_02()