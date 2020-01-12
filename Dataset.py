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
    
if __name__ == '__main__':
     main()
    
