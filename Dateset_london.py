import DataMG
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataMG_london(DataMG.dataMG):

    def unique_stations(data):
        """
        make a data with unique station from the trip data
        :return: data frame
        """
        startSt = data[['StartStation Id']].copy()
        endSt = data[['EndStation Id']].copy()
        ss_np = pd.DataFrame.to_numpy(startSt)
        es_np = pd.DataFrame.to_numpy(endSt)
        alls = np.concatenate((ss_np, es_np), axis=0)
        sts = DataMG.unique_rows(alls)
        sts = pd.DataFrame(sts)
        pd.DataFrame.dropna(sts, inplace=True)
        return sts
    def stationCLs_join(data, st_cls):
        '''
            joining the data with the clustered station based on station id
        :param st_cls: clustered station file path
        :return:
        '''
        cls = pd.read_csv(st_cls)
        cls = cls[['id', 'id_01']].copy()
        data_start = data.join(cls.set_index('id'), on='StartStation Id')
        data_start.rename(columns={'id_01': 'start_st_cluster'}, inplace=True)
        data_start_end = data_start.join(cls.set_index('id'), on='EndStation Id', lsuffix='_end_cls')
        data_start_end.rename(columns={'id_01': 'end_st_cluster'}, inplace=True)
        data_start_end['start_st_cluster'] = pd.to_numeric(data_start_end['start_st_cluster'], downcast='integer')
        data_start_end.dropna(inplace=True)
        return data_start_end

    def station_id(data,key="start station id"):
        """
        this function take the station ids and renumber them
        :return: Data
        """
        def nam(key):
            sk = key.split(" ")
            if sk[0] == "StartStation":
                return "st_id_start"
            elif sk[0] == "EndStation":
                return "st_id_end"
        sts = DataMG_london.unique_stations(data)
        sts_list = sts[0].values.tolist()
        sts_dic = {}
        for i in range(len(sts_list)):
            sts_dic[i] = [i,sts_list[i]]
        sts_renum_data=pd.DataFrame.from_dict(sts_dic , orient="index")
        sts_renum_data = sts_renum_data.rename(columns={0:nam(key),1:key})
        data = data.join(other=sts_renum_data.set_index(key), how="left", on=key, rsuffix='_right')
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
            ks = {}
            for i in range(n):
                nm = key + str(i + 1)
                ks[i]=nm
                pass
            return ks

        dt = data

        for k,n in zip(key,nods):
            data_key = pd.DataFrame(dt[k].copy())
            data_key['Rental Id']=dt['Rental Id']
            print("\n")
            print("__________________________________________________________________________________________________")
            nms = clmname(k, n)
            dic = {}
            for q,i in data_key.iterrows():
                if i[0] is not None:
                    ii=int(i[0])
                else:
                    i = -1
                bin_mask = "0" + str(n) + "b"
                b = format(ii, bin_mask)
                sb = str(b)
                sbl = split(sb)
                ibl = toInt(sbl)
                dic[q]=ibl
            pass
            #print(dic)
            d = pd.DataFrame(dic).T
            d.rename(columns=nms, inplace=True)
            dt = pd.DataFrame.join(dt, d,how='left')
        pass
        return dt

    def tripNum_time_Comp(datas, saveName='TripNum_time_comp.pdf', key=""):
        '''
        this function make a datetime table base on the number of the trips
        :return: Series
        '''

        for data in datas:
            dt_h = DataMG.dataMG.add_hour(data,key="Start Date")
            dt_h = DataMG.dataMG.add_dayofweek(dt_h,key="Start Date")
            dt_d = DataMG.dataMG.add_dayofmonth(dt_h,key="Start Date")
            dt_d = DataMG.dataMG.add_dayofyear(dt_h, key="Start Date")
            if key:
                tn = pd.Series(dt_d[key], name='TripNum').value_counts(sort=False)
                tn_d = tn.to_frame()
                tn_d[key] = tn_d.index
                if key == "dayofmonth":
                    tn_end = dt_d[[key, "dayofweek"]].drop_duplicates()
                    tn_end.set_index(key)
                    tn_d_merg = pd.merge(left=tn_d, right=tn_end, how="inner", left_on=key, right_on=key)
                    mark_on = tn_d_merg.loc[tn_d_merg["dayofweek"].isin([5, 6])]
                    plt.scatter(x=mark_on[key], y=mark_on['TripNum'])
                ax = plt.gca()
                pie = tn_d.plot(kind='line', x=key, ax=ax, grid=True)
                fig = pie.get_figure()
                fig.savefig(saveName)
                print(pd.DataFrame.idxmax(tn_d))
        plt.show()
        return None


def experiment_lnd_01():
    fp = r"CSV/LND/151JourneyDataExtract27Feb2019-05Mar2019.csv"
    filename = "2019-santanderbike-London-test01S30.csv"
    st_cls = r'lndBikeCLS.csv'
    obj = DataMG.dataMG(fp)
    data = obj.read()
    data = DataMG.dataMG.sampler(data,frac=1)
    data = DataMG.dataMG.add_hour(data,key="Start Date")
    data = DataMG.dataMG.add_dayofweek(data, key="Start Date")
    data = DataMG_london.station_id(data,key="StartStation Id")
    data = DataMG_london.station_id(data, key="EndStation Id")
    data = DataMG_london.stationCLs_join(data,st_cls)
    data = DataMG_london.toBinary(data, ['dayofweek', 'hours', "st_id_start", "st_id_end",'start_st_cluster','end_st_cluster'], [3, 5, 10, 10,7,7])
    DataMG.dataMG.exportCSV(data, filename)
    #print(data.iloc[0])
def experiment_lnd_02():
    fps = [r"CSV/LND/151JourneyDataExtract27Feb2019-05Mar2019.csv",
           r"CSV/LND/152JourneyDataExtract06Mar2019-12Mar2019.csv",
           r"CSV/LND/153JourneyDataExtract13Mar2019-19Mar2019.csv",
           r"CSV/LND/154JourneyDataExtract20Mar2019-26Mar2019.csv",
           r"CSV/LND/154JourneyDataExtract20Mar2019-26Mar2019.csv"
           ]
    filename = "2019-santanderbike-London-test02S100.csv"
    st_cls = r'lndBikeCLS.csv'
    data = DataMG.dataMG.massRead(fps)
    data = DataMG.dataMG.sampler(data,frac=1)
    data = DataMG.dataMG.add_hour(data,key="Start Date")
    data = DataMG.dataMG.add_dayofweek(data, key="Start Date")
    data = DataMG_london.station_id(data,key="StartStation Id")
    data = DataMG_london.station_id(data, key="EndStation Id")
    data = DataMG_london.stationCLs_join(data,st_cls)
    data = DataMG_london.toBinary(data, ['dayofweek', 'hours', "st_id_start", "st_id_end",'start_st_cluster','end_st_cluster'], [3, 5, 10, 10,7,7])
    DataMG.dataMG.exportCSV(data, filename)
def experiment_lnd_03():
    fps = [r"CSV/LND/151JourneyDataExtract27Feb2019-05Mar2019.csv",
            r"CSV/LND/152JourneyDataExtract06Mar2019-12Mar2019.csv",
           r"CSV/LND/153JourneyDataExtract13Mar2019-19Mar2019.csv",
           r"CSV/LND/154JourneyDataExtract20Mar2019-26Mar2019.csv",
           r"CSV/LND/154JourneyDataExtract20Mar2019-26Mar2019.csv"
           ]
    data = DataMG.dataMG.massRead(fps)
    dt = DataMG_london.tripNum_time_Comp([data],saveName='LNDTripNum__dayofmonth_Comp.pdf', key = "dayofyear")
if __name__ == '__main__':
    # main()
    experiment_lnd_03()