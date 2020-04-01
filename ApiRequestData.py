import os
import requests
import pandas as pd
import json
from DataMG import dataMG
inpath = os.getcwd()

def cleanJson(data, keys=[]):
    out=[]
    for d in data:
        out.append({key: d[key] for key in keys})
    return out
def JsonToCSV(data,keys,filename="lndBike.csv"):
    st = ""
    for k in keys:
        if keys.index(k) == len(keys) - 1:
            st = st + str(k)
        else:
            st = st + str(k) + ","
    st = st + '\n'
    print(st)
    for d in data:
        for k in keys:
            if keys.index(k)==len(keys)-1:
                st = st + "'"+str(d[k])+"'"
            else:
                st = st + "'"+ str(d[k])+"'" +","
        st = st + "\n"
    exportCSV(st,filename)

def exportCSV(data, filename):
    """
    exporting the dataframe on a csv file in the current directory
    :param filename: the name of the file
    :return: None
    """
    # build the path for openning file
    expt = os.path.join(inpath, filename)
    # open the file to write on
    with open(expt, 'w', newline='',encoding="utf-8") as tp:
        # write the data on csv
        tp.write(data)
        tp.close()
        pass
    return print("done!")
def exportJson(data, filename):
    """
    exporting the dataframe on a json file in the current directory
    :param filename: the name of the file
    :return: None
    """
    # build the path for openning file
    expt = os.path.join(inpath, filename)
    # open the file to write on
    with open(expt, 'w', newline='') as tp:
        # write the data on csv
        json.dump(data,indent=4,fp=tp)
        tp.close()
        pass
    return print("done!")

def request(html = r"https://api.tfl.gov.uk/BikePoint"):
    data = requests.get(html).json() # should check the response!
    JsonToCSV(data,keys=["commonName","lat","lon"])
    # = cleanJson(data,keys=["id","commonName","lat","lon"])
    #exportJson(data, "lndBikeAll.json")


def unique(list1):
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
def jsonTOcsv(fp = r"lndBikeAll.json"):
    with open(fp,"r") as d :
        data = json.load(d)
        d.close()
    JsonToCSV(data, keys=["id","commonName", "lat", "lon"],filename="lndBikeID.csv")
def main():
    fp = r"lndBikeIDQGIS.csv"
    data = dataMG(fp)
    data = data.read()
    data.drop(columns={"field_6"},inplace=True)
    id = data["id"].str.split('_',n = 1, expand = True)
    data["id"]=id[1]
    cls = data.regName.unique()
    cls_dic=pd.DataFrame(cls)
    size = cls_dic.size
    cls_dic["id_01"]= range(size)
    cls_dic.rename(columns={0: "regName"},inplace=True)
    newdata = pd.DataFrame.merge(data,cls_dic, on="regName")
    print(newdata)
    dataMG.exportCSV(newdata,"lndBikeCLS.csv")
if __name__ == '__main__':
    main()