import sys
sys.path.append("/home/rse/rseprojects")
import time
from multiprocessing import Pool
from sklearn import linear_model, preprocessing

from datetime import datetime, date
from functools import partial
from math import sqrt
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from sshtunnel import HandlerSSHTunnelForwarderError

# from alpha.advda.lib.helpers import Helper
# from alpha.advda.lib.anomaly_base import AnomalyBase

from helpers import Helper
from anomaly_base import AnomalyBase
residual = 1.0e-1
MIN_PERCENTAGE = 0.2
ABS_COS_ANGLE_DIFF = 0.75

UPPER_MIN_PERCENTAGE = 0.8
MIN_ABS_COS_ANGLE_DIFF = 0.3

def _get_slope(df, field, drive_id):
 
    dfDrive = df[df["DRIVE_ID"]==drive_id]        
   
    X_cur = dfDrive["NORMALIZED_EVENT_DATE_TIMESTAMP"].to_numpy()
    y_cur = dfDrive[f"NORMALIZED_{field}"].to_numpy()
            
    ransac = linear_model.RANSACRegressor(max_trials=100, residual_threshold=residual)               
    ransac.fit(X_cur.reshape(-1,1), y_cur.reshape(-1,1))       
    inlier_mask = ransac.inlier_mask_
    X_in, y_in = X_cur[inlier_mask==True], y_cur[inlier_mask==True]
    
    model = linear_model.LinearRegression()
    model.fit(X_in.reshape(-1,1), y_in.reshape(-1,1))
    return model.coef_[0], model.intercept_

def _anomaly_drives_by_field(subDf, field):                    
    res = {}
    print(f"RUNNING: {field}")
    try:
        subDf[field] = subDf[field].apply(float)
    except Exception as e:
        try:
            subDf[field] = subDf[field].apply(lambda x: int(str(x), 16))
        except Exception as e:
            return res
    
    res = {}
    capacities = subDf["CAPACITY"].unique()
    capacities.sort()
    for capacity in  capacities:
       
        driveInfo = _anomaly_drives_by_field_per_capacity(subDf[subDf["CAPACITY"]==capacity], field)
        # driveInfo = {"": [], "anomaly_drives": []}
        if driveInfo and driveInfo.get("anomaly_drives"):
                res["normal_drives"] = res.get("normal_drives", []) + driveInfo["normal_drives"]
                res["anomaly_drives"] = res.get("anomaly_drives",[]) + driveInfo.get("anomaly_drives", [])
    print({field: res})
    if res:
        return {field: res}
    else:
        return {}

def _anomaly_drives_by_field_per_capacity(subDf, field):
    
    features = ["EVENT_DATE_TIMESTAMP"]

    X = subDf.loc[:, features].values        
    y = subDf[field].to_numpy()
    
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    yy = y[:, np.newaxis]
    scaler = preprocessing.StandardScaler().fit(yy)
    y_scaled = np.squeeze(scaler.transform(yy))
    
    subDf.loc[:, f"NORMALIZED_{field}"] = y_scaled
    subDf = subDf.assign(NORMALIZED_EVENT_DATE_TIMESTAMP = X_scaled)
    
    driveIDs = subDf["DRIVE_ID"].unique()
    
    slopeDF = pd.DataFrame(columns=['drive_id', 'capacity', 'slope'])
    
    for drive_id in driveIDs:
        coef, intercept = _get_slope(subDf, field, drive_id)        
        slopeDF = slopeDF.append({'drive_id': drive_id, 'capacity': subDf[subDf['DRIVE_ID']==drive_id]['CAPACITY'].reset_index(drop=True)[0], 'slope': np.arctan(coef)}, ignore_index=True)
        
    anomalyDrives, normalDrives = _anomaly_drives_detection_helper(slopeDF)
    return {"anomaly_drives": [[drive, None, None] for drive in anomalyDrives], "normal_drives": normalDrives}
    
def _anomaly_drives_detection_helper(slopeDF):
    res = []
    slopes = slopeDF['slope']
    mean = np.mean(slopes, axis=0)        
    sd = np.std(slopes, axis=0)
    if sd <= 0.1:
        return [], []

    driveSlopes = [ [row['slope'], row['drive_id']] for _, row in slopeDF.iterrows()]
    driveSlopes.sort()
    last = len(driveSlopes) - 1
    normalRightMostPos = 0
    while normalRightMostPos < len(driveSlopes) and driveSlopes[normalRightMostPos][0]< mean + 1.75*sd:
        normalRightMostPos +=1
    normalRightMostPos -= 1
    while last>normalRightMostPos and last>0 and driveSlopes[last][0] - driveSlopes[last-1][0] < 0.75*sd:
        last -=1
    if last > normalRightMostPos and last>0:
        while last < len(driveSlopes):
            res.append(driveSlopes[last][1])
            last +=1
    
    normalLeftMostPos = 0
    while normalLeftMostPos < len(driveSlopes) and driveSlopes[normalLeftMostPos][0]< mean - 1.75*sd:
        normalLeftMostPos +=1
    normalLeftMostPos-=1
    
    first = 0
    while first < normalLeftMostPos and first < len(driveSlopes)-1 and driveSlopes[first][0] - driveSlopes[first+1][0] > -0.75*sd:
        first +=1
    
    if first < normalLeftMostPos and first < len(driveSlopes)-1:
        while first >0:
            res.append(driveSlopes[first][1])
            first -=1

    if not res and sd > 0.2 and driveSlopes:
        res.append(driveSlopes[-1][1])
    elif not res:
        i =  len(driveSlopes) - 1
        while i >= 0 and driveSlopes[i][0]>= mean + 2.5*sd:
            res.append(driveSlopes[i][1])
            i = i - 1
    return res, list(set(slopeDF['drive_id'].to_list()) - set(res))


class AnomalyGrowthRate(AnomalyBase):
    def __init__(self, tuid, collection = [], fields = None):
        self.tuid=tuid
        self.helper = Helper()
        self.collection = collection
        if fields is None and collection:
            fields = self.helper.get_fields_from_collection(collection,  {'$and': [{'11': {'$in': [tuid]}}]})
            self.fields = [field.split()[0] for field in fields]
        else:
            self.fields = fields if isinstance(fields, list) else [fields]
        
    
    def anomaly_drives_detection(self):
        # helper = Helper()
        try:
            t0=time.time()
            
            query = {'$and': [{'11': {'$in': [self.tuid]}}]}        
            response = self.helper.get_common_data_docs(query)     
            
            list_data_ids = []
            commonDocs = []
            if response:
                for data in response:
                    data['1'] = data['1'].strftime('%Y-%m-%d %H:%M:%S')                        
                    commonDocs.append(data) #full data of commondata i.e. capacity, serialnumber, etc. 
                    list_data_ids.append(data['_id'])  
                        
            if not commonDocs:
                return {}
            ### Common DF
            commonDF = pd.DataFrame(commonDocs)
                        
            commonDF = commonDF.rename(columns = {'_id':'COMMON_DATA_ID'})            
            commonDF['0'] = commonDF['0'].astype(str) #capacity
            commonDF['2'] = commonDF['2'].astype(str) #sector size
            commonDF['9'] = commonDF['9'].astype(str) #slot
            
            whereClause = {'13': {'$in': list_data_ids }}
            
            queryFields = {'13':1, "12": 1}
            
            # alarmEvents = {}
            # if self.collection:
            #     tmp = self._anomaly_drives_detection_helper(self.collection, commonDF, whereClause, self.helper)
            #     if tmp and tmp != {}:                
            #         alarmEvents[self.collection] = tmp  
            
            for field in self.helper.decode_name_to_id(self.fields).values():
                queryFields[field] = 1
            
            alarmEvents = {}
            if self.collection:
                tmp = self._anomaly_drives_detection_helper(self.collection, commonDF, whereClause, queryFields, self.helper)
                if tmp and tmp != {}:                
                    alarmEvents[self.collection] = tmp   
        except Exception as e:
            print(f"ERROR: {e}")
        # helper.paramDB.close()
        print(f'Total time: {Helper.calculate_duration(time.time() - t0)}')
        return alarmEvents
    
    # 18-2245-11001
    def _anomaly_drives_detection_helper(self, col, commonDF, whereClause, queryFields, helper):       
        print("=======================================")
        print(f"RUNNING {col}")

        response = helper.collection_data(col, whereClause, queryData=queryFields)        
        result = []    
        for data in response:
            result.append(data)
        fieldDF = pd.DataFrame(result)
        fieldDF = fieldDF.rename(columns = {'13':'COMMON_DATA_ID'})
        
        # colIdName = helper.get_col_id_name()
        colIdName = helper.decode_id_to_name()
        
        ### Merge DF
        t0 = time.time()
        try:
            mergeDF = pd.merge(commonDF, fieldDF, on='COMMON_DATA_ID')
        except Exception as e:
            print(f"ERROR: Can not merge commonDF and fieldDF on 'COMMON_DATA_ID.")            
            return {}
        
        columnsList = mergeDF.columns.values.tolist()
        columnsList.remove('COMMON_DATA_ID')
        for col in columnsList:
            mergeDF.rename(columns={col: colIdName.get(col, col)},inplace=True) #decode column names i.e. '11' -> TUID
        

        # print(f'Merge mergeDF {mergeDF.shape} took: {Helper.calculate_duration(time.time() - t0)}')
        # mergeDF.to_csv(
        #     "C:\\Users\\giang.bui\\projects\\PMS\\development\\mergeDF_new.txt"
        # )
        # mergeDF = pd.read_csv("C:\\Users\\giang.bui\\projects\\PMS\\development\\mergeDF_new.txt", error_bad_lines=False, index_col=False, dtype="unicode")
        
        mergeDF["EVENT_DATE_TIMESTAMP"] = pd.to_datetime(mergeDF["EVENT_DATE"])
        mergeDF["EVENT_DATE_TIMESTAMP"] = mergeDF["EVENT_DATE_TIMESTAMP"].apply(datetime.timestamp)
        
        return self._anomaly_drives(mergeDF)


    def _anomaly_drives(self, df):
        print(f"Starting to detect anomaly drives")
        alarm_list = {}
        features = ["EVENT_DATE_TIMESTAMP"]
        driveIDs = df.DRIVE_ID.unique()
        L = []
        for field in df.columns:
            if field in ['id', 'COMMON_DATA_ID', 'CAPACITY', 'START_TIME', 'FIRMWARE_REV', 'TUID', 'SECTOR_SIZE', 'CHAMBER', 'DRIVE_ID']:
                continue
            if self.fields and field not in self.fields:
                continue
            L.append((df[features+[field, "CAPACITY", "DRIVE_ID"]], field))
        
        
        print(f"Pool len: {len(L)}")       
        # for l in L:
        #     _anomaly_drives_by_field(l[0], l[1])

        if len(L) > 0:
            with Pool(8) as pool:
                results = pool.starmap(_anomaly_drives_by_field, L)

            for res in results:
                if res and res != {}:               
                    # alarm_list.update({"anomaly_drives": res[0], "normal_drives": res[1].tolist()})
                    alarm_list.update(res)
        print(f"Finish detecting anomaly drives")
        return alarm_list
    
    def insert_anomaly_data_to_db(self, jsonData):
        dt = datetime.combine(date.today(), datetime.min.time())
        try:
            self.helper.create_or_update("COL_ANOMALY", {"date": dt, "tuid": self.tuid, "RULE": "GROWTH_RATE", "abnormal_data": jsonData}, condition={"date": dt, "RULE": "GROWTH_RATE_OUTLIER", "tuid": self.tuid})    
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == '__main__':    

    # testingIDs = AnomalyGrowthRate.get_all_testing_tuid()
    
    t0 = time.time()
    # testingIDs = ["18-2377-11690", "18-2245-11001"]
    for tuid in ["18-2704-12649"]:
        print(tuid)
        # alarm = AnomalyLinearity(tuid, collections=["COL_IDENTIFY_DEVICE_DATA"])
        alarm = AnomalyGrowthRate(tuid, collection="COL_HYNIX_EXTENDED_SMART_DEFENSE_ALGORITHM",fields=["DA_HRR"])
        # import pdb; pdb.set_trace()
        # alarm = AnomalyLinearity("31-78-3129")
        # alarm = AnomalyLinearity("18-2291-11229")
        abnormalDrives = alarm.anomaly_drives_detection()   
        if abnormalDrives:
            print(f"Insert {abnormalDrives} to database")
            alarm.insert_anomaly_data_to_db(abnormalDrives)
    t1 = time.time()
    
    # Helper.close_db_session()

    print(f"Total time {Helper.calculate_duration(t1-t0)}")
