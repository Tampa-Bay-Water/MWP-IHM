import numpy as np
import os
import sys
import warnings
import pandas as pd
import pyodbc
import sqlalchemy as sa
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sbn
import re

is_Windows = os.name=='nt'
owinfo_sql = f'''
    SELECT PointID,A.PointName,WFCode,CTSID,SiteID,INTB_OWID,[Name],A.CellID,LayerNumber
        ,Target,TargetType,NewLandElevation-Topo SurfEl2CellTopoOffset
    FROM (
        SELECT PointID,PointName,'OROP CP' PermitType,WFCode,CTSID,SiteID,CellID,INTB_OWID
        FROM [MWP_CWF].[dbo].[OROP_SASwells]
        UNION
        SELECT PointID,PointName,PermitType,WFCode,CTSID,SiteID,CellID,INTB_OWID
        FROM [MWP_CWF].[dbo].[OROP_UFASwells]
    ) A
    LEFT JOIN [INTB2_Input].[dbo].[ObservedWell] OW ON OW.ObservedWellID=A.INTB_OWID
    LEFT JOIN (
        SELECT PointName,TargetWL Target,'OROP_CP' TargetType
        FROM [MWP_CWF].[dbo].[RA_TargetWL]
        UNION
        SELECT PointName,AvgMin Target,'Regulartory' TargetType
        FROM [MWP_CWF].[dbo].[RA_RegWellPermit]
        UNION
        SELECT PointName,MinAvg Target,'SWIMAL' TargetType
        FROM [MWP_CWF].[dbo].[swimalWL]
    ) C ON A.PointName=C.PointName
    INNER JOIN [INTB2_Input].[dbo].[Cell] B on B.CellID=A.CellID
    WHERE INTB_OWID IS NOT NULL
    ORDER BY WFCode,PointName
'''

def get_DBconn():
    import urllib
    warnings.simplefilter(action='ignore', category=UserWarning)
    if is_Windows:
        dv = '{SQL Server}'
        sv = 'localhost'
        db = 'MWP_CWF'
        conn = pyodbc.connect(
            f'DRIVER={dv};SERVER={sv};Database={db};Trusted_Connection=Yes',autocommit=True)
        # params = urllib.parse.quote_plus(f"DRIVER={dv};SERVER={sv};Database={db};Trusted_Connection=Yes")
        # engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
        # conn = engine.connect()
    else:
        dv = '/opt/homebrew/Cellar/msodbcsql17/17.10.6.1/lib/libmsodbcsql.17.dylib'
        sv = 'localhost'
        db = 'MWP_CWF'
        pw = os.environ['DATABASE_SA_PASSWORD']
        conn = pyodbc.connect(
            f'DRIVER={dv};SERVER={sv};Database={db};Uid=SA;Pwd={pw}',
            autocommit=True
        )
        # params = urllib.parse.quote_plus(f"DRIVER={dv};SERVER={sv};DATABASE={db};UID=SA;PWD={pw}")
        # engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
        # conn = engine.connect()       
    return conn


class ReadHead:
    # Read head from MODFLOW binary data file

    def __init__(self,run_dir):
        # Header np datatype
        self.header_dt = np.dtype([
            ('Nlays',np.int32),
            ('Period',np.int32),
            ('PeriodTime',np.float32),
            ('TotalTime',np.float32),
            ('Text','S16'),
            ('Columns',np.int32),
            ('Rows',np.int32),
            ('Layer',np.int32)
        ])

        self.run_dir = run_dir
        self.fname = os.path.join(run_dir,'PEST_Run','IHM_Binary_Files','Head.IHM_COPY')
        x = np.fromfile(self.fname,dtype=self.header_dt,count=1)
        self.nwords = 4
        self.Nlays = x['Nlays'][0]
        self.nrows = x['Rows'][0]    # pass by reference
        self.ncols = x['Columns'][0] # pass by reference
        self.nlays = x['Nlays'][0]   # pass by reference
        self.hbytes = 44
        self.nbytes = self.nrows*self.ncols*self.nwords + self.hbytes # one layer
    
        # Header and one layer data
        self.layer_dt = np.dtype([
            ('header',self.header_dt),
            ('head',np.float32,self.nrows*self.ncols)
        ])

    def get_DBconn(self):
        return get_DBconn()
    
    '''
    def get_DBconn(self):
        warnings.simplefilter(action='ignore', category=UserWarning)
        if self.is_Windows:
            dv = '{SQL Server}'
            sv = 'vgridfs'
            db = 'MWP_CWF'
            conn = pyodbc.connect(
                f'DRIVER={dv};SERVER={sv};Database={db};Trusted_Connection=Yes',autocommit=True)
        else:
            dv = '/opt/homebrew/Cellar/msodbcsql17/17.10.6.1/lib/libmsodbcsql.17.dylib'
            sv = 'localhost'
            db = 'MWP_CWF'
            pw = os.environ['DATABASE_SA_PASSWORD']
            conn = pyodbc.connect(
                f'DRIVER={dv};SERVER={sv};Database={db};Uid=SA;Pwd={pw}',
                autocommit=True
            )
        return conn
    '''

    def readHeader(self,offset=0):
        return np.fromfile(self.fname,dtype=self.header_dt, offset=offset, count=1)

    def readHeadByTstep(self,tstep=1,layers=[1,3]):
        if type(layers) is not list:
            layers = [layers]
        if max(layers>self.Nlays):
            print(f"Layer number must be <= {self.Nlays}.", file=sys.stderr)
            exit(1)
        layers = [l-1 for l in layers]

        if tstep!=None:
            tstep = tstep-1
            offset = tstep*3*self.nbytes
            temp = np.fromfile(self.fname,dtype=self.layer_dt,offset=offset, count=3)[layers]
        else:
            temp = np.fromfile(self.fname,dtype=self.layer_dt)
            indices = [i for i,l in enumerate(temp['header']['Layer']) if l-1 in layers]
            temp = temp[indices]           
        return temp['head'],list(temp['header']['Period']),list(temp['header']['Layer'])

    def readHeadMultiTsteps(self,tsteps=[1,11],layers=[1,3]):
        if type(tsteps) is int:
            tsteps = [tsteps]
        elif type(tsteps) is range:
            tsteps = list(tsteps)
        elif type(tsteps) is list:
            pass
        elif tsteps==None:
            return self.readHeadByTstep(None,layers)
        else:
            print(f"Data type of 'tsteps' as '{type(tsteps)}' is not supported." +
                f"\nUse int, list, range data type instead.", file=sys.stderr)
            exit(1)

        tsteps = [t-1 for t in tsteps]
        if type(layers) is not list:
            layers = [layers]
        if max(layers>self.Nlays):
            print(f"Layer number must be <= {self.Nlays}.", file=sys.stderr)
            exit(1)
        layers = [l-1 for l in layers]

        t0 = min(tsteps)
        offset = t0*3*self.nbytes
        tsteps_span = (max(tsteps)-t0)+1
        tstep_layer = [(t-t0)*3+l for t in tsteps for l in layers]
        temp = np.fromfile(
            self.fname,dtype=self.layer_dt,offset=offset,count=3*tsteps_span
            )[tstep_layer]
        return temp['head'],list(temp['header']['Period']),list(temp['header']['Layer'])
    
    def readHeadByCellIDs(self,tsteps=1,layers=[1,3],CIDs=100123):
        if type(CIDs) is int:
            CIDs = [CIDs]
        elif type(CIDs) is range:
            CIDs = list(CIDs)
        elif type(CIDs) is list:
            pass
        else:
            print(f"Data type of 'CIDs' as '{type(CIDs)}' is not supported." +
                f"\nUse int, list, range data type instead.", file=sys.stderr)

        cellord = [(i+1)*1000+j+1 for i in range(0,self.nrows) for j in range(0,self.ncols)]
        cellindex = [cellord.index(k) for k in CIDs]

        x,t,l = self.readHeadMultiTsteps(tsteps,layers)
        return x[:, cellindex],t,l
    
    def getHeadByWellnames(self,wnames,por=None,date_index=True,need_weekly=False):
        # This function get INTB2 simulated heads for a specified list of OROP wells
        if type(wnames) is not list:
            print(f"Expecting 'wnames' to be a list!")
            wnames = [wnames]
        
        # Get known list of OROP wells (SAS & UFAS) with CellID and Layer Number from database
        conn = self.get_DBconn()
        df = pd.read_sql(f'''
            SELECT A.PointName,CellID, 3 Layer, AvgMin Target 
			FROM OROP_UFASwells A
			INNER JOIN RA_RegWellPermit B ON A.PointName=B.PointName
			UNION
            SELECT A.PointName,CellID, 3 Layer, MinAvg Target 
			FROM OROP_UFASwells A
			INNER JOIN swimalWL B ON A.PointName=B.PointName
            UNION
            SELECT A.PointName,CellID, 1 Layer, TargetWL Target 
			FROM OROP_SASwells A
			LEFT JOIN RA_TargetWL B ON A.PointName=B.PointName
        ''',conn)

        # update layer number fron owinfo
        owinfo = pd.read_sql(owinfo_sql,conn)
        owinfo.loc[owinfo.PointName=='WRW-s','LayerNumber'] = 3
        for i in owinfo.PointName:
            df.loc[df.PointName==i,'Layer'] = np.int8(owinfo.LayerNumber[owinfo.PointName==i])[0]

        df0 = df[['PointName','Target']]
        self.Target = dict(zip(df0['PointName'], df['Target']))

        # Get date range of simulation
        self.Calendar = pd.read_sql(f'''
            SELECT ROW_NUMBER() OVER(ORDER BY Date ASC) AS rownum
                ,CONVERT(varchar, [Date], 23) [Date]
                ,CONVERT(varchar, WeekStart, 23) WeekStart
            FROM [INTB2_Input].[dbo].[INTBCalendar]
            ORDER BY Date
        ''',conn)
        conn.close()

        # Convert specified list of wells to 
        cursor = [i for i,n in enumerate(df.PointName) if n in wnames]
        df = df.iloc[cursor]
        cellids = df.CellID.to_list()

        # Matching por to range of timesteps
        if por!=None:
            sdate = datetime.strptime(por[0],'%m/%d/%Y').date()
            edate = datetime.strptime(por[1],'%m/%d/%Y').date()
            s_stp = [i for i,j in enumerate(self.Calendar['Date']) if j==sdate][0]+1
            e_stp = [i for i,j in enumerate(self.Calendar['Date']) if j==edate][0]+1
            tsteps = range(s_stp,e_stp+1)

            # Read head from MODFLOW binary head file and extract TS by layers of the wells
            head,tsteps,layer = self.readHeadByCellIDs(tsteps,[1,3],cellids)
        else:
            head,tsteps,layer = self.readHeadByCellIDs(por,[1,3],cellids)
        
        # extract columns according to layer
        temp = []
        for j in range(0,len(cellids)):
            indices = [i for i,l in enumerate(layer) if l==df.iloc[j].Layer]
            temp.append(head[indices,j])
        temp = np.column_stack(temp)
        tsteps = [tsteps[i] for i in indices]
        dates = pd.date_range(str(self.Calendar.Date[tsteps[0]-1]), periods=len(tsteps), freq='D')
        if date_index:
            tempDF = pd.DataFrame(temp, columns=df.PointName.to_list(),index=dates)
        else:
            tempDF = pd.concat(
                [pd.DataFrame(self.Calendar.Date[range(tsteps[0]-1,tsteps[-1])])
                ,pd.DataFrame(temp)], axis=1)
            tempDF.columns = ['Date']+df.PointName.to_list()
        if need_weekly:
            tempDF = self.computeWeeklyAvg(tempDF)
        tempDF.index.name = 'Date'
        return tempDF

    def computeWeeklyAvg(self,dailyData):
        tempDF = self.Calendar.set_index('Date')
        dailyData['WeekStart'] = [tempDF.loc[d.strftime('%Y-%m-%d')].WeekStart for d in dailyData.index]
        tempDF = dailyData.groupby('WeekStart').mean()
        tempDF.index = pd.to_datetime(tempDF.index)
        return tempDF
    
    def plotData(self,df):
        pointname = df.columns.to_list()[0]
        if np.isnan(self.Target[pointname]):
            pass
        else:
            df['Target'] = self.Target[pointname]
        fig = plt.figure(figsize=(13,9))
        ax = fig.add_subplot(111)
        sbn.lineplot(ax=ax, data=df, linewidth=0.75, marker=None
            # marker='.', markeredgewidth=0, markersize=4
        )
        ax.set_ylabel('Waterlevel, ft NGVD')
        ax.set_xlabel('Date')
        plt.grid(True)
        plt.title(pointname)
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()
        return fig

class LoadObservedHead:
    # Load observation head from database

    def __init__(self,run_dir):
        self.run_dir = run_dir
        self.fname = os.path.join(run_dir,'PEST_Run','IHM_Binary_Files','Head.IHM_COPY')
        conn = self.get_DBconn()
        self.owinfo_df = pd.read_sql(owinfo_sql,conn)

        # modify table
        self.owinfo_df.loc[self.owinfo_df.PointName=='WRW-s','LayerNumber'] = 3 # simulation has no SAS
        self.owinfo_df.loc[self.owinfo_df.PointName=='RMP-8D1','SurfEl2CellTopoOffset'] = 1.799998

        conn.close()

    def get_DBconn(self):
        return get_DBconn()

    def loadHead(self,wnames,por=None,date_index=True,need_weekly=False):
        if type(wnames) is not list:
            wnames = [wnames]
        indices = [i for i,w in enumerate(self.owinfo_df.PointName) if w in wnames]
        if len(indices)<len(wnames):
            notOROP = set(wnames).difference(set(self.owinfo_df.PointName[indices]))
            raise Warning(f"{notOROP} not in OROP!")
        owinfo = self.owinfo_df.iloc[indices]

        if need_weekly:
            weekstart = ',TS.dbo.WeekStart1([Date]) WeekStart'
        else:
            weekstart = ''

        owStrList1 = str(owinfo['PointName'].to_list()).replace('[','(').replace(']',')')
        owStrList2 = re.sub(r"'([^'']*)'", r'[\1]',owStrList1).replace('(','').replace(')','')
        conn = self.get_DBconn()
        df = pd.read_sql(f'''
            SELECT [Date],{owStrList2}{weekstart}
            FROM (
                SELECT PointName,CAST([Date] as DATE) Date,[Value]
                FROM (
                    SELECT PointName,INTB_OWID FROM [MWP_CWF].[dbo].[OROP_SASwells]
                    UNION
                    SELECT PointName,INTB_OWID FROM [MWP_CWF].[dbo].[OROP_UFASwells]
                ) A
                INNER JOIN [INTB2_Input].[dbo].[ObservedWell] OW ON OW.ObservedWellID=A.INTB_OWID
                INNER JOIN [INTB2_Input].[dbo].[ObservedWellTimeSeries] OWTS ON OW.ObservedWellID=OWTS.ObservedWellID
                WHERE PointName in {owStrList1}
            ) A PIVOT (avg([Value]) FOR PointName in ({owStrList2})) P
            ORDER BY [Date]
        ''',conn)
        conn.close()

        if need_weekly:
            df = df.groupby('WeekStart')[wnames].mean().reset_index()   # Move the date index to a column
            df = df.rename(columns={'WeekStart': 'Date'})
            df = df.reset_index(drop=True)  # Reindex the DataFrame with a new index
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            
        if date_index:
            df.set_index('Date', inplace=True)
        return df

    def plotData(self,df):
        pointname = df.columns.to_list()[0]
        df['Target'] = self.owinfo_df.Target[self.owinfo_df.PointName.to_list().index(pointname)]
        fig = plt.figure(figsize=(13,9))
        ax = fig.add_subplot(111)
        sbn.lineplot(ax=ax, data=df, linewidth=0.75, marker=None
            # marker='.', markeredgewidth=0, markersize=4
        )
        ax.set_ylabel('Waterlevel, ft NGVD')
        ax.set_xlabel('Date')
        plt.grid(True)
        plt.title(pointname)
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()
        return fig

class GetFlow:
    # Get INTB2 flow data

    def __init__(self,run_dir,is_river=True):
        self.run_dir = run_dir.replace('/Volumes/Mac_xSSD','/mnt')
        self.fname = os.path.join(run_dir,'PEST_Run','ReachHistory.csv')
        self.dbin = 'INTB2_Input'
        self.dbout = 'INTB_Output'
        conn = self.get_DBconn()
        with conn.cursor() as cur:
            cur.execute(f'''
                    IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name = '{self.dbin}')
                    BEGIN
                        CREATE DATABASE {self.dbin} 
                        ON (FILENAME = '/var/opt/mssql/data/{self.dbin}.mdf'), 
                        (FILENAME = '/var/opt/mssql/data/{self.dbin}_log.ldf') 
                        FOR ATTACH;
                    END
                '''
            )
            cur.execute(f'''
                    IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name = '{self.dbout}')
                    BEGIN
                        CREATE DATABASE {self.dbout} 
                        ON (FILENAME = '/var/opt/mssql/data/{self.dbout}.mdf'), 
                        (FILENAME = '/var/opt/mssql/data/{self.dbout}_log.ldf') 
                        FOR ATTACH;
                    END
                '''
            )
        if is_river:
            self.FlowStationInfo = pd.read_sql_query(f'''
                SELECT FlowStationID,SiteNumber,StationName,[Description],UseInCalibration
                FROM [INTB2_Input].[dbo].[FlowStation]
                WHERE UseInCalibration=1
                ORDER BY FlowStationID
            ''',conn)
        else:
            self.SpringInfo = pd.read_sql_query(f'''
                SELECT SpringID,Name SpringName,[Description]
                FROM [INTB2_Input].[dbo].[Spring]
                ORDER BY SpringID
            ''',conn)
        conn.close()

    def get_DBconn(self):
        return get_DBconn()

    def getStreamflow_Table(self,stationIDs,SorceOrigin='Sim',need_weekly=False,date_index=True,por=None):
        if type(stationIDs) is not list:
            stationIDs = [stationIDs]
        stalist = str(stationIDs).replace('[','(').replace(']',')')
        idlist = str([f"ID_{i:02}" for i in stationIDs]).replace('[','').replace(']','').replace("'","")

        if SorceOrigin=='Sim':
            db_source = 'INTB_Output.dbo.StreamflowTS'
        else:
            db_source = 'INTB2_Input.dbo.FlowStationTimeseries'
            
        if need_weekly:
            weekstart = ',TS.dbo.WeekStart1([Date]) WeekStart'
        else:
            weekstart = ''

        conn = self.get_DBconn()
        df = pd.read_sql(f"""
            select Date,{idlist}{weekstart} from (
            select Date,'ID_'+right(left(cast(FlowStationID/100. as varchar),4),2) StaID,Value
            from {db_source}
            where FlowStationID in {stalist}
            ) A pivot (AVG(Value) for StaID in ({idlist})) B
            order by Date
        """,conn)
        conn.close()

        if need_weekly:
            idlist = [f"ID_{i:02}" for i in stationIDs]
            df = df.groupby('WeekStart')[idlist].mean().reset_index()
            df = df.rename(columns={'WeekStart': 'Date'})
            df = df.reset_index(drop=True)
            
        if date_index:
            df.set_index('Date', inplace=True)
        return df
    
    def getStreamflow_Vector(self,stationIDs,SorceOrigin='Sim',need_weekly=False,date_index=True,por=None):
        if type(stationIDs) is not list:
            stationIDs = [stationIDs]
        stalist = str(stationIDs).replace('[','(').replace(']',')')

        if SorceOrigin=='Sim':
            db_source = 'INTB_Output.dbo.StreamflowTS'
        else:
            db_source = 'INTB2_Input.dbo.FlowStationTimeseries'

        if need_weekly:
            weekstart = ',TS.dbo.WeekStart1([Date]) WeekStart'
        else:
            weekstart = ''

        conn = self.get_DBconn()
        df = pd.read_sql(f"""
            select Date,'ID_'+right(left(cast(FlowStationID/100. as varchar),4),2) StaID,Value{weekstart}
            from {db_source}
            where FlowStationID in {stalist}
            order by FlowStationID,Date
        """,conn)
        conn.close()

        if need_weekly:
            df = df.groupby(['WeekStart','StaID'])['Value'].mean().reset_index()
            df = df.rename(columns={'WeekStart': 'Date'})
            df = df.reset_index(drop=True)  # Reindex the DataFrame with a new index
            
        if date_index:
            df.set_index(['Date','StaID'], inplace=True)
        return df

    def getSpringflow_Table(self,springIDs,SorceOrigin='Sim',need_weekly=False,date_index=True,por=None):
        if type(springIDs) is not list:
            springIDs = [springIDs]
        stalist = str(springIDs).replace('[','(').replace(']',')')
        idlist = str([f"ID_{i:02}" for i in springIDs]).replace('[','').replace(']','').replace("'","")

        if SorceOrigin=='Sim':
            db_source = 'INTB_Output.dbo.SpringflowTS'
        else:
            db_source = 'INTB2_Input.dbo.ObservedSpringTimeSeries'
            
        if need_weekly:
            weekstart = ',TS.dbo.WeekStart1([Date]) WeekStart'
        else:
            weekstart = ''

        conn = self.get_DBconn()
        df = pd.read_sql(f"""
            select Date,{idlist}{weekstart} from (
            select Date,'ID_'+right(left(cast(SpringID/100. as varchar),4),2) SprID,Value
            from {db_source}
            where SpringID in {stalist}
            ) A pivot (AVG(Value) for SprID in ({idlist})) B
            order by Date
        """,conn)
        conn.close()

        if need_weekly:
            idlist = [f"ID_{i:02}" for i in springIDs]
            df = df.groupby('WeekStart')[idlist].mean().reset_index()
            df = df.rename(columns={'WeekStart': 'Date'})
            df = df.reset_index(drop=True)
            
        if date_index:
            df.set_index('Date', inplace=True)
        return df
    
    def getSpringflow_Vector(self,springIDs,SorceOrigin='Sim',need_weekly=False,date_index=True,por=None):
        if type(springIDs) is not list:
            springIDs = [springIDs]
        stalist = str(springIDs).replace('[','(').replace(']',')')

        if SorceOrigin=='Sim':
            db_source = 'INTB_Output.dbo.SpringflowTS'
        else:
            db_source = 'INTB2_Input.dbo.ObservedSpringTimeSeries'

        if need_weekly:
            weekstart = ',TS.dbo.WeekStart1([Date]) WeekStart'
        else:
            weekstart = ''

        conn = self.get_DBconn()
        df = pd.read_sql(f"""
            select Date,'ID_'+right(left(cast(SpringID/100. as varchar),4),2) SprID,Value{weekstart}
            from {db_source}
            where SpringID in {stalist}
            order by SpringID,Date
        """,conn)
        conn.close()

        if need_weekly:
            df = df.groupby(['WeekStart','SprID'])['Value'].mean().reset_index()
            df = df.rename(columns={'WeekStart': 'Date'})
            df = df.reset_index(drop=True)  # Reindex the DataFrame with a new index
            
        if date_index:
            df.set_index(['Date','SprID'], inplace=True)
        return df


if __name__ == '__main__':
    proj_dir = os.path.dirname(os.path.dirname((__file__)))
    run_dir = os.path.join(proj_dir,'INTB2_bp424')

    exit(0)

