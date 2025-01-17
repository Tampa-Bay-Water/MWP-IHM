-- Insert rows SpringflowTS & StreamflowTS in INTB_Output database
CREATE TABLE [INTB_Output].[dbo].[SpringflowTS] (
    [Date]     DATE       NOT NULL,
    [SpringID] INT        NOT NULL,
    [Value]    FLOAT (53) NULL,
    CONSTRAINT [PK_SpringflowTS] PRIMARY KEY CLUSTERED ([Date] ASC, [SpringID] ASC)
);

INSERT INTO [INTB_Output].[dbo].[SpringflowTS]
SELECT a.Date, b.SpringId
    , a.ROVOL*43560.0/86400.0 AS Springflow
FROM OPENROWSET(BULK '/mnt/IHM/INTB2_bp424/PEST_Run/ReachHistory.csv'
    , FORMATFILE='/mnt/IHM/BCP_format/ReachHistory_format.xml', FIRSTROW=2) as a
INNER JOIN (
    SELECT * FROM INTB2_Input.dbo.Spring WHERE SpringId NOT IN (6,8,9,11)
    ) as b ON b.ReachID = a.ReachID
ORDER BY b.SpringID,Date

CREATE TABLE [INTB_Output].[dbo].[StreamflowTS] (
    [Date]          DATE       NOT NULL,
    [FlowStationID] INT        NOT NULL,
    [Value]         FLOAT (53) NULL,
    CONSTRAINT [PK_NewTable] PRIMARY KEY CLUSTERED ([Date] ASC, [FlowStationID] ASC)
);

INSERT INTO [INTB_Output].[dbo].[StreamflowTS]
SELECT a.Date, b.FlowStationId
    ,SUM((a.ROVOL*43560.0)/86400.0 * b.ScaleFactor) Streamflow
FROM OPENROWSET(BULK '/mnt/IHM/INTB2_bp424/PEST_Run/ReachHistory.csv'
    , FORMATFILE='/mnt/IHM/BCP_format/ReachHistory_format.xml', FIRSTROW=2) as a
INNER JOIN INTB2_Input.dbo.ReachFlowStation b ON b.ReachID = a.ReachID
GROUP BY a.Date, b.FlowStationId
ORDER BY b.FlowStationId,a.Date

--- Find CellID and RiverCellID needed for diffuse flows for Crystal Springs
SELECT P.PolygonID,WF.WaterbodyFragmentID,R.RivercellID,R.LayerNumber
    ,WF.CellID,P.hasChannel,WF.IsChannel,WF.Area,P.PctContributed
    ,FHC.CellID FullHistCellID
    --,RCH.*
FROM MWP_CWF.dbo.CrystalSpringsPolygon P
INNER JOIN [INTB2_Input].[dbo].[WaterbodyFragmentToWaterbodyPolygonMap] PF ON P.PolygonID=PF.PolygonID
INNER JOIN [INTB2_Input].[dbo].[WaterbodyFragment] WF ON PF.WaterbodyFragmentID=WF.WaterbodyFragmentID
INNER JOIN [INTB2_Input].[dbo].[Rivercell] R ON PF.WaterbodyFragmentID=R.WaterbodyFragmentID
LEFT JOIN  [INTB2_Input].[dbo].[FullHistoryCell] FHC on WF.CellID=FHC.CellID
/*
INNER JOIN OPENROWSET( BULK '/mnt/IHM/INTB2_bp424/PEST_RunPlus/RivercellHistory.csv'
    , FORMATFILE = '/mnt/IHM/BCP_format/RivercellHistory_format.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
    ) RCH ON R.RivercellID=RCH.RivercellID
*/
WHERE WF.IsChannel=1
ORDER BY P.PolygonID,WF.WaterbodyFragmentID,R.LayerNumber

--- Update HullHistoryCell table to rerun INTB2
insert into [C:\RUN\IHM\BP424_19892006\INTB2_INPUT.MDF].[dbo].[FullHistoryCell]
values
	(101131, 'Crystal Springs diffuse flow', 101131),
	(100132, 'Crystal Springs diffuse and vent flow', 100132)

--- RivercellID for Spring Vent
select SF.SpringFragmentID,SF.SpringID,[Description],CellID,SF.LayerNumber
from [dbo].[Rivercell] R 
INNER JOIN [dbo].[SpringFragment] SF ON R.SpringFragmentID=SF.SpringFragmentID
WHERE RivercellID=85281

--- Compute Lower Gate Flow
CREATE TABLE [MWP_CWF].[dbo].[CrystalSpringsEstimateTS] (
    [Date]            DATE       NOT NULL,
    [ChannelBaseflow] FLOAT (53) NULL,
    [Reachflow83]     FLOAT (53) NULL,
    [SpringsVent]     FLOAT (53) NULL,
    [UpperGage]       FLOAT (53) NULL,
    [UpperGage_Obs]   FLOAT (53) NULL,
    [TotalSpringflow] FLOAT (53) NULL,
    [LowerGage]       FLOAT (53) NULL,
    [LowerGage_Obs]   FLOAT (53) NULL,
    CONSTRAINT [PK_CrystalSpringsEstimateTS] PRIMARY KEY CLUSTERED ([Date] ASC)
);
INSERT INTO [MWP_CWF].[dbo].[CrystalSpringsEstimateTS]
SELECT A.Date,ChannelBaseFlow,ReachFlow,C.Value SpringVent
    ,D.Value UpperGage, UpperGage_Obs
    ,ChannelBaseFlow+ReachFlow+C.Value TotalSprings
    ,ChannelBaseFlow+ReachFlow+C.Value+D.Value LowerGage
    ,LowerGage_Obs
FROM ( 
--- Channel Flows
SELECT RCH.Date,-Sum(Baseflow)/24/3600 ChannelBaseFlow
FROM MWP_CWF.dbo.CrystalSpringsPolygon P
INNER JOIN [INTB2_Input].[dbo].[WaterbodyFragmentToWaterbodyPolygonMap] PF ON P.PolygonID=PF.PolygonID
INNER JOIN [INTB2_Input].[dbo].[WaterbodyFragment] WF ON PF.WaterbodyFragmentID=WF.WaterbodyFragmentID
INNER JOIN [INTB2_Input].[dbo].[Rivercell] R ON PF.WaterbodyFragmentID=R.WaterbodyFragmentID
--LEFT JOIN  [INTB2_Input].[dbo].[FullHistoryCell] FHC on WF.CellID=FHC.CellID
INNER JOIN OPENROWSET( BULK '/mnt/IHM/INTB2_bp424/PEST_Run/RivercellHistory.csv'
    , FORMATFILE = '/mnt/IHM/BCP_format/RivercellHistory_format.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
    ) RCH ON R.RivercellID=RCH.RivercellID
WHERE WF.IsChannel=1
GROUP BY Date
--ORDER BY Date
) A

--- Reach Flow
INNER JOIN (
SELECT Date,ROVOL*0.5041667 ReachFlow
FROM OPENROWSET( BULK '/mnt/IHM/INTB2_bp424/PEST_Run/ReachHistory.csv'
    , FORMATFILE = '/mnt/IHM/BCP_format/ReachHistory_format.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
    ) RCH
WHERE ReachID=83
) B ON A.Date=B.Date

--- Spring Vent
INNER JOIN (
SELECT [Date], [Value] FROM [INTB_Output].[dbo].[SpringflowTS] WHERE SpringId=2
) C on A.Date=C.Date

--- Streamflow
INNER JOIN (
SELECT [Date], [Value] FROM [INTB_Output].[dbo].[StreamflowTS] WHERE FlowStationID=2
) D on A.Date=D.Date

--- Observed Flow Lower & Upper Gages
LEFT JOIN (
SELECT CAST(datetime as Date) [Date],discharge LowerGage_Obs
FROM OPENROWSET(BULK '/home/mssql/MWP_CWF_csv/02302010.csv'
    , FORMATFILE='/home/mssql/MWP_CWF_csv/02302010.xml', FIRSTROW=3) as a
) E ON E.Date=A.Date
LEFT JOIN (
SELECT CAST(datetime as Date) [Date],discharge UpperGage_Obs
FROM OPENROWSET(BULK '/home/mssql/MWP_CWF_csv/02301990.csv'
    , FORMATFILE='/home/mssql/MWP_CWF_csv/02301990.xml', FIRSTROW=3) as a
) F ON F.Date=A.Date
ORDER BY A.DATE



