USE MWP_CWF

SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
DROP TABLE IF EXISTS [dbo].[INTB_Correction]
CREATE TABLE [dbo].[INTB_Correction](
    [INTB_Version] tinyint NOT NULL,
	[DataType] varchar(10) NOT NULL,
	[LocID] [int] NOT NULL,
	[Slope] [float] NULL,
	[Intercept] [float] NULL
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[INTB_Correction] ADD  CONSTRAINT [PK_INTB_Correction] PRIMARY KEY CLUSTERED 
(
	[INTB_Version] ASC,
    [DataType] ASC,
    [LocID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO
GO
insert into [dbo].[INTB_Correction]
-- INTB 2
SELECT 2 INTB_Version,'spring' DataType,ID,Slope,Intercept
FROM OPENROWSET( BULK '/home/MWP-IHM/INTB2_EDA results/spring_regression_params.csv'
    , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/spring_regression_params.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
) A

UNION
SELECT 2 INTB_Version,'river' DataType,ID,Slope,Intercept
FROM OPENROWSET( BULK '/home/MWP-IHM/INTB2_EDA results/flow_regression_params.csv'
    , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/flow_regression_params.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
) A

UNION
SELECT 2 INTB_Version,'waterlevel' DataType,INTB_OWID ID,Slope,Intercept
FROM OPENROWSET( BULK '/home/MWP-IHM/INTB2_EDA results/wl_regression_params.csv'
    , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/wl_regression_params.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
) A
INNER JOIN (
    SELECT INTB_OWID,PointName
    FROM [dbo].[OROP_SASwells]
UNION
    SELECT INTB_OWID,PointName
    FROM [dbo].[OROP_UFASwells]
) B ON A.PointName=B.PointName AND A.PointName<>'EW-2S-Deep'

UNION
-- INTB 1
SELECT 1 INTB_Version,'spring' DataType,ID,Slope,Intercept
FROM OPENROWSET( BULK '/home/MWP-IHM/INTB1_EDA results/spring_regression_params.csv'
    , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/spring_regression_params.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
) A

UNION
SELECT 1 INTB_Version,'river' DataType,ID,Slope,Intercept
FROM OPENROWSET( BULK '/home/MWP-IHM/INTB1_EDA results/flow_regression_params.csv'
    , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/flow_regression_params.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
) A

UNION
SELECT 1 INTB_Version,'waterlevel' DataType,INTB_OWID ID,Slope,Intercept--,A.PointName
FROM OPENROWSET( BULK '/home/MWP-IHM/INTB1_EDA results/wl_regression_params.csv'
    , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/wl_regression_params.xml'
    , FIRSTROW = 2
    , MAXERRORS = 1
) A
INNER JOIN (
    SELECT INTB_OWID,PointName
    FROM [dbo].[OROP_SASwells]
UNION
    SELECT INTB_OWID,PointName
    FROM [dbo].[OROP_UFASwells]
) B ON A.PointName=B.PointName AND A.PointName<>'EW-2S-Deep'
GO

select * 
from INTB_Correction 
where INTB_Version=2 and DataType='spring'
order by INTB_Version,DataType,LocID

        select 'ID_'+RIGHT(FORMAT(CAST(LocID as FLOAT)/100.,'N2'),2) ID,Slope,Intercept
        from MWP_CWF.dbo.INTB_Correction
        where INTB_Version=2 and DataType='spring'