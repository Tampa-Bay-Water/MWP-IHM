SELECT B.PointID, B.INTB_OWID, B.PointName, WFCode
    -- , Bayesian_Slope, Bayesian_Intercept
    -- , ROUND(C.Slope,2) QC_Slope, ROUND(C.Intercept,2) QC_Intercept
    , A.Cal_Slope, A.Cal_Intercept
    , ROUND(C.Cal_Slope,2) QC_Cal_Slope, ROUND(C.Cal_Intercept,2) QC_Cal_Intercept
    -- , A.Ver_Slope, A.Ver_Intercept
    -- , ROUND(C.Ver_Slope,2) QC_Ver_Slope, ROUND(C.Ver_Intercept,2) QC_Ver_Intercept
    , ROUND(A.Cal_Slope-C.Cal_Slope,2) Cal_Slope_Diff, ROUND(A.Cal_Intercept-C.Cal_Intercept,2) Cal_Intercept_Diff
    , D.TargetType
FROM OPENROWSET( BULK '/home/MWP-IHM/GSI_simobs_corr/bp_424_GW_Only_Intercepts_Slopes.csv'
    , FORMATFILE = '/home/MWP-IHM/GSI_simobs_corr/bp_424_GW_Only_Intercepts_Slopes.xml'
    , FIRSTROW = 2
    , MAXERRORS = 2
) A
INNER JOIN (
    SELECT PointID, PointName, WFCode, INTB_OWID, 'OROP' PermitType
    FROM [dbo].[OROP_SASwells]
    UNION
    SELECT PointID, PointName, WFCode, INTB_OWID, PermitType
    FROM [dbo].[OROP_UFASwells]
) B ON A.ID=B.INTB_OWID
INNER JOIN 
    OPENROWSET( BULK '/home/MWP-IHM/INTB2_EDA results/wl_regression_params.csv'
        , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/wl_regression_params.xml'
        , FIRSTROW = 2
        , MAXERRORS = 2
    ) C ON C.PointName=B.PointName
LEFT JOIN (
    SELECT PointName,TargetWL Target,'OROP_CP' TargetType
    FROM [dbo].[RA_TargetWL]
    UNION
    SELECT PointName,AvgMin Target,'Regulartory' TargetType
    FROM [dbo].[RA_RegWellPermit]
    UNION
    SELECT PointName,MinAvg Target,'SWIMAL' TargetType
    FROM [dbo].[swimalWL]
) D ON B.PointName=D.PointName
--WHERE ABS(A.Cal_Slope-C.Cal_Slope)>0.05 --OR 
    --ABS(A.Cal_Intercept-C.Cal_Intercept)>1
ORDER BY WFCode,B.PointName




SELECT C.ID, C.DataType,C.LocName
    -- , Bayesian_Slope, Bayesian_Intercept
    -- , ROUND(C.Slope,2) QC_Slope, ROUND(C.Intercept,2) QC_Intercept
    , A.Cal_Slope, A.Cal_Intercept
    , ROUND(C.Cal_Slope,2) QC_Cal_Slope, ROUND(C.Cal_Intercept,2) QC_Cal_Intercept
    -- , A.Ver_Slope, A.Ver_Intercept
    -- , ROUND(C.Ver_Slope,2) QC_Ver_Slope, ROUND(C.Ver_Intercept,2) QC_Ver_Intercept
    , ROUND(A.Cal_Slope-C.Cal_Slope,2) Cal_Slope_Diff, ROUND(A.Cal_Intercept-C.Cal_Intercept,2) Cal_Intercept_Diff
FROM OPENROWSET( BULK '/home/MWP-IHM/GSI_simobs_corr/bp_424_stream_spring_Intercepts_Slopes.csv'
    , FORMATFILE = '/home/MWP-IHM/GSI_simobs_corr/bp_424_stream_spring_Intercepts_Slopes.xml'
    , FIRSTROW = 2
    , MAXERRORS = 2
) A
RIGHT JOIN (
    SELECT A.*, 'stream' DataType, B.StationName LocName
    FROM OPENROWSET( BULK '/home/MWP-IHM/INTB2_EDA results/flow_regression_params.csv'
        , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/flow_regression_params.xml'
        , FIRSTROW = 2
        , MAXERRORS = 2
    ) A
    INNER JOIN [dbo].[FlowStation] B ON A.ID=B.FlowStationID
    UNION
    SELECT A.*, 'spring' DataType, B.Name LocName
    FROM OPENROWSET( BULK '/home/MWP-IHM/INTB2_EDA results/spring_regression_params.csv'
        , FORMATFILE = '/home/MWP-IHM/INTB2_EDA results/spring_regression_params.xml'
        , FIRSTROW = 2
        , MAXERRORS = 2
    ) A
    INNER JOIN [dbo].[Spring] B ON A.ID=B.SpringID
) C ON A.ID=C.ID AND SUBSTRING(A.TARGET,8,6)=C.DataType
--WHERE ABS(A.Cal_Slope-C.Cal_Slope)>0.05 --OR 
    --ABS(A.Cal_Intercept-C.Cal_Intercept)>1
ORDER BY A.ID
