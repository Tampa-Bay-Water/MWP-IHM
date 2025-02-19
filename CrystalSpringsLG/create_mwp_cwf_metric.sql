/*
USE master;
GO
IF DB_ID (N'MWP_CWF_metric') IS NOT NULL
DROP DATABASE MWP_CWF_metric;
GO
CREATE DATABASE MWP_CWF_metric;
GO
*/
USE MWP_CWF_metric
drop table [dbo].[Metric_TS]
CREATE TABLE [dbo].[Metric_TS] (
    [RUNID]         SMALLINT    NOT NULL,
    [REALIZATIONID] SMALLINT    NOT NULL,
    [DataType]      VARCHAR(10) NOT NULL,
    [LOCID]         SMALLINT    NOT NULL,
    [Date]          DATE        NOT NULL,
    [METRIC]        VARCHAR(25) NOT NULL,
    [Value]         FLOAT       NULL,
    CONSTRAINT [PK_Metric_TS] PRIMARY KEY CLUSTERED (
        [RUNID] ASC,[REALIZATIONID] ASC, [DataType] ASC, [LOCID] ASC, [Date] ASC, [METRIC] ASC)
);

drop table [dbo].[Metric]
CREATE TABLE [dbo].[Metric] (
    [RUNID]         SMALLINT    NOT NULL,
    [REALIZATIONID] SMALLINT    NOT NULL,
    [DataType]      VARCHAR(10) NOT NULL,
    [LOCID]         SMALLINT    NOT NULL,
    [STATS]         VARCHAR(20) NOT NULL,
    [METRIC]        VARCHAR(25) NOT NULL,
    [Value]         FLOAT       NULL,
    CONSTRAINT [PK_Metric] PRIMARY KEY CLUSTERED (
        [RUNID] ASC,[REALIZATIONID] ASC, [DataType] ASC, [LOCID] ASC, [STATS] ASC, [METRIC] ASC)
);

drop table [dbo].[RunDetail]
CREATE TABLE [dbo].[RunDetail] (
    [RunID]         SMALLINT    NOT NULL,
    [ScenarioID]    SMALLINT    NULL,
    [METRIC]        VARCHAR(99) NULL,
    CONSTRAINT [PK_RunDetail] PRIMARY KEY CLUSTERED (
        [RUNID] ASC)
);

/*
USE intb2_input
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
drop table [dbo].[INTBCalendar]
CREATE TABLE [dbo].[INTBCalendar](
	[Date] [Date] NOT NULL,
	[WeekStart] [Date] NULL,
	[WeekNumber] [int] NULL,
	[MonthStart] [Date] NULL,
    CONSTRAINT [PK_INTBCalendar] PRIMARY KEY CLUSTERED (
        [Date] ASC)
);

declare @sdate as Date = '1989-01-01'
declare @edate as Date = '2060-12-31'
declare @base_wkno as int = 0

--select * from TS.dbo.Weekly1(@sdate,@edate)
declare @temp TABLE ( [Date] Date, WeekStart Date, MonthStart Date)
insert into @temp
select TSTAMP Date, TS.dbo.WeekStart1(TSTAMP) [WeekStart], TS.dbo.MonthStart(Tstamp) [MonthStart]
from TS.dbo.daily(@sdate,@edate)

insert into intb2_input.[dbo].[INTBCalendar]
select b.Date, a.WeekStart, a.rowno+@base_wkno WeekNumber, MonthStart
from (
    select weekstart, ROW_NUMBER() over (ORDER BY a.weekstart) rowno
    from (select distinct weekstart from @temp) a
) a
inner join @temp b on a.weekstart=b.WeekStart
order by b.Date
*/