import numpy as np
import os
import sys
import pandas as pd
import pyodbc
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import LoadData as ld
from LoadData import ReadHead, LoadObservedHead

def plot_MP(w,lowh,rh,need_weekly=False,q=None,sema=None):
    print(f"starting histogram plot '{w}'")
    df,Target = get1WideTable(w,lowh,rh,need_weekly)
    df = getMatchDataTable(w,df)
    # In case simulation and observation data are not overlapped, df is empty
    if len(df)==0:
        print(f"No data for '{w}'!")
        return None

    fig = [plotRegression1(df,Target)]+[plotHist(df,Target)]
    if sema is not None:
        sema.release()

    if q is not None:
        plt.show(block=False)
        if q!=-1:
            q.put(fig)

    return fig

def plotHist(df,Target):
    w = df.columns.tolist()[0]
    if len(df)==0:
        print(f"No data for '{w}'!")
        return None
    # Create a 2x2 subplot grid
    fig = plt.figure(figsize=(10, 8))
    sns.set_theme(style="darkgrid")
    axes = {n: fig.add_subplot(2, 2, n) for n in range(1,5)}

    sns.histplot(data=df, x=w, hue="DataOrigin", element="step", kde=True, ax=axes[1])
    axes[1].set_title(f"Compare Histograms of '{w}'")
    axes[2].set_ylabel("Count", fontsize=8)
    
    temp = df.iloc[range(len(df))]
    temp[w] = Target
    temp['DataOrigin'] = 'Target'
    temp = pd.concat([df,temp])
    li = sns.ecdfplot(data=temp, x=w, hue="DataOrigin", ax=axes[2])
    # sns.lineplot(x=[Target,Target], y=[0,1], ax=axes[2],
    #     linestyle='--', color='black', linewidth=1) #, label='Target')
    axes[2].set_title(f"Compare CDF of '{w}'")
    axes[2].set_ylabel("Probability", fontsize=8)

    sns.boxplot(data=df, x=w, hue="DataOrigin", ax=axes[3])
    for patch in axes[3].patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))
    # sns.despine(offset=10, trim=True, ax=axes[0, 1])
    axes[3].set_title(f"Boxplot of '{w}'")

    sns.violinplot(data=df, x=w, hue="DataOrigin", ax=axes[4], split=True, inner="quart", fill=False)
    axes[4].set_title(f"Violin Plot of Histograms for '{w}'")
    axes[4].set_ylabel("Probability", fontsize=8)

    for k in range(1,5):
        axes[k].grid(color='lightgray')
        axes[k].set_xlabel("Waterlevel ft NGVD", fontsize=8)
        axes[k].tick_params(axis='both', which='major', labelsize=8)
        legend = axes[k].get_legend()
        legend.get_title().set_fontsize(9)
        for text in legend.get_texts():
            text.set_fontsize(8)

    plt.tight_layout()

    proj_dir = os.path.dirname(os.path.realpath(__file__))
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-hist')
    plt.savefig(svfilePath,dpi=300, pad_inches=0.1,facecolor='auto', edgecolor='auto')
    # plt.close()
    return fig

def regRobust(X,Y,alpha=0.05):
    import statsmodels.api as sm
    X = sm.add_constant(X)

    # Fit the robust linear regression model
    model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
    results = model.fit()

    # Make predictions
    predictions = results.predict(X)

    # Calculate prediction intervals
    # predictions_interval = results.t_test(X,alpha=alpha)
    return results,predictions

def plotRegression0(df):
    import matplotlib.gridspec as gridspec
    import SeabornFig2Grid as sfg
    w = df.columns.tolist()[0]
    if len(df)==0:
        print(f"No data for '{w}'!")
        return None
    tempDF = pd.pivot_table(df, values=w, index='Date', columns='DataOrigin', aggfunc='max')
    tempDF['ModelPeriod'] = df.ModelPeriod[df.DataOrigin=='Simulated'].to_list()

    # Create a 2x2 subplot grid
    fig = plt.figure(figsize=(14, 14))   
    axes = gridspec.GridSpec(2,2)

    g0 = sns.lmplot(data=tempDF, x="Observed", y="Simulated")
    plt.title(w)
    g1 = sns.lmplot(data=tempDF, x="Observed", y="Simulated", hue="ModelPeriod", legend=True)
    # g1.ax_joint.scatter(tempDF["Observed"], tempDF["Simulated"], edgecolor="white", linewidth=0.25) #, s=5
    plt.title(w)
    g2 = sns.jointplot(data=tempDF, x="Observed", y="Simulated", kind="kde")
    g2.ax_joint.scatter(tempDF["Observed"], tempDF["Simulated"], s=10, edgecolor="white", linewidth=0.25) #, s=5
    # g2 = sns.lmplot(data=tempDF, x="Observed", y="Simulated", hue="ModelPeriod", legend=True)
    g3 = sns.jointplot(data=tempDF, x="Observed", y="Simulated", hue="ModelPeriod", kind = 'kde', legend=True)
    g3.ax_joint.scatter(
        tempDF.loc[tempDF.ModelPeriod=='Calibration',"Observed"], 
        tempDF.loc[tempDF.ModelPeriod=='Calibration',"Simulated"], 
        s=10, color='blue', edgecolor="white", linewidth=0.25) #, s=5
    g3.ax_joint.scatter(
        tempDF.loc[tempDF.ModelPeriod=='Others',"Observed"], 
        tempDF.loc[tempDF.ModelPeriod=='Others',"Simulated"], 
        s=10, color='orange', edgecolor="white", linewidth=0.25) #, s=5

    sfg.SeabornFig2Grid(g0, fig, axes[0])
    sfg.SeabornFig2Grid(g1, fig, axes[1])
    sfg.SeabornFig2Grid(g2, fig, axes[2])
    sfg.SeabornFig2Grid(g3, fig, axes[3])

    axes.tight_layout(fig)

    proj_dir = os.path.dirname(os.path.realpath(__file__))
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-regress')
    plt.savefig(svfilePath,dpi=300, pad_inches=0.1,facecolor='auto', edgecolor='auto')
    # plt.close()
    return fig

def plotRegression1(df,Target):
    w = df.columns.tolist()[0]
    if len(df)==0:
        print(f"No data for '{w}'!")
        return None
    tempDF = pd.pivot_table(df, values=w, index='Date', columns='DataOrigin', aggfunc='max')
    tempDF['ModelPeriod'] = df.ModelPeriod[df.DataOrigin=='Simulated'].to_list()

    # single period
    # Initialize JointGrid
    g = sns.JointGrid(data=tempDF, x="Observed", y="Simulated", height=8)
    g.plot_joint(sns.kdeplot, fill=True, levels=6)
    g.plot_joint(sns.regplot, robust=True, ci=90,
        line_kws={"color":"red"},
        scatter_kws={"edgecolor": "white","linewidths": 0.25,"s": 40})
    g.plot_marginals(sns.histplot, kde=True)
    g.ax_joint.plot([Target, Target], (plt.gca().get_ylim()), color='green', linewidth=1, linestyle="--")
    g.ax_joint.grid(color='lightgray')
    g.ax_marg_x.grid(color='lightgray')
    g.ax_marg_y.grid(color='lightgray')    # g.ax_joint.plot([Target, Target], (plt.gca().get_ylim()), color='green', linewidth=1, linestyle="--")
    fig1 = plt.gcf()
    fig1.tight_layout()

    # two periods - validation and verification periods
    # use robust regression to predict separate period
    y_cal = df.loc[(df.ModelPeriod=='Calibration' ) & (df.DataOrigin=='Simulated'),w]
    y_ver = df.loc[(df.ModelPeriod=='Others') & (df.DataOrigin=='Simulated'),w]
    x_cal = df.loc[(df.ModelPeriod=='Calibration' ) & (df.DataOrigin=='Observed' ),w]
    x_ver = df.loc[(df.ModelPeriod=='Others') & (df.DataOrigin=='Observed' ),w]
    print(f"Robust Regression for '{w}'")
    if len(x_cal)>3:
        _,yp_cal = regRobust(x_cal,y_cal)
    if len(x_ver)>3:
        _,yp_ver = regRobust(x_ver,y_ver)


    # Initialize JointGrid
    g = sns.JointGrid(data=tempDF, x="Observed", y="Simulated", hue="ModelPeriod", height=8)
    g.plot_joint(sns.kdeplot, levels=6)
    g.plot_joint(sns.scatterplot, edgecolor="white" ,linewidths=0.25 ,s=20)
    g.plot_marginals(sns.histplot, kde=True)
    if len(x_cal)>3:
        g.ax_joint.plot(x_cal, yp_cal, color='blue', linewidth=3)
    if len(x_ver)>3:
        g.ax_joint.plot(x_ver, yp_ver, color='red', linewidth=3)
    g.ax_joint.plot([Target, Target], (plt.gca().get_ylim()), color='green', linewidth=1, linestyle="--")
    g.ax_joint.grid(color='lightgray')
    g.ax_marg_x.grid(color='lightgray')
    g.ax_marg_y.grid(color='lightgray')    # g.ax_joint.plot([Target, Target], (plt.gca().get_ylim()), color='green', linewidth=1, linestyle="--")
    fig2 = plt.gcf()
    fig2.tight_layout()

    proj_dir = os.path.dirname(os.path.realpath(__file__))
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-regress1')
    fig1.savefig(svfilePath,dpi=300, pad_inches=0.1,facecolor='auto', edgecolor='auto')
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-regress2')
    fig2.savefig(svfilePath,dpi=300, pad_inches=0.1,facecolor='auto', edgecolor='auto')

    # plt.close()
    return [fig1, fig2]

def get1WideTable(w,lowh,rh,need_weekly):
    # Wide format table
    obs_ts = lowh.loadHead([w],need_weekly=need_weekly)
    obs_ts.rename(columns={w:f'obs_{w}'},inplace=True)
    sim_ts = rh.getHeadByWellnames([w])
    if need_weekly:
        sim_ts = rh.computeWeeklyAvg(sim_ts)
        sim_ts.index.name = 'Date'
    sim_ts.rename(columns={w:f'sim_{w}'},inplace=True)
    df = sim_ts.join(obs_ts)
    del sim_ts, obs_ts

    df.index.name = 'Date'
    df['ModelPeriod'] = 'Calibration'
    if not df.loc['1996-01-01':'2001-12-31'].empty:
        df.loc['1996-01-01':'2001-12-31', 'ModelPeriod'] = 'Others'
    '''
    if not df.loc['2007-10-01':'2013-09-30'].empty:
        df.loc['2007-10-01':'2013-09-30', 'RA_Period'] = 'First six years'
    if not df.loc['2013-10-01':'2019-09-30'].empty:
        df.loc['2013-10-01':'2019-09-30', 'RA_Period'] = 'Last six years'
    if not df.loc['2019-10-01':'2023-09-30'].empty:
        df.loc['2019-10-01':'2023-09-30', 'RA_Period'] = 'Extended Period'
    '''
    return df,rh.Target[w]

def get1LongTable(w,lowh,rh):
    ts = []
    # Column table
    obs_ts = lowh.loadHead([w],date_index=False)
    obs_ts['DataOrigin'] = 'Observed'

    sim_ts = rh.getHeadByWellnames([w],date_index=False)
    sim_ts['DataOrigin'] = 'Simulated'

    df = pd.concat([sim_ts,obs_ts])
    ts.append(df)
    return ts

def getMatchDataTable(w,df):
    # Match data, get rid of Nan (missing value)
    matchdf = df.dropna(subset=[f'obs_{w}',f'sim_{w}'])
    df1 = matchdf[[f'sim_{w}','ModelPeriod']]
    df1['DataOrigin'] = 'Simulated'
    df1.rename(columns={f'sim_{w}':w},inplace=True)
    df2 = matchdf[[f'obs_{w}','ModelPeriod']]
    df2['DataOrigin'] = 'Observed'
    df2.rename(columns={f'obs_{w}':w},inplace=True)
    df = pd.concat([df1,df2])
    del df1,df2
    return df

def useMP_Queue(wnames,lowh,rh):
    import multiprocessing as mp
    # create queue to get return value
    q = mp.Queue()
    sema = mp.Semaphore(MAX_NUM_PROCESSES)
    procs = []
    for w in wnames:
        sema.acquire()
        p = mp.Process(target=plot_MP, args=(w,lowh,rh,need_weekly,q,sema))
        procs.append(p)
        p.start()

    # get return value
    rtnval = []
    for p in procs:
        rtnval += q.get()

    # wait for all procs to finish
    for p in procs:
        p.join()
    return rtnval

def useMP_Pool(wnames,lowh,rh):
    import multiprocessing as mp
    rtnval = []
    with mp.Pool(processes=5) as pool:
        for w in wnames:
            p = pool.apply_async(plot_MP,args=(w,lowh,rh,need_weekly))
            print(f"Return type from apply_async is '{type(p)}'.")
            r= p.get(timeout=60)
            print(f"Return type from get is '{type(r)}'.")
            rtnval += r

    return rtnval

def use_noMP(wnames,lowh,rh):
    rtnval = []
    for w in wnames:
        rtnval += plot_MP(w,lowh,rh,need_weekly)
    return rtnval


if __name__ == '__main__':
    # from memory_profiler import memory_usage
    need_weekly = True
    use_degug = False
    MAX_NUM_PROCESSES = 10
    sns.set_theme(style="darkgrid")
    
    if ld.is_Windows:
        proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    else:
        proj_dir = os.path.dirname(os.path.realpath(__file__))
    run_dir = os.path.join(os.path.dirname(proj_dir),'INTB2_bp424')

    # Perform EDA for OROP wells
    lowh = LoadObservedHead(run_dir)
    rh = ReadHead(run_dir)

    owinfo = lowh.owinfo_df.iloc[[i for i,targ in enumerate(lowh.owinfo_df.Target) if not np.isnan(targ)]]
    wnames = owinfo.PointName.to_list()
    wnames = [w for w in wnames if w not in 
        ['BUD-14fl','BUD-21fl','WRW-s','Calm-33A','Cosme-3','James-11','TMR-1','TMR-2','TMR-3','TMR-4','TMR-5'
        ,'201-M','EW-113B','EW-139G','EW-2N','EW-2S','EW-2S-Deep','RMP-13D','RMP-16D','RMP-8D1','ROMP-8D'
        ,'TARPON-RD-DEEP','Hills-13','Jacksn26A','SP-42','SP-45','SR-54']]
    if use_degug:
        # use_mp = use_noMP | useMP_Queue | useMP_Pool
        use_mp = 'use_noMP '
        wnames = ['MB-24s']
    else:
        use_mp = 'useMP_Queue'

    start_time = datetime.now()
    if use_mp=='useMP_Queue':
        rtnval = useMP_Queue(wnames,lowh,rh)
    elif use_mp=='useMP_Pool':
        rtnval = useMP_Pool(wnames,lowh,rh)
    else:
        rtnval = use_noMP(wnames,lowh,rh)
        # mem_usage = memory_usage(use_noMP)

    etime = datetime.now()-start_time
    print(f'Elasped time: {etime}')
    if use_degug:
        plt.show()

    # print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    # print('Maximum memory usage: %s' % max(mem_usage))

    # convert image to pdf
    import img2pdf
    plotWarehouse = os.path.join(proj_dir,'plotWarehouse')
    image_files = [i for i in os.listdir(plotWarehouse) if i.endswith(".png")]

    # Sort the list by creation time (the first element of each tuple)
    image_files.sort()

    with open(os.path.join(proj_dir,"all_plots.pdf"), "wb") as file:
        file.write(img2pdf.convert([os.path.join(plotWarehouse,i) for i in image_files]))

    exit(0)
