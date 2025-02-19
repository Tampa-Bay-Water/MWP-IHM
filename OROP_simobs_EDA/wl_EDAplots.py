import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import LoadData as ld
from LoadData import ReadHead, LoadObservedHead
import yaml

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(ROOT_DIR,'config.yaml')
with open(CONFIG_FILE, 'r') as file:
    try:
        CONFIG = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"\033[91mError parsing YAML file: {e}\033[0m", file=sys.stderr)
        CONFIG = None

if ('debugpy' in sys.modules and sys.modules['debugpy'].__file__.find('/.vscode/extensions/') > -1):
    IS_DEBUGGING = True
else:
    IS_DEBUGGING = False
# IS_DEBUGGING = CONFIG['general']['IS_DEBUGGING']
MAX_NUM_PROCESSES = CONFIG['general']['MAX_NUM_PROCESSES']
INTB_VERSION = CONFIG['general']['INTB_VERSION']
FILE_REGRESSION_PARAMS = CONFIG['general']['FILE_REGRESSION_PARAMS']

FIG_TITLE_FONTSIZE = CONFIG['plotting']['FIG_TITLE_FONTSIZE']
TITLE_FONTSIZE = CONFIG['plotting']['TITLE_FONTSIZE']
AX_LABEL_FONTSIZE = CONFIG['plotting']['AX_LABEL_FONTSIZE']

if INTB_VERSION==1:
    # Period of Analysis INTB1
    POA = [CONFIG['INTB1']['POA_sdate'],CONFIG['INTB1']['POA_edate']]
    # Calibration period INTB1
    CAL_PERIOD = [CONFIG['INTB1']['CAL_PERIOD_sdate'],CONFIG['INTB1']['CAL_PERIOD_edate']]
    RUN_DIRNAME = CONFIG['INTB1']['RUN_DIRNAME']
else:
    # Period of Analysis INTB1
    POA = [CONFIG['INTB2']['POA_sdate'],CONFIG['INTB2']['POA_edate']]
    # Calibration period INTB1
    CAL_PERIOD = [CONFIG['INTB2']['CAL_PERIOD_sdate'],CONFIG['INTB2']['CAL_PERIOD_edate']]
    RUN_DIRNAME = CONFIG['INTB2']['RUN_DIRNAME']


def plot_MP(w,lowh,rh,need_weekly=False,q=None,sema=None):
    print(f"starting histogram plot '{w}'")
    df,Target = get1WideTable(w,lowh,rh,need_weekly)
    df = getMatchDataTable(w,df)
    # In case simulation and observation data are not overlapped, df is empty
    if len(df)==0:
        print(f"No data for '{w}'!")
        return None

    fig = [plot_hydrograph(df,Target)] \
        + [plotRegression1(df,Target)] \
        + [plotResidue(df)] \
        + [plotHist(df,Target)]
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
    fig = plt.figure(figsize=(13.5, 9.75))
    sns.set_theme(style="darkgrid")
    axes = {n: fig.add_subplot(2, 2, n) for n in range(1,5)}

    sns.histplot(data=df, x=w, hue="DataOrigin", element="step", kde=True, ax=axes[1])
    axes[1].set_title(f"Compare Histograms of '{w}'")
    axes[2].set_ylabel("Count", fontsize=AX_LABEL_FONTSIZE)
    
    temp = df.iloc[range(len(df))]
    temp[w] = Target
    temp['DataOrigin'] = 'Target'
    temp = pd.concat([df,temp])
    li = sns.ecdfplot(data=temp, x=w, hue="DataOrigin", ax=axes[2])
    # sns.lineplot(x=[Target,Target], y=[0,1], ax=axes[2],
    #     linestyle='--', color='black', linewidth=1) #, label='Target')
    axes[2].set_title(f"Compare CDF of '{w}'")
    axes[2].set_ylabel("Probability", fontsize=AX_LABEL_FONTSIZE)

    sns.boxplot(data=df, x=w, hue="DataOrigin", ax=axes[3])
    for patch in axes[3].patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))
    # sns.despine(offset=10, trim=True, ax=axes[0, 1])
    axes[3].set_title(f"Boxplot of '{w}'")

    sns.violinplot(data=df, x=w, hue="DataOrigin", ax=axes[4], split=True, inner="quart", fill=False)
    axes[4].set_title(f"Violin Plot of Histograms for '{w}'")
    axes[4].set_ylabel("Probability", fontsize=AX_LABEL_FONTSIZE)

    for k in range(1,5):
        axes[k].grid(color='lightgray')
        axes[k].set_xlabel("Waterlevel ft NGVD", fontsize=AX_LABEL_FONTSIZE)
        axes[k].tick_params(axis='both', which='major', labelsize=AX_LABEL_FONTSIZE)
        legend = axes[k].get_legend()
        legend.get_title().set_fontsize(9)
        for text in legend.get_texts():
            text.set_fontsize(8)

    plt.tight_layout()

    proj_dir = os.path.dirname(__file__)
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-hist')
    plt.savefig(svfilePath, dpi=300, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    plt.savefig(svfilePath+'.pdf', orientation="landscape"
        , dpi=300, bbox_inches="tight", pad_inches=1, facecolor='auto', edgecolor='auto')

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

    proj_dir = os.path.dirname(__file__)
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
    g = sns.JointGrid(data=tempDF, x="Simulated", y="Observed", height=9)
    g.plot_joint(sns.kdeplot, fill=True, levels=6)
    g.plot_joint(sns.regplot, robust=True, ci=90,
        line_kws={"color":"red"},
        scatter_kws={"edgecolor": "white","linewidths": 0.25,"s": 40})
    g.plot_marginals(sns.histplot, kde=True)
    g.set_axis_labels("Simulated WL, ft NGVD", "Observed WL, ft NGVD")
    xlim = [round(min(df[w])/2.)*2-2, round(max(df[w])/2.)*2+2]
    g.ax_joint.set_xlim(xlim[0],xlim[1])
    g.ax_joint.plot(xlim, [Target, Target], color='green', linewidth=1, linestyle="--")
    g.ax_joint.grid(color='lightgray')
    g.ax_marg_x.grid(color='lightgray')
    g.ax_marg_y.grid(color='lightgray')
    g.ax_joint.tick_params(axis='both', labelsize=(AX_LABEL_FONTSIZE-1))
    g.ax_joint.minorticks_on()
    g.ax_joint.tick_params(axis='both', which='minor', length=4, color='gray')
    g.figure.subplots_adjust(top=0.95)

    model,yp = regRobust(tempDF.Simulated,tempDF.Observed)
    intercept = model.params['const']
    slope = model.params['Simulated']
    g.ax_joint.text(.98, .98
        , "Robust Linear Regression:" + f"\ny = {intercept:.3f}{slope:+.3f}x"
        , va='top', ha='right', transform=g.ax_joint.transAxes
        , fontsize=TITLE_FONTSIZE, bbox=dict(facecolor='white', alpha=1), ma='left')

    g.figure.suptitle(w, weight='bold', size=FIG_TITLE_FONTSIZE)
    g.figure.subplots_adjust(top=0.95)

    fig1 = plt.gcf()
    fig1.tight_layout()

    # two periods - validation and verification periods
    # use robust regression to predict separate period
    x_cal = df.loc[(df.ModelPeriod=='Calibration' ) & (df.DataOrigin=='Simulated'),w]
    x_ver = df.loc[(df.ModelPeriod=='Others') & (df.DataOrigin=='Simulated'),w]
    y_cal = df.loc[(df.ModelPeriod=='Calibration' ) & (df.DataOrigin=='Observed' ),w]
    y_ver = df.loc[(df.ModelPeriod=='Others') & (df.DataOrigin=='Observed' ),w]
    if len(x_cal)>3:
        cal_model,yp_cal = regRobust(x_cal,y_cal)
        cal_intercept = cal_model.params['const']
        cal_slope = cal_model.params[w]
    if len(x_ver)>3:
        ver_model,yp_ver = regRobust(x_ver,y_ver)
        ver_intercept = ver_model.params['const']
        ver_slope = ver_model.params[w]

    # Initialize JointGrid
    g = sns.JointGrid(data=tempDF, x="Simulated", y="Observed", height=9
        , hue="ModelPeriod", hue_order=['Calibration','Others'])
    g.plot_joint(sns.kdeplot, levels=6)
    g.plot_joint(sns.scatterplot, edgecolor="white" ,linewidths=0.25 ,s=20)
    g.plot_marginals(sns.histplot, kde=True)
    g.ax_joint.set_xlim(xlim[0],xlim[1])
    if len(x_cal)>3:
        g.ax_joint.plot(x_cal, yp_cal, color='blue', linewidth=3)
    if len(x_ver)>3:
        g.ax_joint.plot(x_ver, yp_ver, color='red', linewidth=3)
    g.set_axis_labels("Simulated WL, ft NGVD", "Observed WL, ft NGVD")
    g.ax_joint.plot(xlim, [Target, Target], color='green', linewidth=1, linestyle="--")
    g.ax_joint.grid(color='lightgray')
    g.ax_marg_x.grid(color='lightgray')
    g.ax_marg_y.grid(color='lightgray')
    g.ax_joint.tick_params(axis='both', labelsize=(AX_LABEL_FONTSIZE-1))
    g.ax_joint.minorticks_on()
    g.ax_joint.tick_params(axis='both', which='minor', length=4, color='gray')
    g.figure.suptitle(w, weight='bold', size=FIG_TITLE_FONTSIZE)
    g.figure.subplots_adjust(top=0.95)
    sns.move_legend(g.ax_joint, 'upper left')

    t0 = "Robust Linear Regression:"
    try:
        t1 = f"\nCalibration: y = {cal_intercept:.3f}{cal_slope:+.3f}x"
    except UnboundLocalError:
        print(f"\033[91mEmpty Calibration: {w}\033[0m", file=sys.stderr)
        t1 = ""
        cal_intercept = ""
        cal_slope = ""
    try:
        t2 = f"\n     Others: y = {ver_intercept:.3f}{ver_slope:+.3f}x"
    except UnboundLocalError:
        print(f"\033[91mEmpty Verification: {w}\033[0m", file=sys.stderr)
        t2 = ""
        ver_intercept = ""
        ver_slope = ""
    g.ax_joint.text(.98, .98, t0+t1+t2, va='top', ha='right', transform=g.ax_joint.transAxes
        , fontsize=TITLE_FONTSIZE, bbox=dict(facecolor='white', alpha=1), ma='left')

    plt.tight_layout()
    fig2 = plt.gcf()

    proj_dir = os.path.dirname(__file__)
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-regress1')
    fig1.savefig(svfilePath, dpi=300, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    fig1.savefig(svfilePath+'.pdf'
        , dpi=300, bbox_inches="tight", pad_inches=1, facecolor='auto', edgecolor='auto')
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-regress2')
    fig2.savefig(svfilePath, dpi=300, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    fig2.savefig(svfilePath+'.pdf'
        , dpi=300, bbox_inches="tight", pad_inches=1, facecolor='auto', edgecolor='auto')

    with open(os.path.join(proj_dir,FILE_REGRESSION_PARAMS), "a") as f:
        f.write(f'{w},{cal_slope},{cal_intercept},{ver_slope},{ver_intercept},{slope},{intercept}\n')

    return [fig1, fig2]

def plot_hydrograph(df,Target):
    w = df.columns.tolist()[0]
    if len(df)==0:
        print(f"No data for '{w}'!")
        return None
    fig = plt.figure(figsize=(13.5, 9.75))
    sns.set_theme(style="darkgrid")
    
    temp = df.iloc[[0,-1],:].copy()
    temp[w] = Target
    temp.DataOrigin = 'Target'
    df = pd.concat([df,temp])

    sns.scatterplot(data=df, x='Date', y=w, markers=False, s=0)
    sns.lineplot(data=df, x='Date', y=w, hue='DataOrigin',
        style='DataOrigin', dashes=[(1,0),(1,0),(3,1)])

    plt.grid(True, color='lightgray')
    plt.ylabel("Waterlevel, ft NGVD")
    plt.title(w)
    plt.tight_layout()

    proj_dir = os.path.dirname(__file__)
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-hydrograph')
    plt.savefig(svfilePath, dpi=300, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    plt.savefig(svfilePath+'.pdf', orientation="landscape"
        , dpi=300, bbox_inches="tight", pad_inches=1, facecolor='auto', edgecolor='auto')

    return fig

def plotResidue(df):
    w = df.columns.tolist()[0]
    if len(df)==0:
        print(f"No data for '{w}'!")
        return None
    tempDF = pd.pivot_table(df, values=w, index='Date', columns='DataOrigin', aggfunc='max')
    tempDF['ModelPeriod'] = df.ModelPeriod[df.DataOrigin=='Simulated'].to_list()
    tempDF['Residue'] = tempDF['Observed']-tempDF['Simulated']

    x_cal = tempDF.loc[tempDF.ModelPeriod=='Calibration' ,'Simulated']
    x_ver = tempDF.loc[tempDF.ModelPeriod=='Others' ,'Simulated']
    y_cal = tempDF.loc[tempDF.ModelPeriod=='Calibration' ,'Residue']
    y_ver = tempDF.loc[tempDF.ModelPeriod=='Others' ,'Residue']
    if len(x_cal)>3:
        cal_model,yp_cal = regRobust(x_cal,y_cal)
        cal_intercept = cal_model.params['const']
        cal_slope = cal_model.params['Simulated']
    if len(x_ver)>3:
        ver_model,yp_ver = regRobust(x_ver,y_ver)
        ver_intercept = ver_model.params['const']
        ver_slope = ver_model.params['Simulated']

    # Initialize JointGrid
    g = sns.JointGrid(data=tempDF, x="Simulated", y="Residue", height=9
        , hue="ModelPeriod", hue_order=['Calibration','Others'])
    # g.plot_joint(sns.kdeplot, levels=6)
    g.plot_joint(sns.scatterplot, edgecolor="white" ,linewidths=0.25 ,s=20)
    g.plot_marginals(sns.histplot, kde=True)
    if len(x_cal)>3:
        g.ax_joint.plot(x_cal, yp_cal, color='blue', linewidth=3)
    if len(x_ver)>3:
        g.ax_joint.plot(x_ver, yp_ver, color='red', linewidth=3)
    g.set_axis_labels("Simulated WL, ft NGVD", "Residue, ft")
    g.ax_joint.grid(color='lightgray')
    g.ax_marg_x.grid(color='lightgray')
    g.ax_marg_y.grid(color='lightgray')

    g.figure.suptitle(w, weight='bold', size=FIG_TITLE_FONTSIZE)
    g.figure.subplots_adjust(top=0.95)
    sns.move_legend(g.ax_joint, 'upper left')

    rmse = np.sqrt(np.mean(tempDF['Residue'] ** 2))
    t0 = "Robust Linear Regression:"
    try:
        t1 = f"\nCalibration: y = {cal_intercept:.3f}{cal_slope:+.3f}x"
        rmse_cal = np.sqrt(np.mean(tempDF.loc[tempDF.ModelPeriod=='Calibration', 'Residue'] ** 2))
    except UnboundLocalError:
        # print(f"\033[91mEmpty Calibration: {w}\033[0m", file=sys.stderr)
        t1 = ""
        rmse_cal = float('nan')
    try:
        t2 = f"\n     Others: y = {ver_intercept:.3f}{ver_slope:+.3f}x"
        rmse_ver = np.sqrt(np.mean(tempDF.loc[tempDF.ModelPeriod=='Others' , 'Residue'] ** 2))
    except UnboundLocalError:
        # print(f"\033[91mEmpty Verification: {w}\033[0m", file=sys.stderr)
        t2 = ""
        rmse_ver = float('nan')
    t3 = f"\nRMSE[Cal,Ver,All] = [{rmse_cal:.2f}, {rmse_ver:.2f}, {rmse:.2f}]"
    g.ax_joint.text(.98, .98, t0+t1+t2+t3, va='top', ha='right', transform=g.ax_joint.transAxes
        , fontsize=TITLE_FONTSIZE, bbox=dict(facecolor='white', alpha=1), ma='left')

    plt.tight_layout()
    fig = plt.gcf()

    proj_dir = os.path.dirname(__file__)
    svfilePath = os.path.join(proj_dir,'plotWarehouse',f'{w}-residue')
    fig.savefig(svfilePath, dpi=300, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    fig.savefig(svfilePath+'.pdf'
        , dpi=300, bbox_inches="tight", pad_inches=1, facecolor='auto', edgecolor='auto')

    return fig

def get1WideTable(w,lowh,rh,need_weekly):
    owinfo = lowh.owinfo_df.iloc[[i for i,targ in enumerate(lowh.owinfo_df.Target) if not np.isnan(targ)]]
    LS2Topo = owinfo.SurfEl2CellTopoOffset[owinfo.PointName==w].iloc[0]
    # Wide format table
    obs_ts = lowh.loadHead([w],need_weekly=need_weekly)
    obs_ts.rename(columns={w:f'obs_{w}'},inplace=True)
    
    sim_ts = rh.getHeadByWellnames([w],need_weekly=need_weekly)
    sim_ts[w] += LS2Topo # add cell offset
    sim_ts.rename(columns={w:f'sim_{w}'},inplace=True)

    df = sim_ts.join(obs_ts)
    del sim_ts, obs_ts

    df.index.name = 'Date'
    df['ModelPeriod'] = 'Others'
    df = df.loc[POA[0]:POA[1]]
    df.loc[CAL_PERIOD[0]:CAL_PERIOD[1], 'ModelPeriod'] = 'Calibration'
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
    owinfo = lowh.owinfo_df.iloc[[i for i,targ in enumerate(lowh.owinfo_df.Target) if not np.isnan(targ)]]
    LS2Topo = owinfo.SurfEl2CellTopoOffset[owinfo.PointName==w].iloc[0]

    # Column table
    obs_ts = lowh.loadHead([w],date_index=True,need_weekly=need_weekly)
    obs_ts['DataOrigin'] = 'Observed'

    sim_ts = rh.getHeadByWellnames([w],date_index=True,need_weekly=need_weekly)
    sim_ts[[w]] += LS2Topo # add cell offset
    sim_ts['DataOrigin'] = 'Simulated'

    return pd.concat([sim_ts,obs_ts])

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

# def useMP_Queue(wnames,lowh,rh):
def useMP_Queue():
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

def useMP_Pool():
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

def use_noMP():
    return [plot_MP(w,lowh,rh,need_weekly) for w in wnames]

def move_result():
    import shutil
    # zip plotWarehouse directory (holding graphic images and pdf files) and move to the result directory
    prefix = os.path.basename(__file__).split('_')[0]
    result_dir = os.path.join(os.path.dirname(proj_dir),f'INTB{INTB_VERSION}_EDA results')
    shutil.make_archive(
        os.path.join(result_dir,f'{prefix}_plotWarehouse')
        , 'zip', os.path.join(proj_dir,'plotWarehouse')
    )

    # move csv file
    filename = f'{prefix}_{FILE_REGRESSION_PARAMS}'
    shutil.move(os.path.join(proj_dir, FILE_REGRESSION_PARAMS), os.path.join(result_dir, filename))

    # move merged pdf file
    filename = 'all_plots.pdf'
    shutil.move(os.path.join(proj_dir, filename), os.path.join(result_dir, f'{prefix}_{filename}'))


if __name__ == '__main__':
    from image2pdf import merge_pdf
    # from memory_profiler import memory_usage

    need_weekly = True
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({'font.size': 8, 'savefig.dpi': 300})

    proj_dir = os.path.dirname(__file__)
    run_dir  = os.path.join(os.path.dirname(proj_dir),RUN_DIRNAME)

    # Perform EDA for OROP wells
    lowh = LoadObservedHead(run_dir)
    rh = ReadHead(run_dir)
    owinfo = lowh.owinfo_df.iloc[[i for i,targ in enumerate(lowh.owinfo_df.Target) if not np.isnan(targ)]]
    wnames = owinfo.PointName.to_list()
    wnames = [w for w in wnames if w not in ['RMP-8D1']]

    # sortedDF = owinfo.sort_values(by=['TargetType','WFCode','PointName'])[['PointName']]
    # merge_pdf(proj_dir,sortedDF)
    # move_result()

    if IS_DEBUGGING:
        # use_mp = use_noMP | useMP_Queue | useMP_Pool
        use_mp = 'use_noMP '
        wnames = ['ROMP-8D','RMP-8D1']
        wnames = ['CYC-W56B','TMR-2s','MB-24s']
        wnames = ['Calm-33A','TMR-1As','TMR-4s']
    else:
        use_mp = 'useMP_Queue'
        plotWarehouse = os.path.join(proj_dir,'plotWarehouse')
        for f in os.listdir(plotWarehouse):
            try:
                os.remove(os.path.join(plotWarehouse,f))
            except OSError as err:
                print(err)

    with open(os.path.join(proj_dir,FILE_REGRESSION_PARAMS), "w") as f:
        f.write(f"PointName,Cal_Slope,Cal_Intercept,Ver_Slope,Ver_Intercept,Slope,Intercept\n") 

    start_time = datetime.now()
    if use_mp=='useMP_Queue':
        rtnval = useMP_Queue()
    elif use_mp=='useMP_Pool':
        rtnval = useMP_Pool()
    else:
        rtnval = use_noMP()
        # mem_usage = memory_usage(use_noMP)

    etime = datetime.now()-start_time
    print(f'Elasped time: {etime}')

    # print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    # print('Maximum memory usage: %s' % max(mem_usage))

    if IS_DEBUGGING:
        plt.show()
    else:
        # merge pdf files
        sortedDF = owinfo.sort_values(by=['TargetType','WFCode','PointName'])[['PointName']]
        merge_pdf(proj_dir,sortedDF)
        move_result()

    exit(0)
