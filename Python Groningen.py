# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:10:44 2019
@author: Eelke Snoeren
Python protocol for Indrek's data from Groningen
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain
sns.set()
from PIL import Image
import glob
import os
from matplotlib.backends.backend_pdf import PdfPages

# Say where to save the file and with what name (note the use of / instead of \)
out_path = "H:/Werk/Experimenten/Indrek Groningen/Output/results_observer.xlsx"

# Load all excel files in the folder into one DataFrame and clean up
dataraw = pd.DataFrame()
for f in glob.glob("*.xlsx"):
    df = pd.read_excel(f)
    dataraw = dataraw.append(df,ignore_index=True)

dataraw = dataraw.dropna(axis=0, how='all')
dataraw = dataraw.reset_index()

# Take a look at your data
#dataraw.shape
#dataraw.head()
#dataraw.tail()

# Fill out your short column names behind the definition a-z
A='Unimportant'
B='Index_unimportant'
C='Unimportant'
D='Unimportant'
E='Unimportant'
F='Unimportant'
G='Unimportant'
H='Time_raw'
I='Unimportant'
J='Unimportant'
K='Unimportant'
L='Observation'
M='Unimportant'
N='Behavior_raw'
O='Unknown'
P='Unknown'
Q='Unknown'
R='Unknown'
S='Unknown'
T='Experiment'
U='RatID'
V='Time'
W='NOTSET'
X='RatID_OBS'
Y='Behavior'
Z='Treat_mod'

# Fill out your treatment/stimulus behind definition SA-SZ
SA='Stimulus1'
SB='Stimulus2'
SC='Stimulus3'
SD='Stimulus4'
SE='Stimulus5'
SF='Stimulus6' 

Stimuli_values= (SA,SB,SC,SD)

Timetest=1800 

# Fill out your behavioral observations behind definition BA-BZ
BA='Start rat 1'
BB='Start rat 2'
BC='Start rat 3'
BD='Start rat 4'
BE='Start rat 5'
BF='Start rat 6'
BG='Mount1'
BH='Mount2'
BI='Mount3'
BJ='Mount4'
BK='Mount5'
BL='Mount6'
BM='Intromission1'
BN='Intromission2'
BO='Intromission3'
BP='Intromission4'
BQ='Intromission5'
BR='Intromission6'
BS='Ejaculation1'
BT='Ejaculation2'
BU='Ejaculation3'
BV='Ejaculation4'
BW='Ejaculation5'
BX='Ejaculation6'

# Rename columns (add or remove letters according to number of columns)
dataraw.columns = [A,B,C,D,E,F,G,I,H,J,K,L,M,N,O,P]


# Make a new datafile with selected columns
data=dataraw[[L,H,N]]

# Make RatID
RatID =pd.DataFrame(np.where(data[N]=='Start rat 1', 'R1',"X"), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Start rat 2', 'R2',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Start rat 3', 'R3',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Start rat 4', 'R4',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Start rat 5', 'R5',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Start rat 6', 'R6',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Mount1', 'R1',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Mount2', 'R2',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Mount3', 'R3',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Mount4', 'R4',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Mount5', 'R5',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Mount6', 'R6',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Intromission1', 'R1',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Intromission2', 'R2',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Intromission3', 'R3',RatID[0]), copy = True) 
RatID =pd.DataFrame(np.where(data[N]=='Intromission4', 'R4',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Intromission5', 'R5',RatID[0]), copy = True) 
RatID =pd.DataFrame(np.where(data[N]=='Intromission6', 'R6',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Ejaculation1', 'R1',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Ejaculation2', 'R2',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Ejaculation3', 'R3',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Ejaculation4', 'R4',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Ejaculation5', 'R5',RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(data[N]=='Ejaculation6', 'R6',RatID[0]), copy = True)

# Rename the behaviors into Mount, Intro, and Ejac (instead of the numbering)
Behavior2 =pd.DataFrame(np.where(data[N]=='Start rat 1', 'Start',""), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Start rat 2', 'Start',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Start rat 3', 'Start',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Start rat 4', 'Start',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Start rat 5', 'Start',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Start rat 6', 'Start',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Mount1', 'Mount',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Mount2', 'Mount',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Mount3', 'Mount',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Mount4', 'Mount',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Mount5', 'Mount',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Mount6', 'Mount',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Intromission1', 'Intro',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Intromission2', 'Intro',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Intromission3', 'Intro',Behavior2[0]), copy = True) 
Behavior2 =pd.DataFrame(np.where(data[N]=='Intromission4', 'Intro',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Intromission5', 'Intro',Behavior2[0]), copy = True) 
Behavior2 =pd.DataFrame(np.where(data[N]=='Intromission6', 'Intro',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Ejaculation1', 'Ejac',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Ejaculation2', 'Ejac',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Ejaculation3', 'Ejac',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Ejaculation4', 'Ejac',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Ejaculation5', 'Ejac',Behavior2[0]), copy = True)
Behavior2 =pd.DataFrame(np.where(data[N]=='Ejaculation6', 'Ejac',Behavior2[0]), copy = True)


# Add the dataframes to data
data = pd.concat([data, RatID, Behavior2], sort=False, axis=1)

# Assign column-names
data.columns = [L,H,N,U,Y]

# Select columns and set index to ratID and treatment in final data
data_noindex= data[[L,U,H,Y]]

# Make column with uniquerat-observation-code
data_noindex[X] = data_noindex[U].map(str) + data_noindex[L]

# Number the behaviors on occurance
data_noindex['obs_num'] = data_noindex.groupby(X)[Y].transform(lambda x: np.arange(1, len(x) + 1))

writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
data_noindex.to_excel(writer,sheet_name='data_raw')
writer.save()

# Make a new column that makes an unique name for the behaviors per rat
data_noindex['beh_num_trick'] = data_noindex[X].map(str) + data_noindex[Y]

# Number the behaviors per behavior per rat
data_noindex['beh_num'] = data_noindex.groupby('beh_num_trick')[Y].transform(lambda x: np.arange(1, len(x) + 1))

# Number the behaviors backwards
data_noindex = data_noindex.sort_values(by=[X,H], ascending = False)
data_noindex['obs_num_back'] = data_noindex.groupby(X)[Y].transform(lambda x: np.arange(1, len(x) + 1))
data_noindex['beh_num_back'] = data_noindex.groupby('beh_num_trick')[Y].transform(lambda x: np.arange(1, len(x) + 1))

# Check whether a rat has mounts/intromissions/ejaculations
data_noindex['mount_check'] = np.where(data_noindex['obs_num_back']==1,99999,np.NaN)
data_noindex['mount_check'] = np.where(np.logical_and(data_noindex['beh_num']==1, data_noindex[Y]=='Mount'),1,
            data_noindex['mount_check'])
data_noindex['mount_check'].fillna(method="ffill", inplace = True)
data_noindex['mount_check2'] = np.where(np.logical_and(data_noindex['obs_num']==1, data_noindex['mount_check']==1),1,np.NaN)
data_noindex['mount_check2'] = np.where(np.logical_and(data_noindex['obs_num']==1, data_noindex['mount_check']==99999),2,
            data_noindex['mount_check2'])
data_noindex['mount_check2'].fillna(method="backfill", inplace = True)

data_noindex['intro_check'] = np.where(data_noindex['obs_num_back']==1,99999,np.NaN)
data_noindex['intro_check'] = np.where(np.logical_and(data_noindex['beh_num']==1, data_noindex[Y]=='Intro'),1,
            data_noindex['intro_check'])
data_noindex['intro_check'].fillna(method="ffill", inplace = True)
data_noindex['intro_check2'] = np.where(np.logical_and(data_noindex['obs_num']==1, data_noindex['intro_check']==1),1,np.NaN)
data_noindex['intro_check2'] = np.where(np.logical_and(data_noindex['obs_num']==1, data_noindex['intro_check']==99999),2,
            data_noindex['intro_check2'])
data_noindex['intro_check2'].fillna(method="backfill", inplace = True)

data_noindex['ejac_check'] = np.where(data_noindex['obs_num_back']==1,99999,np.NaN)
data_noindex['ejac_check'] = np.where(np.logical_and(data_noindex['beh_num']==1, data_noindex[Y]=='Ejac'),1,
            data_noindex['ejac_check'])
data_noindex['ejac_check'].fillna(method="ffill", inplace = True)
data_noindex['ejac_check2'] = np.where(np.logical_and(data_noindex['obs_num']==1, data_noindex['ejac_check']==1),1,np.NaN)
data_noindex['ejac_check2'] = np.where(np.logical_and(data_noindex['obs_num']==1, data_noindex['ejac_check']==99999),2,
            data_noindex['ejac_check2'])
data_noindex['ejac_check2'].fillna(method="backfill", inplace = True)

data_noindex = data_noindex.sort_values(by=[X,H])

# Make column with start times 
data_noindex['Starttime']=np.where(data_noindex[Y]=='Start', data_noindex['beh_num'], np.NaN)
data_noindex['Starttime']=np.where(data_noindex['Starttime']==1, data_noindex[H], np.NaN)
    # mark situation where it does not start with start to later fill in starttime in column 
data_noindex['Starttime']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Start'),0,data_noindex['Starttime']) 
data_noindex['Starttime'].fillna(method = "ffill", inplace=True)
    # now we empty the 99999 rows and fill it with the right numbers
data_noindex['Starttime']=np.where(data_noindex['Starttime']==99999, np.NaN, data_noindex['Starttime'])
data_noindex['Starttime'].fillna(method = "backfill", inplace=True)

# Change the time column with the real time from start
data_noindex[V] = data_noindex[H]-data_noindex['Starttime']

# Get the Latency to 1st Mount, Intromission and, Ejaculation
data_noindex['L1MIE'] =np.where((data_noindex['beh_num']==1) & (data_noindex[Y]=='Mount'),
            data_noindex[V],np.NaN)
data_noindex['L1MIE'] =np.where((data_noindex['beh_num']==1) & (data_noindex[Y]=='Intro'),
            data_noindex[V],data_noindex['L1MIE'])
data_noindex['L1MIE'] =np.where((data_noindex['beh_num']==1) & (data_noindex[Y]=='Ejac'),
            data_noindex[V],data_noindex['L1MIE'])
data_noindex['L1MIE'].fillna(method = "ffill", inplace=True)

# Make column with latency to 1st mount
data_noindex['L1M']=np.where(data_noindex[Y]=='Mount', data_noindex['beh_num'], np.NaN)
data_noindex['L1M']=np.where(data_noindex['L1M']==1, data_noindex[V], np.NaN)
    # mark situation where it does not start with mounts to later fill in mount latency in column 
data_noindex['L1M']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Mount'),99999,data_noindex['L1M']) 
data_noindex['L1M'].fillna(method = "ffill", inplace=True)
    # now we empty the 99999 rows and fill it with the right numbers
data_noindex['L1M']=np.where(data_noindex['L1M']==99999, np.NaN, data_noindex['L1M'])
data_noindex['L1M'].fillna(method = "backfill", inplace=True)
    # Take out data points where no mounts have occured
data_noindex['L1M']=np.where(data_noindex['mount_check2']==2, Timetest, data_noindex['L1M'])

# Make column with latency to 1st intromission
data_noindex['L1I']=np.where(data_noindex[Y]=='Intro', data_noindex['beh_num'], np.NaN)
data_noindex['L1I']=np.where(data_noindex['L1I']==1, data_noindex[V], np.NaN)
data_noindex['L1I']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Intro'),99999,data_noindex['L1I']) 
data_noindex['L1I'].fillna(method = "ffill", inplace=True)
data_noindex['L1I']=np.where(data_noindex['L1I']==99999, np.NaN, data_noindex['L1I'])
data_noindex['L1I'].fillna(method = "backfill", inplace=True)
data_noindex['L1I']=np.where(data_noindex['intro_check2']==2, Timetest, data_noindex['L1I'])

# Get the latency of the 1st behavior 
data_noindex['L1B'] =np.where(data_noindex['L1M']>data_noindex['L1I'],data_noindex['L1I'],data_noindex['L1M'])

# Make column with total number of mounts
data_noindex['TM']=np.where(data_noindex[Y]=='Mount', data_noindex['beh_num'], np.NaN)
data_noindex['TM']=np.where(data_noindex['TM']==1, data_noindex['beh_num_back'], np.NaN)
data_noindex['TM']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Mount'),99999,data_noindex['TM']) 
data_noindex['TM'].fillna(method = "ffill", inplace=True)
data_noindex['TM']=np.where(data_noindex['TM']==99999, np.NaN, data_noindex['TM'])
data_noindex['TM'].fillna(method = "backfill", inplace=True)
data_noindex['TM']=np.where(data_noindex['mount_check2']==2, 0, data_noindex['TM'])

# Make column with total number of intromissions
data_noindex['TI']=np.where(data_noindex[Y]=='Intro', data_noindex['beh_num'], np.NaN)
data_noindex['TI']=np.where(data_noindex['TI']==1, data_noindex['beh_num_back'], np.NaN)
data_noindex['TI']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Intro'),99999,data_noindex['TI']) 
data_noindex['TI'].fillna(method = "ffill", inplace=True)
data_noindex['TI']=np.where(data_noindex['TI']==99999, np.NaN, data_noindex['TI'])
data_noindex['TI'].fillna(method = "backfill", inplace=True)
data_noindex['TI']=np.where(data_noindex['intro_check2']==2, 0, data_noindex['TI'])

# Make column with total number of ejaculations
data_noindex['TE']=np.where(data_noindex[Y]=='Ejac', data_noindex['beh_num'], np.NaN)
data_noindex['TE']=np.where(data_noindex['TE']==1, data_noindex['beh_num_back'], np.NaN)
data_noindex['TE']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Ejac'),99999,data_noindex['TE']) 
data_noindex['TE'].fillna(method = "ffill", inplace=True)
data_noindex['TE']=np.where(data_noindex['TE']==99999, np.NaN, data_noindex['TE'])
data_noindex['TE'].fillna(method = "backfill", inplace=True)
data_noindex['TE']=np.where(data_noindex['ejac_check2']==2, 0, data_noindex['TE'])

# Calculate and make column with total copulatory behavior (Mount+intromissions+ejaculations)
data_noindex['TCB']=(data_noindex['TM']+data_noindex['TI']+data_noindex['TE'])

# Calculate and make column with total intromissions ratio (intromissions/intromissions+mounts)
data_noindex['TIR']=(data_noindex['TI']/(data_noindex['TI']+data_noindex['TM']))

# Calculate and make column with total intromissions interval (total time test / intromissions)
data_noindex['TIII']=Timetest/data_noindex['TI']

# Calculate and make column with total copulatory rate (number of mounts and intromissions/ time from first behavior to end)
data_noindex['TCR']=(data_noindex['TM']+data_noindex['TI'])/(Timetest-data_noindex['L1B'])

# Number the ejaculation series per rat (with 10 as last series)
data_noindex['ejac_serie']=np.where(data_noindex[Y]=='Ejac', data_noindex['beh_num'],np.NaN)
data_noindex['ejac_serie']=np.where(data_noindex['obs_num']==1, 10, data_noindex['ejac_serie'])

# Calculate latency to ejaculation from 1st behavior and from 1st intromission
data_noindex['S1LEB']=np.where(data_noindex['ejac_serie']== 1, data_noindex['L1MIE']-data_noindex['L1B'], np.NaN)
data_noindex['S1LEB']=np.where(data_noindex['obs_num']==1,99999,data_noindex['S1LEB']) 
data_noindex['S1LEB'].fillna(method = "ffill", inplace=True)
data_noindex['S1LEB']=np.where(data_noindex['S1LEB']==99999, np.NaN, data_noindex['S1LEB'])
data_noindex['S1LEB'].fillna(method = "backfill", inplace=True)
data_noindex['S1LEB']=np.where(data_noindex['ejac_check2']==2, Timetest, data_noindex['S1LEB'])

data_noindex['S1LEI']=np.where(data_noindex['ejac_serie']== 1, data_noindex['L1MIE']-data_noindex['L1I'], np.NaN)
data_noindex['S1LEI']=np.where(data_noindex['obs_num']==1,99999,data_noindex['S1LEI']) 
data_noindex['S1LEI'].fillna(method = "ffill", inplace=True)
data_noindex['S1LEI']=np.where(data_noindex['S1LEI']==99999, np.NaN, data_noindex['S1LEI'])
data_noindex['S1LEI'].fillna(method = "backfill", inplace=True)
data_noindex['S1LEI']=np.where(data_noindex['ejac_check2']==2, Timetest, data_noindex['S1LEI'])
data_noindex['S1LEI']=np.where(data_noindex['intro_check2']==2, Timetest, data_noindex['S1LEI'])

# Make the first behavior belong to 1st ejaculatory series
data_noindex['ejac_serie_filled']=np.where(data_noindex[Y]=='Ejac', data_noindex['beh_num'],np.NaN)
data_noindex['ejac_serie_filled']=np.where(data_noindex['obs_num']==1, 10, data_noindex['ejac_serie_filled'])
data_noindex['ejac_serie_filled'].fillna(method = "backfill", inplace=True)
data_noindex['ejac_serie_filled']=np.where((data_noindex['ejac_serie_filled'] + data_noindex['obs_num']==11), 1, 
            data_noindex['ejac_serie_filled'])
data_noindex['ejac_serie_filled'].fillna(method = "ffill", inplace=True)


# Make a new column that makes an unique name for the behaviors per rat
data_noindex['beh_num_trick_serie'] = data_noindex[X].map(str) + data_noindex[Y] + data_noindex['ejac_serie_filled'].map(str)

# Number the behaviors per behavior per rat
data_noindex['beh_num_serie'] = data_noindex.groupby('beh_num_trick_serie')[Y].transform(lambda x: np.arange(1, len(x) + 1))

# Number the behaviors backwards
data_noindex = data_noindex.sort_values(by=[X,V], ascending = False)
data_noindex['beh_num_back_serie'] = data_noindex.groupby('beh_num_trick_serie')[Y].transform(lambda x: np.arange(1, len(x) + 1))

data_noindex = data_noindex.sort_values(by=[X,V])

# Make column with number of mounts in 1st ejaculatory series 
data_noindex['S1M']=np.where(data_noindex[Y]=='Mount', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S1M']=np.where(np.logical_and(data_noindex['S1M']==1, data_noindex['ejac_serie_filled']==1), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S1M']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Mount'),99999,
            data_noindex['S1M']) 
data_noindex['S1M'].fillna(method = "ffill", inplace=True)
data_noindex['S1M']=np.where(data_noindex['S1M']==99999, np.NaN, data_noindex['S1M'])
data_noindex['S1M'].fillna(method = "backfill", inplace=True)
data_noindex['S1M']=np.where(data_noindex['mount_check2']==2, 0, data_noindex['S1M'])
data_noindex['S1M']=np.where(data_noindex['TE']==0, data_noindex['TM'], data_noindex['S1M'])

## Make column with number of intromissions in 1st ejaculatory series
data_noindex['S1I']=np.where(data_noindex[Y]=='Intro', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S1I']=np.where(np.logical_and(data_noindex['S1I']==1, data_noindex['ejac_serie_filled']==1), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S1I']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Intro'),99999,
            data_noindex['S1I']) 
data_noindex['S1I'].fillna(method = "ffill", inplace=True)
data_noindex['S1I']=np.where(data_noindex['S1I']==99999, np.NaN, data_noindex['S1I'])
data_noindex['S1I'].fillna(method = "backfill", inplace=True)
data_noindex['S1I']=np.where(data_noindex['intro_check2']==2, 0, data_noindex['S1I'])
data_noindex['S1I']=np.where(data_noindex['TE']==0, data_noindex['TI'], data_noindex['S1I'])

# Make a check column for the moments there are no mounts and intromissions left for the 2nd ejaculatory series 
data_noindex['mount_check_serie2']=np.where((data_noindex['TM']-data_noindex['S1M']==0), 2, 1)
data_noindex['mount_check_serie2']=np.where(data_noindex['TE']<1, 2, data_noindex['mount_check_serie2'])
data_noindex['intro_check_serie2']=np.where((data_noindex['TI']-data_noindex['S1I']==0), 2, 1)
data_noindex['intro_check_serie2']=np.where(data_noindex['TE']<1, 2, data_noindex['intro_check_serie2'])

# Calculate and make column with copulatory behavior (Mount+intromissions+ejaculations) in 1st ejaculatory series
data_noindex['S1CB']=(data_noindex['S1M']+data_noindex['S1I'])

# Calculate and make column with intromissions ratio (intromissions/intromissions+mounts) in 1st ejaculatory series
data_noindex['S1IR']=(data_noindex['S1I']/(data_noindex['S1I']+data_noindex['S1M']))

# Calculate and make column with intromissions interval (latency to ejacualation / intromissions) in 1st ejaculatory series
data_noindex['S1III']=data_noindex['S1LEB']/data_noindex['S1I']

# Calculate and make column with copulatory rate (number of mounts and intromissions/ latency to ejaculation from 1st behavior) 
# in 1st ejaculatory series
data_noindex['S1CR']=(data_noindex['S1M']+data_noindex['S1I'])/(data_noindex['S1LEB'])        

# Make column with number of mounts in 2nd ejaculatory series 
data_noindex['S2M']=np.where(data_noindex[Y]=='Mount', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S2M']=np.where(np.logical_and(data_noindex['S2M']==1, data_noindex['ejac_serie_filled']==2), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S2M']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S2M']) 
data_noindex['S2M']=np.where(data_noindex['obs_num_back']==1, 77777, data_noindex['S2M'])
data_noindex['S2M']=np.where(data_noindex['TE']<2, 88888, data_noindex['S2M'])         
data_noindex['S2M'].fillna(method = "ffill", inplace=True)
data_noindex['S2M']=np.where(data_noindex['S2M']==99999, np.NaN, data_noindex['S2M'])
data_noindex['S2M'].fillna(method = "backfill", inplace=True)
data_noindex['S2M']=np.where(data_noindex['obs_num_back']==1, np.NaN, data_noindex['S2M'])
data_noindex['S2M'].fillna(method = "ffill", inplace=True)
data_noindex['S2M']=np.where(data_noindex['S2M']==88888, np.NaN, data_noindex['S2M'])
data_noindex['S2M']=np.where(data_noindex['S2M']==77777, 0, data_noindex['S2M'])

# Make column with number of intromissions in 2nd ejaculatory series
data_noindex['S2I']=np.where(data_noindex[Y]=='Intro', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S2I']=np.where(np.logical_and(data_noindex['S2I']==1, data_noindex['ejac_serie_filled']==2), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S2I']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S2I']) 
data_noindex['S2I']=np.where(data_noindex['obs_num_back']==1, 77777, data_noindex['S2I'])
data_noindex['S2I']=np.where(data_noindex['TE']<2, 88888, data_noindex['S2I'])         
data_noindex['S2I'].fillna(method = "ffill", inplace=True)
data_noindex['S2I']=np.where(data_noindex['S2I']==99999, np.NaN, data_noindex['S2I'])
data_noindex['S2I'].fillna(method = "backfill", inplace=True)
data_noindex['S2I']=np.where(data_noindex['obs_num_back']==1, np.NaN, data_noindex['S2I'])
data_noindex['S2I'].fillna(method = "ffill", inplace=True)
data_noindex['S2I']=np.where(data_noindex['S2I']==88888, np.NaN, data_noindex['S2I'])
data_noindex['S2I']=np.where(data_noindex['S2I']==77777, 0, data_noindex['S2I'])

# Make column with number of mounts in 3rd ejaculatory series 
data_noindex['S3M']=np.where(data_noindex[Y]=='Mount', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S3M']=np.where(np.logical_and(data_noindex['S3M']==1, data_noindex['ejac_serie_filled']==3), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S3M']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S3M']) 
data_noindex['S3M']=np.where(data_noindex['obs_num_back']==1, 77777, data_noindex['S3M'])
data_noindex['S3M']=np.where(data_noindex['TE']<3, 88888, data_noindex['S3M'])         
data_noindex['S3M'].fillna(method = "ffill", inplace=True)
data_noindex['S3M']=np.where(data_noindex['S3M']==99999, np.NaN, data_noindex['S3M'])
data_noindex['S3M'].fillna(method = "backfill", inplace=True)
data_noindex['S3M']=np.where(data_noindex['obs_num_back']==1, np.NaN, data_noindex['S3M'])
data_noindex['S3M'].fillna(method = "ffill", inplace=True)
data_noindex['S3M']=np.where(data_noindex['S3M']==88888, np.NaN, data_noindex['S3M'])
data_noindex['S3M']=np.where(data_noindex['S3M']==77777, 0, data_noindex['S3M'])

## Make column with number of intromissions in 3rd ejaculatory series
data_noindex['S3I']=np.where(data_noindex[Y]=='Intro', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S3I']=np.where(np.logical_and(data_noindex['S3I']==1, data_noindex['ejac_serie_filled']==3), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S3I']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S3I']) 
data_noindex['S3I']=np.where(data_noindex['obs_num_back']==1, 77777, data_noindex['S3I'])
data_noindex['S3I']=np.where(data_noindex['TE']<3, 88888, data_noindex['S3I'])         
data_noindex['S3I'].fillna(method = "ffill", inplace=True)
data_noindex['S3I']=np.where(data_noindex['S3I']==99999, np.NaN, data_noindex['S3I'])
data_noindex['S3I'].fillna(method = "backfill", inplace=True)
data_noindex['S3I']=np.where(data_noindex['obs_num_back']==1, np.NaN, data_noindex['S3I'])
data_noindex['S3I'].fillna(method = "ffill", inplace=True)
data_noindex['S3I']=np.where(data_noindex['S3I']==88888, np.NaN, data_noindex['S3I'])
data_noindex['S3I']=np.where(data_noindex['S3I']==77777, 0, data_noindex['S3I'])

# Make column with number of mounts in 4th ejaculatory series 
data_noindex['S4M']=np.where(data_noindex[Y]=='Mount', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S4M']=np.where(np.logical_and(data_noindex['S4M']==1, data_noindex['ejac_serie_filled']==4), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S4M']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S4M']) 
data_noindex['S4M']=np.where(data_noindex['obs_num_back']==1, 77777, data_noindex['S4M'])
data_noindex['S4M']=np.where(data_noindex['TE']<4, 88888, data_noindex['S4M'])         
data_noindex['S4M'].fillna(method = "ffill", inplace=True)
data_noindex['S4M']=np.where(data_noindex['S4M']==99999, np.NaN, data_noindex['S4M'])
data_noindex['S4M'].fillna(method = "backfill", inplace=True)
data_noindex['S4M']=np.where(data_noindex['obs_num_back']==1, np.NaN, data_noindex['S4M'])
data_noindex['S4M'].fillna(method = "ffill", inplace=True)
data_noindex['S4M']=np.where(data_noindex['S4M']==88888, np.NaN, data_noindex['S4M'])
data_noindex['S4M']=np.where(data_noindex['S4M']==77777, 0, data_noindex['S4M'])

## Make column with number of intromissions in 4th ejaculatory series
data_noindex['S4I']=np.where(data_noindex[Y]=='Intro', data_noindex['beh_num_serie'], np.NaN)
data_noindex['S4I']=np.where(np.logical_and(data_noindex['S4I']==1, data_noindex['ejac_serie_filled']==4), 
            data_noindex['beh_num_back_serie'], np.NaN)
data_noindex['S4I']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S4I']) 
data_noindex['S4I']=np.where(data_noindex['obs_num_back']==1, 77777, data_noindex['S4I'])
data_noindex['S4I']=np.where(data_noindex['TE']<4, 88888, data_noindex['S4I'])         
data_noindex['S4I'].fillna(method = "ffill", inplace=True)
data_noindex['S4I']=np.where(data_noindex['S4I']==99999, np.NaN, data_noindex['S4I'])
data_noindex['S4I'].fillna(method = "backfill", inplace=True)
data_noindex['S4I']=np.where(data_noindex['obs_num_back']==1, np.NaN, data_noindex['S4I'])
data_noindex['S4I'].fillna(method = "ffill", inplace=True)
data_noindex['S4I']=np.where(data_noindex['S4I']==88888, np.NaN, data_noindex['S4I'])
data_noindex['S4I']=np.where(data_noindex['S4I']==77777, 0, data_noindex['S4I'])

# Make a check column for the moments there are no mounts and intromissions left for the 3rd and 4th ejaculatory series 
data_noindex['mount_check_serie3']=np.where((data_noindex['TM']-data_noindex['S1M']-data_noindex['S2M']==0), 2, 1)
data_noindex['mount_check_serie3']=np.where(data_noindex['TE']<2, 2, data_noindex['mount_check_serie3'])
data_noindex['intro_check_serie3']=np.where((data_noindex['TI']-data_noindex['S1I']-data_noindex['S2I']==0), 2, 1)
data_noindex['intro_check_serie3']=np.where(data_noindex['TE']<2, 2, data_noindex['intro_check_serie3'])

data_noindex['mount_check_serie4']=np.where((data_noindex['TM']-data_noindex['S1M']-data_noindex['S2M']-data_noindex['S3M']==0), 2, 1)
data_noindex['mount_check_serie4']=np.where(data_noindex['TE']<3, 2, data_noindex['mount_check_serie4'])
data_noindex['intro_check_serie4']=np.where((data_noindex['TI']-data_noindex['S1I']-data_noindex['S2I']-data_noindex['S3I']==0), 2, 1)
data_noindex['intro_check_serie4']=np.where(data_noindex['TE']<3, 2, data_noindex['intro_check_serie4'])

# Make a column with the time of 1st mount in ejaculatory series 10 - the left overs
data_noindex['S10TM']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==10, 
            data_noindex['beh_num_serie']==1),1, np.NaN)
data_noindex['S10TM']=np.where(np.logical_and(data_noindex['S10TM']==1, data_noindex[Y]=='Mount'),
            data_noindex[V], np.NaN)
data_noindex['S10TM']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S10TM']) 
data_noindex['S10TM'].fillna(method = "ffill", inplace=True)
data_noindex['S10TM']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S10TM']) 
data_noindex['S10TM']=np.where(np.logical_and(data_noindex['obs_num_back']==1, data_noindex[Y]=='Ejac'),
            77777, data_noindex['S10TM'])
data_noindex['S10TM']=np.where(data_noindex['S10TM']==99999, np.NaN, data_noindex['S10TM'])
data_noindex['S10TM'].fillna(method = "backfill", inplace=True)
data_noindex['S10TM']=np.where(data_noindex['TM']==0,77777,data_noindex['S10TM'])
data_noindex['S10TM']=np.where(data_noindex['S10TM']==88888, np.NaN, data_noindex['S10TM'])
data_noindex['S10TM'].fillna(method = "ffill", inplace=True)
data_noindex['S10TM']=np.where(data_noindex['S10TM']==77777,np.NaN,data_noindex['S10TM'])

# Make a column with the time of 1st intromission in ejaculatory series 10 - the left overs
data_noindex['S10TI']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==10, 
            data_noindex['beh_num_serie']==1),1, np.NaN)
data_noindex['S10TI']=np.where(np.logical_and(data_noindex['S10TI']==1, data_noindex[Y]=='Intro'),
            data_noindex[V], np.NaN)
data_noindex['S10TI']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S10TI']) 
data_noindex['S10TI'].fillna(method = "ffill", inplace=True)
data_noindex['S10TI']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S10TI']) 
data_noindex['S10TI']=np.where(np.logical_and(data_noindex['obs_num_back']==1, data_noindex[Y]=='Ejac'),
            77777, data_noindex['S10TI'])
data_noindex['S10TI']=np.where(data_noindex['S10TI']==99999, np.NaN, data_noindex['S10TI'])
data_noindex['S10TI'].fillna(method = "backfill", inplace=True)
data_noindex['S10TI']=np.where(data_noindex['TI']==0,77777,data_noindex['S10TI'])
data_noindex['S10TI']=np.where(data_noindex['S10TI']==88888, np.NaN, data_noindex['S10TI'])
data_noindex['S10TI'].fillna(method = "ffill", inplace=True)
data_noindex['S10TI']=np.where(data_noindex['S10TI']==77777,np.NaN,data_noindex['S10TI'])


# Make column with time of 1st mount in 2nd ejaculatory series
data_noindex['S2TM']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==2, 
            data_noindex['beh_num_serie']==1),1, np.NaN)
data_noindex['S2TM']=np.where(np.logical_and(data_noindex['S2TM']==1, data_noindex[Y]=='Mount'),
            data_noindex[V], np.NaN)
data_noindex['S2TM']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S2TM']) 
data_noindex['S2TM'].fillna(method = "ffill", inplace=True)
data_noindex['S2TM']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S2TM']) 
data_noindex['S2TM']=np.where(data_noindex['S2TM']==99999, np.NaN, data_noindex['S2TM'])
data_noindex['S2TM'].fillna(method = "backfill", inplace=True)
data_noindex['S2TM']=np.where(data_noindex['TE']<2,77777,data_noindex['S2TM'])
data_noindex['S2TM']=np.where(data_noindex['S2M']==0,77777,data_noindex['S2TM'])
    # To make sure we have the mount in case it was not within a complete ejaculatory series
data_noindex['S2TM']=np.where(np.logical_and(data_noindex['S2TM']==77777, data_noindex['mount_check_serie2']==1),
            data_noindex['S10TM'], data_noindex['S2TM'])
data_noindex['S2TM']=np.where(data_noindex['S2TM']==88888, np.NaN, data_noindex['S2TM'])
data_noindex['S2TM'].fillna(method = "ffill", inplace=True)
data_noindex['S2TM']=np.where(data_noindex['S2TM']==77777,np.NaN,data_noindex['S2TM'])

# Make column with time of 1st intromission in 2nd ejaculatory series
data_noindex['S2TI']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==2, 
            data_noindex['beh_num_serie']==1),1, np.NaN)
data_noindex['S2TI']=np.where(np.logical_and(data_noindex['S2TI']==1, data_noindex[Y]=='Intro'),
            data_noindex[V], np.NaN)
data_noindex['S2TI']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S2TI']) 
data_noindex['S2TI'].fillna(method = "ffill", inplace=True)
data_noindex['S2TI']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S2TI']) 
data_noindex['S2TI']=np.where(data_noindex['S2TI']==99999, np.NaN, data_noindex['S2TI'])
data_noindex['S2TI'].fillna(method = "backfill", inplace=True)
data_noindex['S2TI']=np.where(data_noindex['TE']<2,77777,data_noindex['S2TI'])
data_noindex['S2TI']=np.where(data_noindex['S2I']==0,77777,data_noindex['S2TI'])
data_noindex['S2TI']=np.where(np.logical_and(data_noindex['S2TI']==77777, data_noindex['intro_check_serie2']==1),
            data_noindex['S10TI'], data_noindex['S2TI'])
data_noindex['S2TI']=np.where(data_noindex['S2TI']==88888, np.NaN, data_noindex['S2TI'])
data_noindex['S2TI'].fillna(method = "ffill", inplace=True)
data_noindex['S2TI']=np.where(data_noindex['S2TI']==77777,np.NaN,data_noindex['S2TI'])

# Make column with time of 1st ejaculation
data_noindex['S1TE']=np.where(data_noindex['ejac_serie']==1,data_noindex[V], np.NaN)
data_noindex['S1TE']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S1TE']) 
data_noindex['S1TE'].fillna(method = "ffill", inplace=True)
data_noindex['S1TE']=np.where(np.logical_and(data_noindex['obs_num_back']==1, data_noindex['ejac_serie_filled']!=1),
            88888, data_noindex['S1TE']) 
data_noindex['S1TE']=np.where(data_noindex['S1TE']==99999, np.NaN, data_noindex['S1TE'])
data_noindex['S1TE'].fillna(method = "backfill", inplace=True)
data_noindex['S1TE']=np.where(data_noindex['TE']<1,77777,data_noindex['S1TE'])
data_noindex['S1TE']=np.where(data_noindex['S1TE']==88888, np.NaN, data_noindex['S1TE'])
data_noindex['S1TE'].fillna(method = "ffill", inplace=True)
data_noindex['S1TE']=np.where(data_noindex['S1TE']==77777,np.NaN,data_noindex['S1TE'])

# Make column with time of 2nd ejaculation
data_noindex['S2TE']=np.where(data_noindex['ejac_serie']==2,data_noindex[V], np.NaN)
data_noindex['S2TE']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S2TE']) 
data_noindex['S2TE'].fillna(method = "ffill", inplace=True)
data_noindex['S2TE']=np.where(np.logical_and(data_noindex['obs_num_back']==1, data_noindex['ejac_serie_filled']!=2),
            88888, data_noindex['S2TE']) 
data_noindex['S2TE']=np.where(data_noindex['S2TE']==99999, np.NaN, data_noindex['S2TE'])
data_noindex['S2TE'].fillna(method = "backfill", inplace=True)
data_noindex['S2TE']=np.where(data_noindex['TE']<2,77777,data_noindex['S2TE'])
data_noindex['S2TE']=np.where(data_noindex['S2TE']==88888, np.NaN, data_noindex['S2TE'])
data_noindex['S2TE'].fillna(method = "ffill", inplace=True)
data_noindex['S2TE']=np.where(data_noindex['S2TE']==77777,np.NaN,data_noindex['S2TE'])

# Make column with time of 1st behavior in 2nd ejaculatory series
data_noindex['S2TB']=np.where(data_noindex['S2TM']<data_noindex['S2TI'], data_noindex['S2TM'], 
            data_noindex['S2TI'])

# Make column with time of 1st mount in 3rd ejaculatory series
data_noindex['S3TM']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==3, 
            data_noindex['beh_num_serie']==1),1, np.NaN)
data_noindex['S3TM']=np.where(np.logical_and(data_noindex['S3TM']==1, data_noindex[Y]=='Mount'),
            data_noindex[V], np.NaN)
data_noindex['S3TM']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S3TM']) 
data_noindex['S3TM'].fillna(method = "ffill", inplace=True)
data_noindex['S3TM']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S3TM']) 
data_noindex['S3TM']=np.where(data_noindex['S3TM']==99999, np.NaN, data_noindex['S3TM'])
data_noindex['S3TM'].fillna(method = "backfill", inplace=True)
data_noindex['S3TM']=np.where(data_noindex['TE']<3,77777,data_noindex['S3TM'])
data_noindex['S3TM']=np.where(data_noindex['S3M']==0,77777,data_noindex['S3TM'])
data_noindex['S3TM']=np.where(np.logical_and(data_noindex['S3TM']==77777, data_noindex['mount_check_serie3']==1),
            data_noindex['S10TM'], data_noindex['S3TM'])
data_noindex['S3TM']=np.where(data_noindex['S3TM']==88888, np.NaN, data_noindex['S3TM'])
data_noindex['S3TM'].fillna(method = "ffill", inplace=True)
data_noindex['S3TM']=np.where(data_noindex['S3TM']==77777,np.NaN,data_noindex['S3TM'])

# Make column with time of 1st intromission in 3rd ejaculatory series
data_noindex['S3TI']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==3, 
            data_noindex['beh_num_serie']==1),
            1, np.NaN)
data_noindex['S3TI']=np.where(np.logical_and(data_noindex['S3TI']==1, data_noindex[Y]=='Intro'),
            data_noindex[V], np.NaN)
data_noindex['S3TI']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S3TI']) 
data_noindex['S3TI'].fillna(method = "ffill", inplace=True)
data_noindex['S3TI']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S3TI']) 
data_noindex['S3TI']=np.where(data_noindex['S3TI']==99999, np.NaN, data_noindex['S3TI'])
data_noindex['S3TI'].fillna(method = "backfill", inplace=True)
data_noindex['S3TI']=np.where(data_noindex['TE']<3,77777,data_noindex['S3TI'])
data_noindex['S3TI']=np.where(data_noindex['S3I']==0,77777,data_noindex['S3TI'])
data_noindex['S3TI']=np.where(np.logical_and(data_noindex['S3TI']==77777, data_noindex['intro_check_serie3']==1),
            data_noindex['S10TI'], data_noindex['S3TI'])
data_noindex['S3TI']=np.where(data_noindex['S3TI']==88888, np.NaN, data_noindex['S3TI'])
data_noindex['S3TI'].fillna(method = "ffill", inplace=True)
data_noindex['S3TI']=np.where(data_noindex['S3TI']==77777,np.NaN,data_noindex['S3TI'])

# Make column with time of 3rd ejaculation
data_noindex['S3TE']=np.where(data_noindex['ejac_serie']==3,data_noindex[V], np.NaN)
data_noindex['S3TE']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S3TE']) 
data_noindex['S3TE'].fillna(method = "ffill", inplace=True)
data_noindex['S3TE']=np.where(np.logical_and(data_noindex['obs_num_back']==1, data_noindex['ejac_serie_filled']!=3),
            88888, data_noindex['S3TE']) 
data_noindex['S3TE']=np.where(data_noindex['S3TE']==99999, np.NaN, data_noindex['S3TE'])
data_noindex['S3TE'].fillna(method = "backfill", inplace=True)
data_noindex['S3TE']=np.where(data_noindex['TE']<3,77777,data_noindex['S3TE'])
data_noindex['S3TE']=np.where(data_noindex['S3TE']==88888, np.NaN, data_noindex['S3TE'])
data_noindex['S3TE'].fillna(method = "ffill", inplace=True)
data_noindex['S3TE']=np.where(data_noindex['S3TE']==77777,np.NaN,data_noindex['S3TE'])

# Make column with time of 1st behavior in 3rd ejaculatory series
data_noindex['S3TB']=np.where(data_noindex['S3TM']<data_noindex['S3TI'], data_noindex['S3TM'], 
            data_noindex['S3TI'])

# Make column with time of 1st mount in 4th ejaculatory series
data_noindex['S4TM']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==4, 
            data_noindex['beh_num_serie']==1),1, np.NaN)
data_noindex['S4TM']=np.where(np.logical_and(data_noindex['S4TM']==1, data_noindex[Y]=='Mount'),
            data_noindex[V], np.NaN)
data_noindex['S4TM']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S4TM']) 
data_noindex['S4TM'].fillna(method = "ffill", inplace=True)
data_noindex['S4TM']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S4TM']) 
data_noindex['S4TM']=np.where(data_noindex['S4TM']==99999, np.NaN, data_noindex['S4TM'])
data_noindex['S4TM'].fillna(method = "backfill", inplace=True)
data_noindex['S4TM']=np.where(data_noindex['TE']<4,77777,data_noindex['S4TM'])
data_noindex['S4TM']=np.where(data_noindex['S4M']==0,77777,data_noindex['S4TM'])
data_noindex['S4TM']=np.where(np.logical_and(data_noindex['S4TM']==77777, data_noindex['mount_check_serie4']==1),
            data_noindex['S10TM'], data_noindex['S4TM'])
data_noindex['S4TM']=np.where(data_noindex['S4TM']==88888, np.NaN, data_noindex['S4TM'])
data_noindex['S4TM'].fillna(method = "ffill", inplace=True)
data_noindex['S4TM']=np.where(data_noindex['S4TM']==77777,np.NaN,data_noindex['S4TM'])

# Make column with time of 1st intromission in 4th ejaculatory series
data_noindex['S4TI']=np.where(np.logical_and(data_noindex['ejac_serie_filled']==4, 
            data_noindex['beh_num_serie']==1), 1, np.NaN)
data_noindex['S4TI']=np.where(np.logical_and(data_noindex['S4TI']==1, data_noindex[Y]=='Intro'),
            data_noindex[V], np.NaN)
data_noindex['S4TI']=np.where(data_noindex['obs_num']==1, 99999, data_noindex['S4TI']) 
data_noindex['S4TI'].fillna(method = "ffill", inplace=True)
data_noindex['S4TI']=np.where(data_noindex['obs_num_back']==1, 88888, data_noindex['S4TI']) 
data_noindex['S4TI']=np.where(data_noindex['S4TI']==99999, np.NaN, data_noindex['S4TI'])
data_noindex['S4TI'].fillna(method = "backfill", inplace=True)
data_noindex['S4TI']=np.where(data_noindex['TE']<4,77777,data_noindex['S4TI'])
data_noindex['S4TI']=np.where(data_noindex['S4I']==0,77777,data_noindex['S4TI'])
data_noindex['S4TI']=np.where(np.logical_and(data_noindex['S4TI']==77777, data_noindex['intro_check_serie4']==1),
            data_noindex['S10TI'], data_noindex['S4TI'])
data_noindex['S4TI']=np.where(data_noindex['S4TI']==88888, np.NaN, data_noindex['S4TI'])
data_noindex['S4TI'].fillna(method = "ffill", inplace=True)
data_noindex['S4TI']=np.where(data_noindex['S4TI']==77777,np.NaN,data_noindex['S4TI'])

# Make column with time of 1st behavior in 4th ejaculatory series
data_noindex['S4TB']=np.where(data_noindex['S4TM']<data_noindex['S4TI'], data_noindex['S4TM'], 
            data_noindex['S4TI'])

# Calculate PEIB and PEII of 1st, 2nd and 3rd ejaculatory series
data_noindex['S1PEIB']=data_noindex['S2TB']-data_noindex['S1TE']
data_noindex['S1PEII']=data_noindex['S2TI']-data_noindex['S1TE']

data_noindex['S2PEIB']=data_noindex['S3TB']-data_noindex['S2TE']
data_noindex['S2PEII']=data_noindex['S3TI']-data_noindex['S2TE']

data_noindex['S3PEIB']=data_noindex['S4TB']-data_noindex['S2TE']
data_noindex['S3PEII']=data_noindex['S4TI']-data_noindex['S2TE']

# Calculate latency to ejaculation from 1st behavior in 2nd and 3rd ejaculatory series
data_noindex['S2LEB']=data_noindex['S2TE']-data_noindex['S2TB']
data_noindex['S3LEB']=data_noindex['S3TE']-data_noindex['S3TB']

# Calculate latency to ejaculation from 1st intromission in 2nd and 3rd ejaculatory series
data_noindex['S2LEI']=data_noindex['S2TE']-data_noindex['S2TI']
data_noindex['S3LEI']=data_noindex['S3TE']-data_noindex['S3TI']

# Calculate and make column with copulatory behavior (Mount+intromissions+ejaculations) in 2nd ejaculatory series
data_noindex['S2CB']=(data_noindex['S2M']+data_noindex['S2I'])

# Calculate and make column with intromissions ratio (intromissions/intromissions+mounts) in 2nd ejaculatory series
data_noindex['S2IR']=(data_noindex['S2I']/(data_noindex['S2I']+data_noindex['S2M']))

# Calculate and make column with intromissions interval (latency to ejacualation / intromissions) in 2nd ejaculatory series
data_noindex['S2III']=data_noindex['S2LEB']/data_noindex['S2I']

# Calculate and make column with copulatory rate (number of mounts and intromissions/ latency to ejaculation from 1st behavior) 
# in 2nd ejaculatory series
data_noindex['S2CR']=(data_noindex['S2M']+data_noindex['S2I'])/(data_noindex['S2LEB']) 

# Calculate and make column with copulatory behavior (Mount+intromissions+ejaculations) in 3rd ejaculatory series
data_noindex['S3CB']=(data_noindex['S3M']+data_noindex['S3I'])

# Calculate and make column with intromissions ratio (intromissions/intromissions+mounts) in 3rd ejaculatory series
data_noindex['S3IR']=(data_noindex['S3I']/(data_noindex['S3I']+data_noindex['S3M']))

# Calculate and make column with intromissions interval (latency to ejacualation / intromissions) in 3rd ejaculatory series
data_noindex['S3III']=data_noindex['S3LEB']/data_noindex['S3I']

# Calculate and make column with copulatory rate (number of mounts and intromissions/ latency to ejaculation from 1st behavior) 
# in 3rd ejaculatory series
data_noindex['S3CR']=(data_noindex['S3M']+data_noindex['S3I'])/(data_noindex['S3LEB']) 

data_noindex=data_noindex.replace([np.inf, -np.inf], np.nan)


# Make a new excel sheet with only the important data
data_total=data_noindex.copy()
data_total=data_total[['Observation','RatID','RatID_OBS','L1B',
                       'L1M','L1I','TM','TI','TE','TCB','TIR','TIII','TCR','S1M','S1I','S1CB',
                       'S1IR','S1III','S1CR','S1LEB','S1LEI','S1PEIB','S1PEII','S2M','S2I',
                       'S2CB','S2IR','S2III','S2CR','S2LEB','S2LEI','S2PEIB','S2PEII','S3M',
                       'S3I','S3CB','S3IR','S3III','S3CR','S3LEB','S3LEI','S3PEIB','S3PEII']]

results=data_total.groupby(X).max()
results=results[['Observation','RatID','L1B',
                       'L1M','L1I','TM','TI','TE','TCB','TIR','TIII','TCR','S1M','S1I','S1CB',
                       'S1IR','S1III','S1CR','S1LEB','S1LEI','S1PEIB','S1PEII','S2M','S2I',
                       'S2CB','S2IR','S2III','S2CR','S2LEB','S2LEI','S2PEIB','S2PEII','S3M',
                       'S3I','S3CB','S3IR','S3III','S3CR','S3LEB','S3LEI','S3PEIB','S3PEII']]

# Make a sheet to explain the columns
data_info=pd.DataFrame()
data_info['Code']=('Observation','RatID','L1B',
                       'L1M','L1I','TM','TI','TE','TCB','TIR','TIII','TCR','S1M','S1I','S1CB',
                       'S1IR','S1III','S1CR','S1LEB','S1LEI','S1PEIB','S1PEII','S2M','S2I',
                       'S2CB','S2IR','S2III','S2CR','S2LEB','S2LEI','S2PEIB','S2PEII','S3M',
                       'S3I','S3CB','S3IR','S3III','S3CR','S3LEB','S3LEI','S3PEIB','S3PEII')
data_info['Explanation']=('Observation', 'RatID',
         'Latency to 1st behavior','Latency to 1st mount', 'Latency to 1st intromission',
         'Total number of mounts','Total number of intromissions','Total number of ejaculations',
         'Total number of mounts, intromissions, and ejaculations','Total intromission ratio = TI/(TI+TM)',
         'Total inter intromission interval = Total time test / number of intromissions',
         'Total copulatory rate = mounts+intromissions+ejaculation/Total time test',
         'Number of mounts 1st ejaculatory serie', 'Number of intromissions 1st ejaculatory serie',
         'Number of mounts and intromissions 1st ejaculatory serie',
         'Intromission ration 1st ejaculatory serie','Inter intromissions interval 1st ejaculatory serie',
         'Copulatory rate 1st ejaculatory serie', 'Latency to ejaculation from 1st behavior 1st ejaculatory serie',
         'Latency to ejaculation from 1st intromissions 1st ejaculatory serie',
         'postejaculatory interval to 1st behavior 1st ejaculatory serie',
         'postejaculatory interval to 1st intromissions 1st ejaculatory serie',
         'Number of mounts 2nd ejaculatory serie', 'Number of intromissions 2nd ejaculatory serie',
         'Number of mounts and intromissions 2nd ejaculatory serie',
         'Intromission ration 2nd ejaculatory serie','Inter intromissions interval 2nd ejaculatory serie',
         'Copulatory rate 2nd ejaculatory serie', 'Latency to ejaculation from 1st behavior 2nd ejaculatory serie',
         'Latency to ejaculation from 1st intromissions 2nd ejaculatory serie',
         'postejaculatory interval to 1st behavior 2nd ejaculatory serie',
         'postejaculatory interval to 1st intromissions 2nd ejaculatory serie',
         'Number of mounts 3rd ejaculatory serie', 'Number of intromissions 3rd ejaculatory serie',
         'Number of mounts and intromissions 3rd ejaculatory serie',
         'Intromission ration 3rd ejaculatory serie','Inter intromissions interval 3rd ejaculatory serie',
         'Copulatory rate 3rd ejaculatory serie', 'Latency to ejaculation from 1st behavior 3rd ejaculatory serie',
         'Latency to ejaculation from 1st intromissions 3rd ejaculatory serie',
         'postejaculatory interval to 1st behavior 3rd ejaculatory serie',
         'postejaculatory interval to 1st intromissions 3rd ejaculatory serie')

writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
data_info.to_excel(writer,'data_info')
results.to_excel(writer,'data')
data_noindex.to_excel(writer,sheet_name='data_raw')
writer.save()
