# -*- coding: utf-8 -*-
"""
Created on March 09 17:05:35 2019

@author: Eelke Snoeren
DREADD DATA WITH 3 VIRUS GROUPS AND 2 TREATMENTS

THS PYTHON SCRIPT WILL RUN YOUR RAW DATA FILE FROM OBSERVER (IF SAVED AS XLSX-FILE)
AND MAKE IT AUTOMATICALLY IN GRAPHS AND A NEW DATA-EXCEL SHEET

CONDITIONS:
THE SCRIPT PUTS THE TESTTIME FOR LATENCIES IN CASES WHERE NO 
MOUNTS/INTROMISSION/EJACULATIONS WERE OBTAINED
IT LEAVES CELLS OF PEI OR NEXT EJACULATORY SERIES EMPTY
EXCEPT IN 1ST EJACULATORY SERIE
                                                             
TO DO BEFOREHAND
1) CHANGE THE PATH OF PYTHON TO YOUR DATA FOLDER
2) CHANGE THE FILENAME TO THE RIGHT DOCUMENT
3) FILL IN TREATMENT GROUPS (if your have more than 6, add them in SPSS-file 
conditions as well)
4) FILL IN THE TIME OF TEST
5) MATCH X-AX SCALE TO NUMBER OF TREATMENT GROUPS
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain
sns.set()
from PIL import Image

# Assign spreadsheet filename to `file`
file = 'observer_raw_data_sex.xlsx'

# Load and clean up of the data file for DataFrames
xlsx = pd.ExcelFile(file)
file_sheets = []
for sheet in xlsx.sheet_names:
    file_sheets.append(xlsx.parse(sheet))
dataraw = pd.concat(file_sheets)
dataraw = dataraw.dropna(axis=0, how='all')

# Take a look at your data
#dataraw.shape
#dataraw.head()
#dataraw.tail()

# Fill out your short column names behind the definition a-z
A='Unimportant'
B='Unimportant'
C='Unimportant'
D='Unimportant'
E='Time_hmsf'
F='Time_hms'
G='Unimportant'
H='Time'
I='Unimportant'
J='Observation'
K='Unimportant'
L='Behavior_raw'
M='Event_type'
N='RatID1'
O='RatID2' 
P='RatID3' 
Q='RatID4'
R='Treatment1'
S='Treatment2'
T='Treatment3'
U='Treatment4'
V='RatID'
W='Treatment'
X='IDTREAT'
Y='Behavior'

# Fill out your treatment/stimulus behind definition SA-SZ
SA='CTR-VEH'
SB='CTR-CNO'
SC='STIM-VEH'
SD='STIM-CNO'
SE='INH-VEH'
SF='INH-CNO' 

Stimuli_values= (SA,SB,SC,SD,SE,SF)

# Set position of bar on X axis - MAKE SURE IT MATCHES YOUR NUMBER OF GROUPS
# set width of bar
barWidth = 2
x1 = [0, 5, 10, 15, 20, 25] 
x2 = [x + barWidth for x in x1]
x3 = [0, 3, 6, 9, 12, 15]

Treatment_values= (SA,SB,SC,SD,SE,SF)
Timetest = 1800

# Rename columns (add or remove letters according to number of columns)
dataraw.columns = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U]

# Make a new datafile with selected columns
dataraw2=dataraw[[H,J,L,N,O,P,Q,R,S,T,U]]

#data.set_index([C,D], inplace=True)

# Create dataframes for ratID and Treatment 
RatID =pd.DataFrame(np.where(dataraw2[L]=='Mount2', dataraw2[O],dataraw2[N]), copy = True)
RatID =pd.DataFrame(np.where(dataraw2[L]=='Mount3', dataraw2[P],RatID[0]), copy = True)  
RatID =pd.DataFrame(np.where(dataraw2[L]=='Mount4', dataraw2[Q],RatID[0]), copy = True) 
RatID =pd.DataFrame(np.where(dataraw2[L]=='Intro2', dataraw2[O],RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(dataraw2[L]=='Intro3', dataraw2[P],RatID[0]), copy = True)  
RatID =pd.DataFrame(np.where(dataraw2[L]=='Intro4', dataraw2[Q],RatID[0]), copy = True) 
RatID =pd.DataFrame(np.where(dataraw2[L]=='Ejac2', dataraw2[O],RatID[0]), copy = True)
RatID =pd.DataFrame(np.where(dataraw2[L]=='Ejac3', dataraw2[P],RatID[0]), copy = True)  
RatID =pd.DataFrame(np.where(dataraw2[L]=='Ejac4', dataraw2[Q],RatID[0]), copy = True)

Treatment =pd.DataFrame(np.where(dataraw2[L]=='Mount2', dataraw2[S],dataraw2[R]), copy = True)
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Mount3', dataraw2[T],Treatment[0]), copy = True)  
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Mount4', dataraw2[U],Treatment[0]), copy = True) 
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Intro2', dataraw2[S],Treatment[0]), copy = True)
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Intro3', dataraw2[T],Treatment[0]), copy = True)  
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Intro4', dataraw2[U],Treatment[0]), copy = True) 
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Ejac2', dataraw2[S],Treatment[0]), copy = True)
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Ejac3', dataraw2[T],Treatment[0]), copy = True)  
Treatment =pd.DataFrame(np.where(dataraw2[L]=='Ejac4', dataraw2[U],Treatment[0]), copy = True)            

# Rename the behaviors into Mount, Intro, and Ejac (instead of the numbering)
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Mount1', 'Mount',dataraw2[L]), copy = True)
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Mount2', 'Mount',Behavior[0]), copy = True)
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Mount3', 'Mount',Behavior[0]), copy = True)  
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Mount4', 'Mount',Behavior[0]), copy = True)
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Intro1', 'Intro',Behavior[0]), copy = True) 
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Intro2', 'Intro',Behavior[0]), copy = True)
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Intro3', 'Intro',Behavior[0]), copy = True)  
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Intro4', 'Intro',Behavior[0]), copy = True) 
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Ejac1', 'Ejac',Behavior[0]), copy = True)
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Ejac2', 'Ejac',Behavior[0]), copy = True)
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Ejac3', 'Ejac',Behavior[0]), copy = True)  
Behavior =pd.DataFrame(np.where(dataraw2[L]=='Ejac4', 'Ejac',Behavior[0]), copy = True)  

# Create an unique code for rat per treatment (in case we have within-subject design)
IDTREAT =pd.DataFrame(np.where(Treatment[0]==1,(RatID[0]*100+1),Treatment[0]), copy = True)
IDTREAT =pd.DataFrame(np.where(Treatment[0]==2,(RatID[0]*100+2),IDTREAT[0]), copy = True)
IDTREAT =pd.DataFrame(np.where(Treatment[0]==3,(RatID[0]*100+3),IDTREAT[0]), copy = True)
IDTREAT =pd.DataFrame(np.where(Treatment[0]==4,(RatID[0]*100+4),IDTREAT[0]), copy = True)
IDTREAT =pd.DataFrame(np.where(Treatment[0]==5,(RatID[0]*100+5),IDTREAT[0]), copy = True)
IDTREAT =pd.DataFrame(np.where(Treatment[0]==6,(RatID[0]*100+6),IDTREAT[0]), copy = True)

# Adjust dataraw dataframe
dataraw2= dataraw2[[H,J,L]]
dataraw2 = dataraw2.reset_index()

# Add the dataframes to data
dataraw2 = pd.concat([dataraw2, RatID, Treatment, IDTREAT, Behavior], sort=False, axis=1)

# Assign column-names
dataraw2.columns = [A,H,J,L,V,W,X,Y]

# Select columns and set index to ratID and treatment in final data
data_noindex= dataraw2[[J,V,W,X,H,Y]]
#data= data_noindex.copy()
#data.set_index([J,V,W], inplace=True)

# Make a column 'durations' that contains the time spent on THAT behavior
data_noindex = data_noindex.sort_values(by=[X,H])
data_noindex['time_diff'] = data_noindex[H].diff()
data_noindex.loc[data_noindex.IDTREAT != data_noindex.IDTREAT.shift(), 'time_diff'] = None
data_noindex['durations'] = data_noindex.time_diff.shift(-1)

## In case the script does not work, you can check the data here
#writer = pd.ExcelWriter('sex_results_python.xlsx')
#dataraw.to_excel(writer, 'dataraw')
#dataraw2.to_excel(writer,'data')
#writer.save()

# Number the behaviors on occurance
data_noindex['obs_num'] = data_noindex.groupby(X)[Y].transform(lambda x: np.arange(1, len(x) + 1))

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

# Get the latency of the 1st behavior 
data_noindex['L1B'] =np.where(data_noindex['obs_num']==1,data_noindex['Time'],np.NaN)
data_noindex['L1B'].fillna(method = "ffill", inplace=True)

# Get the Latency to 1st Mount, Intromission and, Ejaculation
data_noindex['L1MIE'] =np.where((data_noindex['beh_num']==1) & (data_noindex[Y]=='Mount'),
            data_noindex['Time'],np.NaN)
data_noindex['L1MIE'] =np.where((data_noindex['beh_num']==1) & (data_noindex[Y]=='Intro'),
            data_noindex['Time'],data_noindex['L1MIE'])
data_noindex['L1MIE'] =np.where((data_noindex['beh_num']==1) & (data_noindex[Y]=='Ejac'),
            data_noindex['Time'],data_noindex['L1MIE'])
data_noindex['L1MIE'].fillna(method = "ffill", inplace=True)

# Make column with latency to 1st mount
data_noindex['L1M']=np.where(data_noindex[Y]=='Mount', data_noindex['beh_num'], np.NaN)
data_noindex['L1M']=np.where(data_noindex['L1M']==1, data_noindex[H], np.NaN)
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
data_noindex['L1I']=np.where(data_noindex['L1I']==1, data_noindex[H], np.NaN)
data_noindex['L1I']=np.where(np.logical_and(data_noindex['obs_num']==1,data_noindex[Y]!='Intro'),99999,data_noindex['L1I']) 
data_noindex['L1I'].fillna(method = "ffill", inplace=True)
data_noindex['L1I']=np.where(data_noindex['L1I']==99999, np.NaN, data_noindex['L1I'])
data_noindex['L1I'].fillna(method = "backfill", inplace=True)
data_noindex['L1I']=np.where(data_noindex['intro_check2']==2, Timetest, data_noindex['L1I'])

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
data_noindex['TII']=Timetest/data_noindex['TI']

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
data_noindex = data_noindex.sort_values(by=[X,H], ascending = False)
data_noindex['beh_num_back_serie'] = data_noindex.groupby('beh_num_trick_serie')[Y].transform(lambda x: np.arange(1, len(x) + 1))

data_noindex = data_noindex.sort_values(by=[X,H])

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
data_noindex['S1II']=data_noindex['S1LEB']/data_noindex['S1I']

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
            data_noindex['Time'], np.NaN)
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
            data_noindex['Time'], np.NaN)
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
            data_noindex['Time'], np.NaN)
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
            data_noindex['Time'], np.NaN)
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
data_noindex['S1TE']=np.where(data_noindex['ejac_serie']==1,data_noindex['Time'], np.NaN)
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
data_noindex['S2TE']=np.where(data_noindex['ejac_serie']==2,data_noindex['Time'], np.NaN)
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
            data_noindex['Time'], np.NaN)
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
            data_noindex['Time'], np.NaN)
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
data_noindex['S3TE']=np.where(data_noindex['ejac_serie']==3,data_noindex['Time'], np.NaN)
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
            data_noindex['Time'], np.NaN)
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
            data_noindex['Time'], np.NaN)
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
data_noindex['S2II']=data_noindex['S2LEB']/data_noindex['S2I']

# Calculate and make column with copulatory rate (number of mounts and intromissions/ latency to ejaculation from 1st behavior) 
# in 2nd ejaculatory series
data_noindex['S2CR']=(data_noindex['S2M']+data_noindex['S2I'])/(data_noindex['S2LEB']) 

# Calculate and make column with copulatory behavior (Mount+intromissions+ejaculations) in 3rd ejaculatory series
data_noindex['S3CB']=(data_noindex['S3M']+data_noindex['S3I'])

# Calculate and make column with intromissions ratio (intromissions/intromissions+mounts) in 3rd ejaculatory series
data_noindex['S3IR']=(data_noindex['S3I']/(data_noindex['S3I']+data_noindex['S3M']))

# Calculate and make column with intromissions interval (latency to ejacualation / intromissions) in 3rd ejaculatory series
data_noindex['S3II']=data_noindex['S3LEB']/data_noindex['S3I']

# Calculate and make column with copulatory rate (number of mounts and intromissions/ latency to ejaculation from 1st behavior) 
# in 3rd ejaculatory series
data_noindex['S3CR']=(data_noindex['S3M']+data_noindex['S3I'])/(data_noindex['S3LEB']) 

# Make a new excel sheet with only the important data
data_total=data_noindex.copy()
data_total=data_total[['Observation','RatID','Treatment','IDTREAT','L1B',
                       'L1M','L1I','TM','TI','TE','TCB','TIR','TII','TCR','S1M','S1I','S1CB',
                       'S1IR','S1II','S1CR','S1LEB','S1LEI','S1PEIB','S1PEII','S2M','S2I',
                       'S2CB','S2IR','S2II','S2CR','S2LEB','S2LEI','S2PEIB','S2PEII','S3M',
                       'S3I','S3CB','S3IR','S3II','S3CR','S3LEB','S3LEI','S3PEIB','S3PEII']]

results=data_total.groupby(X).max()
results['IDTREAT']=(results['RatID']*100+results['Treatment'])
results=results[['Observation','IDTREAT','RatID','Treatment','L1B',
                       'L1M','L1I','TM','TI','TE','TCB','TIR','TII','TCR','S1M','S1I','S1CB',
                       'S1IR','S1II','S1CR','S1LEB','S1LEI','S1PEIB','S1PEII','S2M','S2I',
                       'S2CB','S2IR','S2II','S2CR','S2LEB','S2LEI','S2PEIB','S2PEII','S3M',
                       'S3I','S3CB','S3IR','S3II','S3CR','S3LEB','S3LEI','S3PEIB','S3PEII']]

# Make a sheet to explain the columns
data_info=pd.DataFrame()
data_info['Code']=('Observation','RatID','Treatment','IDTREAT','L1B',
                       'L1M','L1I','TM','TI','TE','TCB','TIR','TII','TCR','S1M','S1I','S1CB',
                       'S1IR','S1II','S1CR','S1LEB','S1LEI','S1PEIB','S1PEII','S2M','S2I',
                       'S2CB','S2IR','S2II','S2CR','S2LEB','S2LEI','S2PEIB','S2PEII','S3M',
                       'S3I','S3CB','S3IR','S3II','S3CR','S3LEB','S3LEI','S3PEIB','S3PEII')
data_info['Explanation']=('Observation', 'RatID','Treatment','Unique code per rat per treatment',
         'Latency to 1st behavior','Latency to 1st mount', 'Latency to 1st intromission',
         'Total number of mounts','Total number of intromissions','Total number of ejaculations',
         'Total number of mounts and intromissions','Intromission ratio = TI/(TI+TM)',
         'Intromission interval = Time to ejaculation from 1st behavior / intromissions',
         'Copulatory rate = mounts+intromissions/time to ejaculation from 1st behavior',
         'Number of mounts 1st ejaculatory serie', 'Number of intromissions 1st ejaculatory serie',
         'Number of mounts and intromissions 1st ejaculatory serie',
         'Intromission ration 1st ejaculatory serie','Intromissions interval 1st ejaculatory serie',
         'Copulatory rate 1st ejaculatory serie', 'Latency to ejaculation from 1st behavior 1st ejaculatory serie',
         'Latency to ejaculation from 1st intromissions 1st ejaculatory serie',
         'postejaculatory interval to 1st behavior 1st ejaculatory serie',
         'postejaculatory interval to 1st intromissions 1st ejaculatory serie',
         'Number of mounts 2nd ejaculatory serie', 'Number of intromissions 2nd ejaculatory serie',
         'Number of mounts and intromissions 2nd ejaculatory serie',
         'Intromission ration 2nd ejaculatory serie','Intromissions interval 2nd ejaculatory serie',
         'Copulatory rate 2nd ejaculatory serie', 'Latency to ejaculation from 1st behavior 2nd ejaculatory serie',
         'Latency to ejaculation from 1st intromissions 2nd ejaculatory serie',
         'postejaculatory interval to 1st behavior 2nd ejaculatory serie',
         'postejaculatory interval to 1st intromissions 2nd ejaculatory serie',
         'Number of mounts 3rd ejaculatory serie', 'Number of intromissions 3rd ejaculatory serie',
         'Number of mounts and intromissions 3rd ejaculatory serie',
         'Intromission ration 3rd ejaculatory serie','Intromissions interval 3rd ejaculatory serie',
         'Copulatory rate 3rd ejaculatory serie', 'Latency to ejaculation from 1st behavior 3rd ejaculatory serie',
         'Latency to ejaculation from 1st intromissions 3rd ejaculatory serie',
         'postejaculatory interval to 1st behavior 3rd ejaculatory serie',
         'postejaculatory interval to 1st intromissions 3rd ejaculatory serie',)

# Make an empty DataFrame for statistical output
data_stat=pd.DataFrame()

# Statistics on the data
mean=results.groupby('Treatment')['L1B','L1M','L1I','TM','TI','TE','TCB','TIR','TII',
                 'TCR','S1M','S1I','S1CB','S1IR','S1II','S1CR','S1LEB','S1LEI',
                 'S1PEIB','S1PEII','S2M','S2I','S2CB','S2IR','S2II','S2CR','S2LEB',
                 'S2LEI','S2PEIB','S2PEII','S3M','S3I','S3CB','S3IR','S3II','S3CR',
                 'S3LEB','S3LEI','S3PEIB','S3PEII'].mean()
median=results.groupby('Treatment')['L1B','L1M','L1I','TM','TI','TE','TCB','TIR','TII',
                 'TCR','S1M','S1I','S1CB','S1IR','S1II','S1CR','S1LEB','S1LEI',
                 'S1PEIB','S1PEII','S2M','S2I','S2CB','S2IR','S2II','S2CR','S2LEB',
                 'S2LEI','S2PEIB','S2PEII','S3M','S3I','S3CB','S3IR','S3II','S3CR',
                 'S3LEB','S3LEI','S3PEIB','S3PEII'].median()
std=results.groupby('Treatment')['L1B','L1M','L1I','TM','TI','TE','TCB','TIR','TII',
                 'TCR','S1M','S1I','S1CB','S1IR','S1II','S1CR','S1LEB','S1LEI',
                 'S1PEIB','S1PEII','S2M','S2I','S2CB','S2IR','S2II','S2CR','S2LEB',
                 'S2LEI','S2PEIB','S2PEII','S3M','S3I','S3CB','S3IR','S3II','S3CR',
                 'S3LEB','S3LEI','S3PEIB','S3PEII'].std()
sem=results.groupby('Treatment')['L1B','L1M','L1I','TM','TI','TE','TCB','TIR','TII',
                 'TCR','S1M','S1I','S1CB','S1IR','S1II','S1CR','S1LEB','S1LEI',
                 'S1PEIB','S1PEII','S2M','S2I','S2CB','S2IR','S2II','S2CR','S2LEB',
                 'S2LEI','S2PEIB','S2PEII','S3M','S3I','S3CB','S3IR','S3II','S3CR',
                 'S3LEB','S3LEI','S3PEIB','S3PEII'].sem()
var=results.groupby('Treatment')['L1B','L1M','L1I','TM','TI','TE','TCB','TIR','TII',
                 'TCR','S1M','S1I','S1CB','S1IR','S1II','S1CR','S1LEB','S1LEI',
                 'S1PEIB','S1PEII','S2M','S2I','S2CB','S2IR','S2II','S2CR','S2LEB',
                 'S2LEI','S2PEIB','S2PEII','S3M','S3I','S3CB','S3IR','S3II','S3CR',
                 'S3LEB','S3LEI','S3PEIB','S3PEII'].var()
q25=results.groupby('Treatment')['L1B','L1M','L1I','TM','TI','TE','TCB','TIR','TII',
                 'TCR','S1M','S1I','S1CB','S1IR','S1II','S1CR','S1LEB','S1LEI',
                 'S1PEIB','S1PEII','S2M','S2I','S2CB','S2IR','S2II','S2CR','S2LEB',
                 'S2LEI','S2PEIB','S2PEII','S3M','S3I','S3CB','S3IR','S3II','S3CR',
                 'S3LEB','S3LEI','S3PEIB','S3PEII'].quantile(q=0.25, axis=0)
q75=results.groupby('Treatment')['L1B','L1M','L1I','TM','TI','TE','TCB','TIR','TII',
                 'TCR','S1M','S1I','S1CB','S1IR','S1II','S1CR','S1LEB','S1LEI',
                 'S1PEIB','S1PEII','S2M','S2I','S2CB','S2IR','S2II','S2CR','S2LEB',
                 'S2LEI','S2PEIB','S2PEII','S3M','S3I','S3CB','S3IR','S3II','S3CR',
                 'S3LEB','S3LEI','S3PEIB','S3PEII'].quantile(q=0.75, axis=0)

# Calculate n per group and squareroot for sem median
npg=results.groupby('Treatment').size()
sqrtn=np.sqrt(npg)*1.34

# Calculate standard error of median
semedianL1B = pd.DataFrame(((q75['L1B']-q25['L1B'])/sqrtn), copy=True)
semedianL1M = pd.DataFrame(((q75['L1M']-q25['L1M'])/sqrtn), copy=True)
semedianL1I = pd.DataFrame(((q75['L1I']-q25['L1I'])/sqrtn), copy=True)
semedianTM = pd.DataFrame(((q75['TM']-q25['TM'])/sqrtn), copy=True)
semedianTI = pd.DataFrame(((q75['TI']-q25['TI'])/sqrtn), copy=True)
semedianTE = pd.DataFrame(((q75['TE']-q25['TE'])/sqrtn), copy=True)
semedianTCB = pd.DataFrame(((q75['TCB']-q25['TCB'])/sqrtn), copy=True)
semedianTIR = pd.DataFrame(((q75['TIR']-q25['TIR'])/sqrtn), copy=True)
semedianTII = pd.DataFrame(((q75['TII']-q25['TII'])/sqrtn), copy=True)
semedianTCR = pd.DataFrame(((q75['TCR']-q25['TCR'])/sqrtn), copy=True)
semedianS1M = pd.DataFrame(((q75['S1M']-q25['S1M'])/sqrtn), copy=True)
semedianS1I = pd.DataFrame(((q75['S1I']-q25['S1I'])/sqrtn), copy=True)
semedianS1CB = pd.DataFrame(((q75['S1CB']-q25['S1CB'])/sqrtn), copy=True)
semedianS1IR = pd.DataFrame(((q75['S1IR']-q25['S1IR'])/sqrtn), copy=True)
semedianS1II = pd.DataFrame(((q75['S1II']-q25['S1II'])/sqrtn), copy=True)
semedianS1CR = pd.DataFrame(((q75['S1CR']-q25['S1CR'])/sqrtn), copy=True)
semedianS1LEB = pd.DataFrame(((q75['S1LEB']-q25['S1LEB'])/sqrtn), copy=True)
semedianS1LEI = pd.DataFrame(((q75['S1LEI']-q25['S1LEI'])/sqrtn), copy=True)
semedianS1PEIB = pd.DataFrame(((q75['S1PEIB']-q25['S1PEIB'])/sqrtn), copy=True)
semedianS1PEII = pd.DataFrame(((q75['S1PEII']-q25['S1PEII'])/sqrtn), copy=True)
semedianS2M = pd.DataFrame(((q75['S2M']-q25['S2M'])/sqrtn), copy=True)
semedianS2I = pd.DataFrame(((q75['S2I']-q25['S2I'])/sqrtn), copy=True)
semedianS2CB = pd.DataFrame(((q75['S2CB']-q25['S2CB'])/sqrtn), copy=True)
semedianS2IR = pd.DataFrame(((q75['S2IR']-q25['S2IR'])/sqrtn), copy=True)
semedianS2II = pd.DataFrame(((q75['S2II']-q25['S2II'])/sqrtn), copy=True)
semedianS2CR = pd.DataFrame(((q75['S2CR']-q25['S2CR'])/sqrtn), copy=True)
semedianS2LEB = pd.DataFrame(((q75['S2LEB']-q25['S2LEB'])/sqrtn), copy=True)
semedianS2LEI = pd.DataFrame(((q75['S2LEI']-q25['S2LEI'])/sqrtn), copy=True)
semedianS2PEIB = pd.DataFrame(((q75['S2PEIB']-q25['S2PEIB'])/sqrtn), copy=True)
semedianS2PEII = pd.DataFrame(((q75['S2PEII']-q25['S2PEII'])/sqrtn), copy=True)
semedianS3M = pd.DataFrame(((q75['S3M']-q25['S3M'])/sqrtn), copy=True)
semedianS3I = pd.DataFrame(((q75['S3I']-q25['S3I'])/sqrtn), copy=True)
semedianS3CB = pd.DataFrame(((q75['S3CB']-q25['S3CB'])/sqrtn), copy=True)
semedianS3IR = pd.DataFrame(((q75['S3IR']-q25['S3IR'])/sqrtn), copy=True)
semedianS3II = pd.DataFrame(((q75['S3II']-q25['S3II'])/sqrtn), copy=True)
semedianS3CR = pd.DataFrame(((q75['S3CR']-q25['S3CR'])/sqrtn), copy=True)
semedianS3LEB = pd.DataFrame(((q75['S3LEB']-q25['S3LEB'])/sqrtn), copy=True)
semedianS3LEI = pd.DataFrame(((q75['S3LEI']-q25['S3LEI'])/sqrtn), copy=True)
semedianS3PEIB = pd.DataFrame(((q75['S3PEIB']-q25['S3PEIB'])/sqrtn), copy=True)
semedianS3PEII = pd.DataFrame(((q75['S3PEII']-q25['S3PEII'])/sqrtn), copy=True)

semedian = pd.concat([semedianL1B, semedianL1M, semedianL1I , semedianTM, 
                      semedianTI, semedianTE, semedianTCB, semedianTIR, semedianTII, 
                      semedianTCR, semedianS1M, semedianS1I, semedianS1CB, 
                      semedianS1IR, semedianS1II, semedianS1CR, semedianS1LEB, 
                      semedianS1LEI, semedianS1PEIB, semedianS1PEII, semedianS2M, 
                      semedianS2I, semedianS2CB, semedianS2IR, semedianS2II, 
                      semedianS2CR, semedianS2LEB, semedianS2LEI, semedianS2PEIB, 
                      semedianS2PEII, semedianS3M, semedianS3I, semedianS3CB, 
                      semedianS3IR, semedianS3II, semedianS3CR , semedianS3LEB, 
                      semedianS3LEI, semedianS3PEIB, semedianS3PEII], sort=False, axis=1)
         
# Write mean statistics to the new dataframe data_stat
data_stat_mean= pd.concat([data_stat, mean, sem], sort=False, axis=1)
data_stat_rest= pd.concat([data_stat, median, semedian, q25, q75, std, var], sort=False, axis=1)

# Rename the column names
data_stat_mean.columns=['mean_L1B','mean_L1M','mean_L1I','mean_TM','mean_TI',
                        'mean_TE','mean_TCB','mean_TIR','mean_TII','mean_TCR',
                        'mean_S1M','mean_S1I','mean_S1CB','mean_S1IR','mean_S1II',
                        'mean_S1CR','mean_S1LEB','mean_S1LEI','mean_S1PEIB',
                        'mean_S1PEII','mean_S2M','mean_S2I','mean_S2CB','mean_S2IR',
                        'mean_S2II','mean_S2CR','mean_S2LEB','mean_S2LEI','mean_S2PEIB',
                        'mean_S2PEII','mean_S3M','mean_S3I','mean_S3CB','mean_S3IR',
                        'mean_S3II','mean_S3CR','mean_S3LEB','mean_S3LEI','mean_S3PEIB',
                        'mean_S3PEII', 'sem_L1B','sem_L1M','sem_L1I','sem_TM','sem_TI',
                        'sem_TE','sem_TCB','sem_TIR','sem_TII','sem_TCR','sem_S1M',
                        'sem_S1I','sem_S1CB','sem_S1IR','sem_S1II','sem_S1CR',
                        'sem_S1LEB','sem_S1LEI','sem_S1PEIB','sem_S1PEII','sem_S2M',
                        'sem_S2I','sem_S2CB','sem_S2IR','sem_S2II','sem_S2CR',
                        'sem_S2LEB','sem_S2LEI','sem_S2PEIB','sem_S2PEII','sem_S3M',
                        'sem_S3I','sem_S3CB','sem_S3IR','sem_S3II','sem_S3CR',
                        'sem_S3LEB','sem_S3LEI','sem_S3PEIB','sem_S3PEII']
data_stat_rest.columns=['median_L1B','median_L1M','median_L1I','median_TM','median_TI',
                        'median_TE','median_TCB','median_TIR','median_TII','median_TCR',
                        'median_S1M','median_S1I','median_S1CB','median_S1IR','median_S1II',
                        'median_S1CR','median_S1LEB','median_S1LEI','median_S1PEIB',
                        'median_S1PEII','median_S2M','median_S2I','median_S2CB','median_S2IR',
                        'median_S2II','median_S2CR','median_S2LEB','median_S2LEI','median_S2PEIB',
                        'median_S2PEII','median_S3M','median_S3I','median_S3CB','median_S3IR',
                        'median_S3II','median_S3CR','median_S3LEB','median_S3LEI','median_S3PEIB',
                        'median_S3PEII', 'semedian_L1B','semedian_L1M','semedian_L1I','semedian_TM','semedian_TI',
                        'semedian_TE','semedian_TCB','semedian_TIR','semedian_TII','semedian_TCR',
                        'semedian_S1M','semedian_S1I','semedian_S1CB','semedian_S1IR','semedian_S1II',
                        'semedian_S1CR','semedian_S1LEB','semedian_S1LEI','semedian_S1PEIB',
                        'semedian_S1PEII','semedian_S2M','semedian_S2I','semedian_S2CB','semedian_S2IR',
                        'semedian_S2II','semedian_S2CR','semedian_S2LEB','semedian_S2LEI','semedian_S2PEIB',
                        'semedian_S2PEII','semedian_S3M','semedian_S3I','semedian_S3CB','semedian_S3IR',
                        'semedian_S3II','semedian_S3CR','semedian_S3LEB','semedian_S3LEI','semedian_S3PEIB',
                        'semedian_S3PEII', 'q25_L1B','q25_L1M','q25_L1I','q25_TM','q25_TI',
                        'q25_TE','q25_TCB','q25_TIR','q25_TII','q25_TCR','q25_S1M',
                        'q25_S1I','q25_S1CB','q25_S1IR','q25_S1II','q25_S1CR',
                        'q25_S1LEB','q25_S1LEI','q25_S1PEIB','q25_S1PEII','q25_S2M',
                        'q25_S2I','q25_S2CB','q25_S2IR','q25_S2II','q25_S2CR',
                        'q25_S2LEB','q25_S2LEI','q25_S2PEIB','q25_S2PEII','q25_S3M',
                        'q25_S3I','q25_S3CB','q25_S3IR','q25_S3II','q25_S3CR',
                        'q25_S3LEB','q25_S3LEI','q25_S3PEIB','q25_S3PEII', 'q75_L1B',
                        'q75_L1M','q75_L1I','q75_TM','q75_TI','q75_TE','q75_TCB',
                        'q75_TIR','q75_TII','q75_TCR','q75_S1M','q75_S1I','q75_S1CB',
                        'q75_S1IR','q75_S1II','q75_S1CR','q75_S1LEB','q75_S1LEI',
                        'q75_S1PEIB','q75_S1PEII','q75_S2M','q75_S2I','q75_S2CB',
                        'q75_S2IR','q75_S2II','q75_S2CR','q75_S2LEB','q75_S2LEI',
                        'q75_S2PEIB','q75_S2PEII','q75_S3M','q75_S3I','q75_S3CB',
                        'q75_S3IR','q75_S3II','q75_S3CR','q75_S3LEB','q75_S3LEI',
                        'q75_S3PEIB','q75_S3PEII','std_L1B','std_L1M','std_L1I','std_TM','std_TI',
                        'std_TE','std_TCB','std_TIR','std_TII','std_TCR','std_S1M',
                        'std_S1I','std_S1CB','std_S1IR','std_S1II','std_S1CR',
                        'std_S1LEB','std_S1LEI','std_S1PEIB','std_S1PEII','std_S2M',
                        'std_S2I','std_S2CB','std_S2IR','std_S2II','std_S2CR',
                        'std_S2LEB','std_S2LEI','std_S2PEIB','std_S2PEII','std_S3M',
                        'std_S3I','std_S3CB','std_S3IR','std_S3II','std_S3CR',
                        'std_S3LEB','std_S3LEI','std_S3PEIB','std_S3PEII', 'var_L1B',
                        'var_L1M','var_L1I','var_TM','var_TI','var_TE','var_TCB',
                        'var_TIR','var_TII','var_TCR','var_S1M','var_S1I','var_S1CB',
                        'var_S1IR','var_S1II','var_S1CR','var_S1LEB','var_S1LEI',
                        'var_S1PEIB','var_S1PEII','var_S2M','var_S2I','var_S2CB',
                        'var_S2IR','var_S2II','var_S2CR','var_S2LEB','var_S2LEI',
                        'var_S2PEIB','var_S2PEII','var_S3M','var_S3I','var_S3CB',
                        'var_S3IR','var_S3II','var_S3CR','var_S3LEB','var_S3LEI',
                        'var_S3PEIB','var_S3PEII']
                           
# Re-order the columns
data_stat_mean = data_stat_mean[['mean_L1B','sem_L1B','mean_L1M','sem_L1M',
                                 'mean_L1I','sem_L1I','mean_TM','sem_TM',
                                 'mean_TI','sem_TI','mean_TE','sem_TE',
                                 'mean_TCB','sem_TCB','mean_TIR', 'sem_TIR',
                                 'mean_TII','sem_TII','mean_TCR', 'sem_TCR',
                                 'mean_S1M','sem_S1M','mean_S1I','sem_S1I',
                                 'mean_S1CB','sem_S1CB','mean_S1IR','sem_S1IR',
                                 'mean_S1II','sem_S1II','mean_S1CR','sem_S1CR',
                                 'mean_S1LEB','sem_S1LEB','mean_S1LEI','sem_S1LEI',
                                 'mean_S1PEIB','sem_S1PEIB','mean_S1PEII','sem_S1PEII',
                                 'mean_S2M','sem_S2M','mean_S2I','sem_S2I',
                                 'mean_S2CB','sem_S2CB','mean_S2IR','sem_S2IR',
                                 'mean_S2II','sem_S2II','mean_S2CR','sem_S2CR',
                                 'mean_S2LEB','sem_S2LEB','mean_S2LEI','sem_S2LEI',
                                 'mean_S2PEIB','sem_S2PEIB','mean_S2PEII','sem_S2PEII',
                                 'mean_S3M','sem_S3M','mean_S3I','sem_S3I',
                                 'mean_S3CB','sem_S3CB','mean_S3IR','sem_S3IR',
                                 'mean_S3II','sem_S3II','mean_S3CR','sem_S3CR',
                                 'mean_S3LEB','sem_S3LEB','mean_S3LEI','sem_S3LEI',
                                 'mean_S3PEIB','sem_S3PEIB','mean_S3PEII','sem_S3PEII']]             
  
data_stat_rest.columns=[ 'median_L1B','semedian_L1B','median_L1M','semedian_L1M',
                                 'median_L1I','semedian_L1I','median_TM','semedian_TM',
                                 'median_TI','semedian_TI','median_TE','semedian_TE',
                                 'median_TCB','semedian_TCB','median_TIR', 'semedian_TIR',
                                 'median_TII','semedian_TII','median_TCR', 'semedian_TCR',
                                 'median_S1M','semedian_S1M','median_S1I','semedian_S1I',
                                 'median_S1CB','semedian_S1CB','median_S1IR','semedian_S1IR',
                                 'median_S1II','semedian_S1II','median_S1CR','semedian_S1CR',
                                 'median_S1LEB','semedian_S1LEB','median_S1LEI','semedian_S1LEI',
                                 'median_S1PEIB','semedian_S1PEIB','median_S1PEII','semedian_S1PEII',
                                 'median_S2M','semedian_S2M','median_S2I','semedian_S2I',
                                 'median_S2CB','semedian_S2CB','median_S2IR','semedian_S2IR',
                                 'median_S2II','semedian_S2II','median_S2CR','semedian_S2CR',
                                 'median_S2LEB','semedian_S2LEB','median_S2LEI','semedian_S2LEI',
                                 'median_S2PEIB','semedian_S2PEIB','median_S2PEII','semedian_S2PEII',
                                 'median_S3M','semedian_S3M','median_S3I','semedian_S3I',
                                 'median_S3CB','semedian_S3CB','median_S3IR','semedian_S3IR',
                                 'median_S3II','semedian_S3II','median_S3CR','semedian_S3CR',
                                 'median_S3LEB','semedian_S3LEB','median_S3LEI','semedian_S3LEI',
                                 'median_S3PEIB','semedian_S3PEIB','median_S3PEII','semedian_S3PEII','q25_L1B','q25_L1M','q25_L1I','q25_TM','q25_TI',
                                 'q25_TE','q25_TCB','q25_TIR','q25_TII','q25_TCR','q25_S1M',
                                 'q25_S1I','q25_S1CB','q25_S1IR','q25_S1II','q25_S1CR',
                                 'q25_S1LEB','q25_S1LEI','q25_S1PEIB','q25_S1PEII','q25_S2M',
                                 'q25_S2I','q25_S2CB','q25_S2IR','q25_S2II','q25_S2CR',
                                 'q25_S2LEB','q25_S2LEI','q25_S2PEIB','q25_S2PEII','q25_S3M',
                                 'q25_S3I','q25_S3CB','q25_S3IR','q25_S3II','q25_S3CR',
                                 'q25_S3LEB','q25_S3LEI','q25_S3PEIB','q25_S3PEII', 'q75_L1B',
                                 'q75_L1M','q75_L1I','q75_TM','q75_TI','q75_TE','q75_TCB',
                                 'q75_TIR','q75_TII','q75_TCR','q75_S1M','q75_S1I','q75_S1CB',
                                 'q75_S1IR','q75_S1II','q75_S1CR','q75_S1LEB','q75_S1LEI',
                                 'q75_S1PEIB','q75_S1PEII','q75_S2M','q75_S2I','q75_S2CB',
                                 'q75_S2IR','q75_S2II','q75_S2CR','q75_S2LEB','q75_S2LEI',
                                 'q75_S2PEIB','q75_S2PEII','q75_S3M','q75_S3I','q75_S3CB',
                                 'q75_S3IR','q75_S3II','q75_S3CR','q75_S3LEB','q75_S3LEI',
                                 'q75_S3PEIB','q75_S3PEII','std_L1B','std_L1M','std_L1I',
                                 'std_TM','std_TI','std_TE','std_TCB','std_TIR','std_TII',
                                 'std_TCR','std_S1M','std_S1I','std_S1CB','std_S1IR',
                                 'std_S1II','std_S1CR','std_S1LEB','std_S1LEI','std_S1PEIB',
                                 'std_S1PEII','std_S2M','std_S2I','std_S2CB','std_S2IR',
                                 'std_S2II','std_S2CR','std_S2LEB','std_S2LEI','std_S2PEIB',
                                 'std_S2PEII','std_S3M','std_S3I','std_S3CB','std_S3IR',
                                 'std_S3II','std_S3CR','std_S3LEB','std_S3LEI','std_S3PEIB',
                                 'std_S3PEII', 'var_L1B','var_L1M','var_L1I','var_TM',
                                 'var_TI','var_TE','var_TCB','var_TIR','var_TII','var_TCR',
                                 'var_S1M','var_S1I','var_S1CB','var_S1IR','var_S1II',
                                 'var_S1CR','var_S1LEB','var_S1LEI','var_S1PEIB','var_S1PEII',
                                 'var_S2M','var_S2I','var_S2CB','var_S2IR','var_S2II',
                                 'var_S2CR','var_S2LEB','var_S2LEI','var_S2PEIB','var_S2PEII',
                                 'var_S3M','var_S3I','var_S3CB','var_S3IR','var_S3II','var_S3CR',
                                 'var_S3LEB','var_S3LEI','var_S3PEIB','var_S3PEII']
                      
#data_stat_mean.info()

# Create dataframes for SPSS excelsheet
SPSS=pd.DataFrame()
SPSS1 = results.loc[results['Treatment'] == 1]
SPSS1.columns=['Observation1','IDTREAT','RatID1','Treatment1','L1B1',
                       'L1M1','L1I1','TM1','TI1','TE1','TCB1','TIR1','TII1','TCR1','S1M1','S1I1','S1CB1',
                       'S1IR1','S1II1','S1CR1','S1LEB1','S1LEI1','S1PEIB1','S1PEII1','S2M1','S2I1',
                       'S2CB1','S2IR1','S2II1','S2CR1','S2LEB1','S2LEI1','S2PEIB1','S2PEII1','S3M1',
                       'S3I1','S3CB1','S3IR1','S3II1','S3CR1','S3LEB1','S3LEI1','S3PEIB1','S3PEII1']
SPSS1.set_index(['RatID1'], inplace= True)
SPSS1 = SPSS1.sort_values(['RatID1'],ascending=True)

SPSS2 = results.loc[results['Treatment'] == 2]
SPSS2.columns=['Observation2','IDTREAT','RatID2','Treatment2','L1B2',
                       'L1M2','L1I2','TM2','TI2','TE2','TCB2','TIR2','TII2','TCR2','S1M2','S1I2','S1CB2',
                       'S1IR2','S1II2','S1CR2','S1LEB2','S1LEI2','S1PEIB2','S1PEII2','S2M2','S2I2',
                       'S2CB2','S2IR2','S2II2','S2CR2','S2LEB2','S2LEI2','S2PEIB2','S2PEII2','S3M2',
                       'S3I2','S3CB2','S3IR2','S3II2','S3CR2','S3LEB2','S3LEI2','S3PEIB2','S3PEII2']
SPSS2.set_index(['RatID2'], inplace= True)
SPSS2 = SPSS2.sort_values(['RatID2'],ascending=True)

SPSS3 = results.loc[results['Treatment'] == 3]
SPSS3.columns=['Observation3','IDTREAT','RatID3','Treatment3','L1B3',
                       'L1M3','L1I3','TM3','TI3','TE3','TCB3','TIR3','TII3','TCR3','S1M3','S1I3','S1CB3',
                       'S1IR3','S1II3','S1CR3','S1LEB3','S1LEI3','S1PEIB3','S1PEII3','S2M3','S2I3',
                       'S2CB3','S2IR3','S2II3','S2CR3','S2LEB3','S2LEI3','S2PEIB3','S2PEII3','S3M3',
                       'S3I3','S3CB3','S3IR3','S3II3','S3CR3','S3LEB3','S3LEI3','S3PEIB3','S3PEII3']
SPSS3.set_index(['RatID3'], inplace= True)
SPSS3 = SPSS3.sort_values(['RatID3'],ascending=True)

SPSS4 = results.loc[results['Treatment'] == 4]
SPSS4.columns=['Observation4','IDTREAT','RatID4','Treatment4','L1B4',
                       'L1M4','L1I4','TM4','TI4','TE4','TCB4','TIR4','TII4','TCR4','S1M4','S1I4','S1CB4',
                       'S1IR4','S1II4','S1CR4','S1LEB4','S1LEI4','S1PEIB4','S1PEII4','S2M4','S2I4',
                       'S2CB4','S2IR4','S2II4','S2CR4','S2LEB4','S2LEI4','S2PEIB4','S2PEII4','S3M4',
                       'S3I4','S3CB4','S3IR4','S3II4','S3CR4','S3LEB4','S3LEI4','S3PEIB4','S3PEII4']
SPSS4.set_index(['RatID4'], inplace= True)
SPSS4 = SPSS4.sort_values(['RatID4'],ascending=True)

SPSS5 = results.loc[results['Treatment'] == 5]
SPSS5.columns=['Observation5','IDTREAT','RatID5','Treatment5','L1B5',
                       'L1M5','L1I5','TM5','TI5','TE5','TCB5','TIR5','TII5','TCR5','S1M5','S1I5','S1CB5',
                       'S1IR5','S1II5','S1CR5','S1LEB5','S1LEI5','S1PEIB5','S1PEII5','S2M5','S2I5',
                       'S2CB5','S2IR5','S2II5','S2CR5','S2LEB5','S2LEI5','S2PEIB5','S2PEII5','S3M5',
                       'S3I5','S3CB5','S3IR5','S3II5','S3CR5','S3LEB5','S3LEI5','S3PEIB5','S3PEII5']
SPSS5.set_index(['RatID5'], inplace= True)
SPSS5 = SPSS5.sort_values(['RatID5'],ascending=True)

SPSS6 = results.loc[results['Treatment'] == 6]
SPSS6.columns=['Observation6','IDTREAT','RatID6','Treatment6','L1B6',
                       'L1M6','L1I6','TM6','TI6','TE6','TCB6','TIR6','TII6','TCR6','S1M6','S1I6','S1CB6',
                       'S1IR6','S1II6','S1CR6','S1LEB6','S1LEI6','S1PEIB6','S1PEII6','S2M6','S2I6',
                       'S2CB6','S2IR6','S2II6','S2CR6','S2LEB6','S2LEI6','S2PEIB6','S2PEII6','S3M6',
                       'S3I6','S3CB6','S3IR6','S3II6','S3CR6','S3LEB6','S3LEI6','S3PEIB6','S3PEII6']
SPSS6.set_index(['RatID6'], inplace= True)
SPSS6 = SPSS6.sort_values(['RatID6'],ascending=True)

# Combining them in 1 dataframe
SPSS= pd.concat([SPSS, SPSS1, SPSS2, SPSS3, SPSS4, SPSS5, SPSS6], axis=1)

# Re-order the columns and then delete the useless 'treatment' columns
SPSS = SPSS[['L1B1', 'L1B2', 'L1B3', 'L1B4', 'L1B5', 'L1B6', 
             'L1M1', 'L1M2', 'L1M3', 'L1M4', 'L1M5', 'L1M6', 'L1I1', 'L1I2', 'L1I3', 
             'L1I4', 'L1I5', 'L1I6', 'TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TI1', 
             'TI2', 'TI3', 'TI4', 'TI5', 'TI6', 'TE1', 'TE2', 'TE3', 'TE4', 'TE5', 'TE6',
             'TCB1', 'TCB2', 'TCB3', 'TCB4', 'TCB5', 'TCB6', 'TIR1', 'TIR2', 'TIR3', 'TIR4',
             'TIR5', 'TIR6', 'TII1', 'TII2', 'TII3', 'TII4', 'TII5', 'TII6', 'TCR1', 'TCR2',
             'TCR3', 'TCR4', 'TCR5', 'TCR6', 'S1M1', 'S1M2', 'S1M3', 'S1M4', 'S1M5', 'S1M6', 
             'S1I1', 'S1I2', 'S1I3', 'S1I4', 'S1I5', 'S1I6', 'S1CB1', 'S1CB2', 'S1CB3', 'S1CB4', 
             'S1CB5', 'S1CB6', 'S1IR1', 'S1IR2', 'S1IR3', 'S1IR4', 'S1IR4', 'S1IR6', 'S1II1', 
             'S1II2', 'S1II3', 'S1II4', 'S1II5', 'S1II6', 'S1CR1', 'S1CR2', 'S1CR3', 'S1CR4', 
             'S1CR5', 'S1CR6', 'S1LEB1', 'S1LEB2', 'S1LEB3', 'S1LEB4', 'S1LEB5', 'S1LEB6',
             'S1LEI1', 'S1LEI2', 'S1LEI3', 'S1LEI4', 'S1LEI5', 'S1LEI6', 'S1PEIB1', 'S1PEIB2', 
             'S1PEIB3', 'S1PEIB4', 'S1PEIB5', 'S1PEIB6', 'S1PEII1', 'S1PEII2', 'S1PEII3', 
             'S1PEII4', 'S1PEII5', 'S1PEII6', 'S2M1', 'S2M2', 'S2M3', 'S2M4', 'S2M5', 'S2M6',
             'S2I1', 'S2I2', 'S2I3', 'S2I4', 'S2I5', 'S2I6', 'S2CB1', 'S2CB2', 'S2CB3', 'S2CB4', 
             'S2CB5', 'S2CB6', 'S2IR1', 'S2IR2', 'S2IR3', 'S2IR4', 'S2IR4', 'S2IR6', 'S2II1', 
             'S2II2', 'S2II3', 'S2II4', 'S2II5', 'S2II6', 'S2CR1', 'S2CR2', 'S2CR3', 'S2CR4', 
             'S2CR5', 'S2CR6', 'S2LEB1', 'S2LEB2', 'S2LEB3', 'S2LEB4', 'S2LEB5', 'S2LEB6', 
             'S2LEI1', 'S2LEI2', 'S2LEI3', 'S2LEI4', 'S2LEI5', 'S2LEI6', 'S2PEIB1', 'S2PEIB2', 
             'S2PEIB3', 'S2PEIB4', 'S2PEIB5', 'S2PEIB6', 'S2PEII1', 'S2PEII2', 'S2PEII3', 
             'S2PEII4', 'S2PEII5', 'S2PEII6', 'S3M1', 'S3M2', 'S3M3', 'S3M4', 'S3M5', 'S3M6', 
             'S3I1', 'S3I2', 'S3I3', 'S3I4', 'S3I5', 'S3I6', 'S3CB1', 'S3CB2', 'S3CB3', 'S3CB4',
             'S3CB5', 'S3CB6', 'S3IR1', 'S3IR2', 'S3IR3', 'S3IR4', 'S3IR4', 'S3IR6', 'S3II1', 
             'S3II2', 'S3II3', 'S3II4', 'S3II5', 'S3II6', 'S3CR1', 'S3CR2', 'S3CR3', 'S3CR4', 
             'S3CR5', 'S3CR6', 'S3LEB1', 'S3LEB2', 'S3LEB3', 'S3LEB4', 'S3LEB5', 'S3LEB6', 
             'S3LEI1', 'S3LEI2', 'S3LEI3', 'S3LEI4', 'S3LEI5', 'S3LEI6', 'S3PEIB1', 'S3PEIB2',
             'S3PEIB3', 'S3PEIB4', 'S3PEIB5', 'S3PEIB6', 'S3PEII1', 'S3PEII2', 'S3PEII3', 
             'S3PEII4', 'S3PEII5', 'S3PEII6']]

# Delete the columns without numbers (NaN) 
SPSS = SPSS.dropna(axis=1, how='all')

# Write new data to excel with multiple sheets
writer = pd.ExcelWriter('sex_results_python.xlsx')
data_info.to_excel(writer,'data_info')
results.to_excel(writer,'results')
data_stat_mean.to_excel(writer,'stats')
data_stat_rest.to_excel(writer,'stats_extra')
SPSS.to_excel(writer,'SPSS_within')
data_noindex.to_excel(writer, 'data_noindex')
writer.save()

# Make the plot for L1B
bar_L1B =plt.bar(x3, mean['L1B'], color= 'blue', width=barWidth, edgecolor='white')       
 
# Find highest value
h1 = results['L1B'].max()
highest = (h1+50)

# Add xticks on the middle of the group bars
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to first behavior (s)', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)

#plotting of the scatters
bar_L1B = plt.scatter(results['Treatment']*3-3, results['L1B'], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)

plt.tight_layout()
plt.show()

# Save figure as jpg
bar_L1B.figure.savefig('Bar_L1B.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for L1M
bar_L1M =plt.bar(x3, mean['L1M'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['L1M'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to first mount (s)', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_L1M = plt.scatter(results['Treatment']*3-3, results['L1M'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_L1M.figure.savefig('Bar_L1M.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for L1I
bar_L1I =plt.bar(x3, mean['L1I'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['L1I'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to first intromission (s)', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_L1I = plt.scatter(results['Treatment']*3-3, results['L1I'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_L1I.figure.savefig('Bar_L1I.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for TM
bar_TM =plt.bar(x3, mean['TM'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['TM'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Total mounts', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_TM = plt.scatter(results['Treatment']*3-3, results['TM'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_TM.figure.savefig('Bar_TM.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for TI
bar_TI =plt.bar(x3, mean['TI'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['TI'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Total intromissions', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_TI = plt.scatter(results['Treatment']*3-3, results['TI'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_TI.figure.savefig('Bar_TI.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for TE
bar_TE =plt.bar(x3, mean['TE'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['TE'].max()
highest = (h1+1)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Total ejaculations', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_TE = plt.scatter(results['Treatment']*3-3, results['TE'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_TE.figure.savefig('Bar_TE.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for TCB
bar_TCB =plt.bar(x3, mean['TCB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['TCB'].max()
highest = (h1+10)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Total copulations', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_TCB = plt.scatter(results['Treatment']*3-3, results['TCB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_TCB.figure.savefig('Bar_TCB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for TIR
bar_TIR =plt.bar(x3, mean['TIR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = 1
highest = h1
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Total intromissions ratio', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_TIR = plt.scatter(results['Treatment']*3-3, results['TIR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_TIR.figure.savefig('Bar_TIR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for TII
bar_TII =plt.bar(x3, mean['TII'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['TII'].max()
highest = (h1+10)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Total intromission interval', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_TII = plt.scatter(results['Treatment']*3-3, results['TII'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_TII.figure.savefig('Bar_TII.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for TCR
bar_TCR =plt.bar(x3, mean['TCR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['TCR'].max()
highest = (h1+0.02)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Total copulatory rate', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_TCR = plt.scatter(results['Treatment']*3-3, results['TCR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_TCR.figure.savefig('Bar_TCR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1M
bar_S1M =plt.bar(x3, mean['S1M'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1M'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Mounts in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1M = plt.scatter(results['Treatment']*3-3, results['S1M'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1M.figure.savefig('Bar_S1M.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1I
bar_S1I =plt.bar(x3, mean['S1I'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1I'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromissions in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1I = plt.scatter(results['Treatment']*3-3, results['S1I'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1I.figure.savefig('Bar_S1I.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1CB
bar_S1CB =plt.bar(x3, mean['S1CB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1CB'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Copulations in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1CB = plt.scatter(results['Treatment']*3-3, results['S1CB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1CB.figure.savefig('Bar_S1CB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1IR
bar_S1IR =plt.bar(x3, mean['S1IR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = 1
highest = h1
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromission ratio in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1IR = plt.scatter(results['Treatment']*3-3, results['S1IR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1IR.figure.savefig('Bar_S1IR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1II
bar_S1II =plt.bar(x3, mean['S1II'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1II'].max()
highest = (h1+10)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromission interval in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1II = plt.scatter(results['Treatment']*3-3, results['S1II'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1II.figure.savefig('Bar_S1II.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1CR
bar_S1CR =plt.bar(x3, mean['S1CR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1CR'].max()
highest = (h1+0.02)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Copulatory rate in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1CR = plt.scatter(results['Treatment']*3-3, results['S1CR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1CR.figure.savefig('Bar_S1CR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1LEB
bar_S1LEB =plt.bar(x3, mean['S1LEB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1LEB'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to 1st ejaculation from 1st behavior', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1LEB = plt.scatter(results['Treatment']*3-3, results['S1LEB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1LEB.figure.savefig('Bar_S1LEB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1LEI
bar_S1LEB =plt.bar(x3, mean['S1LEI'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1LEI'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to 1st ejaculation from 1st intromission', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1LEI = plt.scatter(results['Treatment']*3-3, results['S1LEI'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1LEI.figure.savefig('Bar_S1LEI.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1PEIB
bar_S1PEIB =plt.bar(x3, mean['S1PEIB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1PEIB'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Postejaculatory interval to 1st behavior in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1PEIB = plt.scatter(results['Treatment']*3-3, results['S1PEIB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1PEIB.figure.savefig('Bar_S1PEIB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S1PEII
bar_S1PEII =plt.bar(x3, mean['S1PEII'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S1PEII'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Postejaculatory interval to 1st intromission in 1st series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S1PEII = plt.scatter(results['Treatment']*3-3, results['S1PEII'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S1PEII.figure.savefig('Bar_S1PEII.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2M
bar_S2M =plt.bar(x3, mean['S2M'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2M'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Mounts in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2M = plt.scatter(results['Treatment']*3-3, results['S2M'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2M.figure.savefig('Bar_S2M.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2I
bar_S2I =plt.bar(x3, mean['S2I'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2I'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromissions in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2I = plt.scatter(results['Treatment']*3-3, results['S2I'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2I.figure.savefig('Bar_S2I.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2CB
bar_S2CB =plt.bar(x3, mean['S2CB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2CB'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Copulations in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2CB = plt.scatter(results['Treatment']*3-3, results['S2CB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2CB.figure.savefig('Bar_S2CB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2IR
bar_S2IR =plt.bar(x3, mean['S2IR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = 1
highest = h1
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromission ratio in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2IR = plt.scatter(results['Treatment']*3-3, results['S2IR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2IR.figure.savefig('Bar_S2IR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2II
bar_S2II =plt.bar(x3, mean['S2II'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2II'].max()
highest = (h1+10)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromission interval in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2II = plt.scatter(results['Treatment']*3-3, results['S2II'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2II.figure.savefig('Bar_S2II.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2CR
bar_S2CR =plt.bar(x3, mean['S2CR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2CR'].max()
highest = (h1+0.02)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Copulatory rate in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2CR = plt.scatter(results['Treatment']*3-3, results['S2CR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2CR.figure.savefig('Bar_S2CR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2LEB
bar_S2LEB =plt.bar(x3, mean['S2LEB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2LEB'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to 2nd ejaculation from 1st behavior', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2LEB = plt.scatter(results['Treatment']*3-3, results['S2LEB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2LEB.figure.savefig('Bar_S2LEB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2LEI
bar_S2LEB =plt.bar(x3, mean['S2LEI'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2LEI'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to 2nd ejaculation from 1st intromission', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2LEI = plt.scatter(results['Treatment']*3-3, results['S2LEI'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2LEI.figure.savefig('Bar_S2LEI.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2PEIB
bar_S2PEIB =plt.bar(x3, mean['S2PEIB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2PEIB'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Postejaculatory interval to 1st behavior in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2PEIB = plt.scatter(results['Treatment']*3-3, results['S2PEIB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2PEIB.figure.savefig('Bar_S2PEIB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S2PEII
bar_S2PEII =plt.bar(x3, mean['S2PEII'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S2PEII'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Postejaculatory interval to 1st intromission in 2nd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S2PEII = plt.scatter(results['Treatment']*3-3, results['S2PEII'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S2PEII.figure.savefig('Bar_S2PEII.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3M
bar_S3M =plt.bar(x3, mean['S3M'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3M'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Mounts in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3M = plt.scatter(results['Treatment']*3-3, results['S3M'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3M.figure.savefig('Bar_S3M.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3I
bar_S3I =plt.bar(x3, mean['S3I'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3I'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromissions in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3I = plt.scatter(results['Treatment']*3-3, results['S3I'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3I.figure.savefig('Bar_S3I.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3CB
bar_S3CB =plt.bar(x3, mean['S3CB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3CB'].max()
highest = (h1+5)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Copulations in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3CB = plt.scatter(results['Treatment']*3-3, results['S3CB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3CB.figure.savefig('Bar_S3CB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3IR
bar_S3IR =plt.bar(x3, mean['S3IR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = 1
highest = h1
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromission ratio in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3IR = plt.scatter(results['Treatment']*3-3, results['S3IR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3IR.figure.savefig('Bar_S3IR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3II
bar_S3II =plt.bar(x3, mean['S3II'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3II'].max()
highest = (h1+10)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Intromission interval in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3II = plt.scatter(results['Treatment']*3-3, results['S3II'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3II.figure.savefig('Bar_S3II.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3CR
bar_S3CR =plt.bar(x3, mean['S3CR'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3CR'].max()
highest = (h1+0.02)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Copulatory rate in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3CR = plt.scatter(results['Treatment']*3-3, results['S3CR'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3CR.figure.savefig('Bar_S3CR.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3LEB
bar_S3LEB =plt.bar(x3, mean['S3LEB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3LEB'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to 3rd ejaculation from 1st behavior', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3LEB = plt.scatter(results['Treatment']*3-3, results['S3LEB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3LEB.figure.savefig('Bar_S3LEB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3LEI
bar_S3LEB =plt.bar(x3, mean['S3LEI'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3LEI'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Latency to 3rd ejaculation from 1st intromission', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3LEI = plt.scatter(results['Treatment']*3-3, results['S3LEI'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3LEI.figure.savefig('Bar_S3LEI.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3PEIB
bar_S3PEIB =plt.bar(x3, mean['S3PEIB'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3PEIB'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Postejaculatory interval to 1st behavior in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3PEIB = plt.scatter(results['Treatment']*3-3, results['S3PEIB'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3PEIB.figure.savefig('Bar_S3PEIB.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for S3PEII
bar_S3PEII =plt.bar(x3, mean['S3PEII'], color= 'blue', width=barWidth, edgecolor='white')       
h1 = results['S3PEII'].max()
highest = (h1+50)
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Postejaculatory interval to 1st intromission in 3rd series', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)
bar_S3PEII = plt.scatter(results['Treatment']*3-3, results['S3PEII'], facecolors=['black'], 
                      edgecolors='none',s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()
bar_S3PEII.figure.savefig('Bar_S3PEII.jpg', dpi=300, figsize=(16.0, 10.0))

# Save all figures in one pdf file
# Create an imagelist that contains a list with all image filenames
im1 = Image.open('Bar_L1B.jpg')
im2 = Image.open('Bar_L1M.jpg')
im3 = Image.open('Bar_L1I.jpg')
im4 = Image.open('Bar_TM.jpg')
im5 = Image.open('Bar_TI.jpg')
im6 = Image.open('Bar_TE.jpg')
im7 = Image.open('Bar_TCB.jpg')
im8 = Image.open('Bar_TIR.jpg')
im9 = Image.open('Bar_TII.jpg')
im10 = Image.open('Bar_TCR.jpg')
im11 = Image.open('Bar_S1M.jpg')
im12 = Image.open('Bar_S1I.jpg')
im13 = Image.open('Bar_S1CB.jpg')
im14 = Image.open('Bar_S1IR.jpg')
im15 = Image.open('Bar_S1II.jpg')
im16 = Image.open('Bar_S1CR.jpg')
im17 = Image.open('Bar_S1LEB.jpg')
im18 = Image.open('Bar_S1LEI.jpg')
im19 = Image.open('Bar_S1PEIB.jpg')
im20 = Image.open('Bar_S1PEII.jpg')
im21 = Image.open('Bar_S2M.jpg')
im22 = Image.open('Bar_S2I.jpg')
im23 = Image.open('Bar_S2CB.jpg')
im24 = Image.open('Bar_S2IR.jpg')
im25 = Image.open('Bar_S2II.jpg')
im26 = Image.open('Bar_S2CR.jpg')
im27 = Image.open('Bar_S2LEB.jpg')
im28 = Image.open('Bar_S2LEI.jpg')
im29 = Image.open('Bar_S2PEIB.jpg')
im30 = Image.open('Bar_S2PEII.jpg')
im31 = Image.open('Bar_S3M.jpg')
im32 = Image.open('Bar_S3I.jpg')
im33 = Image.open('Bar_S3CB.jpg')
im34 = Image.open('Bar_S3IR.jpg')
im35 = Image.open('Bar_S3II.jpg')
im36 = Image.open('Bar_S3CR.jpg')
im37 = Image.open('Bar_S3LEB.jpg')
im38 = Image.open('Bar_S3LEI.jpg')
im39 = Image.open('Bar_S3PEIB.jpg')
im40 = Image.open('Bar_S3PEII.jpg')

imagelistS1 = [im2, im3, im4, im5, im6, im7, im8, im9, im10, im11,im12,im13,im14,
              im15,im16,im17,im18,im19,im20]
imagelistS2 = [im22,im23,im24,im25,im26,im27,im28,im29,im30]
imagelistS3 = [im32,im33,im34,im35,im36,im37,im38,im39,im40]

# Create filename for saving
pdf1_filename = "Figures_S1.pdf"
pdf2_filename = "Figures_S2.pdf"
pdf3_filename = "Figures_S3.pdf"

# Save the figures in one pfd
im1.save(pdf1_filename, "PDF" ,resolution=300.0, save_all=True, append_images=imagelistS1)
im21.save(pdf2_filename, "PDF" ,resolution=300.0, save_all=True, append_images=imagelistS2)
im31.save(pdf3_filename, "PDF" ,resolution=300.0, save_all=True, append_images=imagelistS3)