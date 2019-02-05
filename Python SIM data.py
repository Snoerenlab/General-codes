# -*- coding: utf-8 -*-
"""
Created on Fri Jan  18 22:51:03 2019

@author: Eelke Snoeren

THS PYTHON SCRIPT WILL RUN YOUR RAW DATA FILE FROM ETHOVISION (IF SAVED AS CSV
COMMA SEPARATED) AND MAKE IT AUTOMATICALLY IN GRAPHS AND A NEW DATA-EXCEL SHEET

TO DO BEFOREHAND
1) CHANGE THE PATH OF PYTHON TO YOUR DATA FOLDER
2) CHANGE THE FILENAME TO THE RIGHT DOCUMENT
3) CHECK WHETHER A-Q MATCHES YOUR COLUMNS
4) FILL IN TREATMENT GROUPS
5) MATCH X-AX SCALE TO NUMBER OF TREATMENT GROUPS
6) FILL SPSS FILES WHEN NUMBER OF TREATMENT IS HIGHER
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import chain
sns.set()
from PIL import Image

# Assign spreadsheet filename to `file`
file = 'TestSIMfile2.csv'

# Load and clean up of the data file for DataFrames
dataraw = pd.read_csv(file, header=3, sep= ";")

# Fill out your short column names behind the definition a-z
A='Result'
B='Trial'
C='RatID'
D='Treatment'
E='Stimulusbox'
F='Distance'
G='unimportant'
H='unimportant'
I='unimportant'
J='Time_zone1'
K='Visit_zone1'
L='Latency_zone1'
M='Time_zone2'
N='Visit_zone2'
O='Latency_zone2' 
P='Movement' 
Q='Velocity'
R='TimeC'
S='TimeS'
T='PS'
U='VisitC'
V='VisitS'
W='LatencyC'
X='LatencyS'

# Fill out your titles for columns behind definition TA-TZ
TD= 'Treatment'
TF= 'Distance moved (m)'
TP= 'Time spent moving (s)'
TQ= 'Velocity (m/s)'
TR= 'Time spent near control (s)'
TS= 'Time spent near stimulus (s)'
TT= 'Preference Score'
TU= 'Visits to control'
TV= 'Visits to stimulus'
TW= 'Latency to visit control (s)'
TX= 'Latency to visit stimulus (s)'

# Fill out your treatment/stimulus behind definition SA-SZ
SA='Control'
SB='Stimulus1'
SC='Stimulus2'
SD='Stimulus3'
SE='Stimulus4'
SF='Stimulus5' 

Stimuli_values= (SA,SB,SC,SD,SE,SF)

# Set position of bar on X axis - MAKE SURE IT MATCHES YOUR NUMBER OF GROUPS
# set width of bar
barWidth = 2
x1 = [0, 5, 10, 15, 20] 
x2 = [x + barWidth for x in x1]
x3 = [0, 3, 6, 9, 12]

# Rename columns (add or remove letters according to number of columns)
dataraw.columns = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q]

# Make a new datafile with selected columns
dataraw2=dataraw[[C,D,E,F,J,K,L,M,N,O,P,Q]]
#data.set_index([C,D], inplace=True)

## Create dataframes for Time, Visits, and Latency for stimulus and control 
TimeS = pd.DataFrame(np.where(dataraw2[E]==1, dataraw2[J], dataraw2[M]), copy = True)
TimeC = pd.DataFrame(np.where(dataraw2[E]==1, dataraw2[M], dataraw2[J]), copy = True)
VisitS = pd.DataFrame(np.where(dataraw2[E]==1, dataraw2[K], dataraw2[N]), copy = True)
VisitC = pd.DataFrame(np.where(dataraw2[E]==1, dataraw2[N], dataraw2[K]), copy = True)
LatencyS = pd.DataFrame(np.where(dataraw2[E]==1, dataraw2[L], dataraw2[O]), copy = True)
LatencyC = pd.DataFrame(np.where(dataraw2[E]==1, dataraw2[O], dataraw2[L]), copy = True)
PS = pd.DataFrame((TimeS/(TimeS+TimeC)), copy = True)

# Add the dataframes to data
dataraw2 = pd.concat([dataraw2, TimeC, TimeS, PS, VisitC, VisitS, LatencyC, LatencyS], sort=False, axis=1)
# Assign column-names
dataraw2.columns = [C,D,E,F,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X]
# Select columns and set index to ratID and treatment in final data
data= dataraw2[[C,D,R,S,T,U,V,W,X,F,P,Q]]
data_noindex=data.copy()
data.set_index([C,D], inplace=True)

## See the data cleaned up
#data.info()
#data_select.info()

## See some columns in the data
#print(data[[H,I]])

# Make an empty DataFrame for statistical output
data_stat=pd.DataFrame()

# Statistics on the data
mean=data.groupby(D)[R,S,T,U,V,W,X,F,P,Q].mean()
median=data.groupby(D)[R,S,T,U,V,W,X,F,P,Q].median()
std=data.groupby(D)[R,S,T,U,V,W,X,F,P,Q].std()
sem=data.groupby(D)[R,S,T,U,V,W,X,F,P,Q].sem()
var=data.groupby(D)[R,S,T,U,V,W,X,F,P,Q].var()
q25=data.groupby(D)[R,S,T,U,V,W,X,F,P,Q].quantile(q=0.25, axis=0)
q75=data.groupby(D)[R,S,T,U,V,W,X,F,P,Q].quantile(q=0.75, axis=0)

# Calculate n per group and squareroot for sem median
npg=data.groupby(D).size()
sqrtn=np.sqrt(npg)*1.34

# Calculate standard error of median
semedianR = pd.DataFrame(((q75[R]-q25[R])/sqrtn), copy=True)
semedianS = pd.DataFrame(((q75[S]-q25[S])/sqrtn), copy=True)
semedianT = pd.DataFrame(((q75[T]-q25[T])/sqrtn), copy=True)
semedianU = pd.DataFrame(((q75[U]-q25[U])/sqrtn), copy=True)
semedianV = pd.DataFrame(((q75[V]-q25[V])/sqrtn), copy=True)
semedianW = pd.DataFrame(((q75[W]-q25[W])/sqrtn), copy=True)
semedianX = pd.DataFrame(((q75[X]-q25[X])/sqrtn), copy=True)
semedianF = pd.DataFrame(((q75[F]-q25[F])/sqrtn), copy=True)
semedianP = pd.DataFrame(((q75[P]-q25[P])/sqrtn), copy=True)
semedianQ = pd.DataFrame(((q75[Q]-q25[Q])/sqrtn), copy=True)
semedian = pd.concat([semedianR, semedianS, semedianT, semedianU, semedianV, semedianW, semedianX,
                     semedianF, semedianP, semedianQ], sort=False, axis=1)
         
# Write mean statistics to the new dataframe data_stat
data_stat_mean= pd.concat([data_stat, mean, sem], sort=False, axis=1)
data_stat_rest= pd.concat([data_stat, median, semedian, q25, q75, std, var], sort=False, axis=1)

# Rename the column names
data_stat_mean.columns=['mean_%s'%R, 'mean_%s'%S,'mean_%s'%T,'mean_%s'%U,
                           'mean_%s'%V,'mean_%s'%W,'mean_%s'%X,'mean_%s'%F,
                           'mean_%s'%P,'mean_%s'%Q, 'sem_%s'%R, 'sem_%s'%S,
                           'sem_%s'%T,'sem_%s'%U,'sem_%s'%V, 'sem_%s'%W,
                           'sem_%s'%X,'sem_%s'%F,'sem_%s'%P,
                           'sem_%s'%Q]
data_stat_rest.columns=['median_%s'%R, 'median_%s'%S,'median_%s'%T,'median_%s'%U,
                           'median_%s'%V,'median_%s'%W, 'median_%s'%X,'median_%s'%F,
                           'median_%s'%P,'median_%s'%Q,'semedian_%s'%R, 'semedian_%s'%S,
                           'semedian_%s'%T,'semedian_%s'%U,'semedian_%s'%V,'semedian_%s'%W,
                           'semedian_%s'%X,'semedian_%s'%F,'semedian_%s'%P,'semedian_%s'%Q,
                           'q25_%s'%R, 'q25_%s'%S,'q25_%s'%T,'q25_%s'%U,'q25_%s'%V,
                           'q25_%s'%W, 'q25_%s'%X,'q25_%s'%F,'q25_%s'%P,'q25_%s'%Q,
                           'q75_%s'%R, 'q75_%s'%S,'q75_%s'%T,'q75_%s'%U,'q75_%s'%V,
                           'q75_%s'%W, 'q75_%s'%X,'q75_%s'%F,'q75_%s'%P,'q75_%s'%Q,
                           'std_%s'%R, 'std_%s'%S,'std_%s'%T,'std_%s'%U, 'std_%s'%V,
                           'std_%s'%W, 'std_%s'%X,'std_%s'%F,'std_%s'%P,'std_%s'%Q, 
                           'var_%s'%R, 'var_%s'%S,'var_%s'%T,'var_%s'%U, 'var_%s'%V,
                           'var_%s'%W, 'var_%s'%X,'var_%s'%F, 'var_%s'%P,'var_%s'%Q]
                           
# Re-order the columns
data_stat_mean = data_stat_mean[['mean_TimeC','sem_TimeC', 'mean_TimeS',
                                       'sem_TimeS','mean_PS', 'sem_PS',
                                       'mean_VisitC','sem_VisitC', 'mean_VisitS',
                                       'sem_VisitS','mean_LatencyC','sem_LatencyC',
                                       'mean_LatencyS','sem_LatencyS','mean_Distance',
                                       'sem_Distance','mean_Movement', 'sem_Movement',
                                       'mean_Velocity','sem_Velocity']]
data_stat_rest = data_stat_rest[['median_TimeC', 'semedian_TimeC','median_TimeS', 'semedian_TimeS', 
                                 'median_PS', 'semedian_PS','median_VisitC', 'semedian_VisitC', 
                                 'median_VisitS', 'semedian_VisitS', 'median_LatencyC',
                                 'semedian_LatencyC', 'median_LatencyS', 'semedian_LatencyS',
                                 'median_Distance','semedian_Distance','median_Movement',
                                 'semedian_Movement', 'median_Velocity', 'semedian_Velocity',
                                 'std_TimeC', 'std_TimeS','std_PS','std_VisitC', 'std_VisitS', 
                                 'std_LatencyC','std_LatencyS', 'std_Distance','std_Movement',                    
                                 'std_Velocity', 'var_TimeC', 'var_TimeS', 'var_PS','var_VisitC', 
                                 'var_VisitS','var_LatencyC', 'var_LatencyS', 'var_Distance',
                                 'var_Movement', 'var_Velocity','q25_TimeC', 'q25_TimeS',
                                 'q25_PS','q25_VisitC', 'q25_VisitS', 'q25_LatencyC',
                                 'q25_LatencyS', 'q25_Distance','q25_Movement',                    
                                 'q25_Velocity','q75_TimeC', 'q75_TimeS',
                                 'q75_PS','q75_VisitC', 'q75_VisitS', 'q75_LatencyC',
                                 'q75_LatencyS', 'q75_Distance','q75_Movement',                    
                                 'q75_Velocity']]
#data_stat_mean.info()

# Create dataframes for SPSS excelsheet
SPSS=pd.DataFrame()
SPSS1 = data_noindex.loc[data_noindex[D] == 1]
SPSS1.columns=['RatID1', 'Treatment1', 'TimeC1', 'TimeS1', 'PS1', 'VisitC1', 'VisitS1',
               'LatencyC1', 'LatencyS1', 'Distance1', 'Movement1', 'Velocity1']
SPSS1.set_index(['RatID1'], inplace= True)
SPSS1 = SPSS1.sort_values(['RatID1'],ascending=True)

SPSS2 = data_noindex.loc[data_noindex[D] == 2]
SPSS2.columns=['RatID2', 'Treatment2', 'TimeC2', 'TimeS2', 'PS2', 'VisitC2', 'VisitS2',
               'LatencyC2', 'LatencyS2', 'Distance2', 'Movement2', 'Velocity2']
SPSS2.set_index(['RatID2'], inplace= True)
SPSS2 = SPSS2.sort_values(['RatID2'],ascending=True)

SPSS3 = data_noindex.loc[data_noindex[D] == 3]
SPSS3.columns=['RatID3', 'Treatment3', 'TimeC3', 'TimeS3', 'PS3', 'VisitC3', 'VisitS3',
               'LatencyC3', 'LatencyS3', 'Distance3', 'Movement3', 'Velocity3']
SPSS3.set_index(['RatID3'], inplace= True)
SPSS3 = SPSS3.sort_values(['RatID3'],ascending=True)

SPSS4 = data_noindex.loc[data_noindex[D] == 4]
SPSS4.columns=['RatID4', 'Treatment4', 'TimeC4', 'TimeS4', 'PS4', 'VisitC4', 'VisitS4',
               'LatencyC4', 'LatencyS4', 'Distance4', 'Movement4', 'Velocity4']
SPSS4.set_index(['RatID4'], inplace= True)
SPSS4 = SPSS4.sort_values(['RatID4'],ascending=True)

SPSS5 = data_noindex.loc[data_noindex[D] == 5]
SPSS5.columns=['RatID5', 'Treatment5', 'TimeC5', 'TimeS5', 'PS5', 'VisitC5', 'VisitS5',
               'LatencyC5', 'LatencyS5', 'Distance5', 'Movement5', 'Velocity5']
SPSS5.set_index(['RatID5'], inplace= True)
SPSS5 = SPSS5.sort_values(['RatID5'],ascending=True)

SPSS6 = data_noindex.loc[data_noindex[D] == 6]
SPSS6.columns=['RatID6', 'Treatment6', 'TimeC6', 'TimeS6', 'PS6', 'VisitC6', 'VisitS6',
               'LatencyC6', 'LatencyS6', 'Distance6', 'Movement6', 'Velocity6']
SPSS6.set_index(['RatID6'], inplace= True)
SPSS6 = SPSS6.sort_values(['RatID6'],ascending=True)

# Combining them in 1 dataframe
SPSS= pd.concat([SPSS, SPSS1, SPSS2, SPSS3, SPSS4, SPSS5, SPSS6], axis=1, join_axes=[SPSS1.index])

# Re-order the columns and then delete the useless 'treatment' columns
SPSS = SPSS[['Treatment1', 'Treatment2', 'Treatment3', 'Treatment4', 'Treatment5', 'Treatment6',
             'TimeC1', 'TimeC2', 'TimeC3', 'TimeC4', 'TimeC5', 'TimeC6', 'TimeS1', 'TimeS2',
             'TimeS3', 'TimeS4', 'TimeS5', 'TimeS6', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',
             'VisitC1', 'VisitC2', 'VisitC3', 'VisitC4','VisitC5','VisitC6','VisitS1', 
             'VisitS2', 'VisitS3', 'VisitS4', 'VisitS5', 'VisitS6', 'LatencyC1', 'LatencyC2',
             'LatencyC3', 'LatencyC4', 'LatencyC5', 'LatencyC6','LatencyS1', 'LatencyS2',
             'LatencyS3', 'LatencyS4', 'LatencyS5', 'LatencyS6', 'Distance1', 'Distance2',
             'Distance3', 'Distance4', 'Distance5', 'Distance6','Movement1', 'Movement2',
             'Movement3', 'Movement4', 'Movement5', 'Movement6','Velocity1', 'Velocity2',
             'Velocity3','Velocity4', 'Velocity5', 'Velocity6']]
SPSS = SPSS[['TimeC1', 'TimeC2', 'TimeC3', 'TimeC4', 'TimeC5', 'TimeC6', 'TimeS1', 'TimeS2',
             'TimeS3', 'TimeS4', 'TimeS5', 'TimeS6', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',
             'VisitC1', 'VisitC2', 'VisitC3', 'VisitC4','VisitC5','VisitC6','VisitS1', 
             'VisitS2', 'VisitS3', 'VisitS4', 'VisitS5', 'VisitS6', 'LatencyC1', 'LatencyC2',
             'LatencyC3', 'LatencyC4', 'LatencyC5', 'LatencyC6','LatencyS1', 'LatencyS2',
             'LatencyS3', 'LatencyS4', 'LatencyS5', 'LatencyS6', 'Distance1', 'Distance2',
             'Distance3', 'Distance4', 'Distance5', 'Distance6','Movement1', 'Movement2',
             'Movement3', 'Movement4', 'Movement5', 'Movement6','Velocity1', 'Velocity2',
             'Velocity3','Velocity4', 'Velocity5', 'Velocity6']]

# Delete the columns without numbers (NaN) 
SPSS = SPSS.dropna(axis=1, how='all')


## Make numpy object array from DataFrames - not needed but code works
#np_data_stat_mean=data_stat_mean.values
#np_data_stat_rest=data_stat_mean.values
#np_data=data.values

# Write new data to excel with multiple sheets
writer = pd.ExcelWriter('results_python.xlsx')
data.to_excel(writer,'data')
data_stat_mean.to_excel(writer,'Stat_mean')
data_stat_rest.to_excel(writer,'Stat_rest')
SPSS.to_excel(writer,'SPSS_repeatedmeasures')
writer.save()

# Plot the data in bar charts with individual datapoints
# Make the plot for time
bar_time =plt.bar(x1, mean[R], color='grey', width=barWidth, edgecolor='white', label='Control', 
            yerr=sem[R])
bar_time =plt.bar(x2, mean[S], color='blue', width=barWidth, edgecolor='white', label='Stimulus', 
            yerr=sem[S])
# Find highest value
h1 = dataraw2[R].max()
h2 = dataraw2[S].max()

if h1 > h2:
    highest = (h1 + 50)
else:
    highest = (h2 + 50)

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth/2 for r in x1], Stimuli_values, rotation='vertical')
plt.title('Time spent in incentive zone (s)', fontweight = 'bold', fontsize = 16)
plt.legend()
plt.ylim(bottom=0, top= highest)

#plotting of the scatters
bar_time = plt.scatter(dataraw2[D]*5-5, dataraw2[R], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)
bar_time = plt.scatter(dataraw2[D]*5-5+barWidth, dataraw2[S], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()

# Save figure as jpg
bar_time.figure.savefig('Bar_time.jpg', dpi=300, figsize=(16.0, 10.0))
plt.close('Bar_time.jpg')

# Make the plot for visits
bar_visit =plt.bar(x1, mean[U], color='grey', width=barWidth, edgecolor='white', label='Control', 
            yerr=sem[U])
bar_visit =plt.bar(x2, mean[V], color='blue', width=barWidth, edgecolor='white', label='Stimulus', 
            yerr=sem[V])

# Find highest value
h1 = dataraw2[U].max()
h2 = dataraw2[V].max()

if h1 > h2:
    highest = (h1 + 10)
else:
    highest = (h2 + 10)
    
# Add xticks on the middle of the group bars
plt.xticks([r + barWidth/2 for r in x1], Stimuli_values, rotation='vertical')
plt.title('Visits to incentive zone', fontweight = 'bold', fontsize = 16)
plt.legend()
plt.ylim(bottom=0, top= highest)

#plotting of the scatters
bar_visit = plt.scatter(dataraw2[D]*5-5, dataraw2[U], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)
bar_visit = plt.scatter(dataraw2[D]*5-5+barWidth, dataraw2[V], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()

# Save figure as jpg
bar_visit.figure.savefig('Bar_visit.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for latency
bar_latency =plt.bar(x1, mean[W], color='grey', width=barWidth, edgecolor='white', label='Control', 
            yerr=sem[W])
bar_latency =plt.bar(x2, mean[X], color='blue', width=barWidth, edgecolor='white', label='Stimulus', 
            yerr=sem[X])

# Find highest value
h1 = dataraw2[W].max()
h2 = dataraw2[X].max()

if h1 > h2:
    highest = (h1 + 50)
else:
    highest = (h2 + 50)
    
# Add xticks on the middle of the group bars
plt.xticks([r + barWidth/2 for r in x1], Stimuli_values, rotation='vertical')
plt.title('Latency to incentive zone', fontweight = 'bold', fontsize = 16)
plt.legend()
plt.ylim(bottom=0, top= highest)

#plotting of the scatters
bar_latency = plt.scatter(dataraw2[D]*5-5, dataraw2[W], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)
bar_latency = plt.scatter(dataraw2[D]*5-5+barWidth, dataraw2[X], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)
plt.tight_layout()
plt.show()

# Save figure as jpg
bar_latency.figure.savefig('Bar_latency.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for PS
bar_PS =plt.bar(x3, mean[T]-0.5, color= 'blue', width=barWidth, edgecolor='white', label='Control', 
            yerr=sem[T])       
 
# Add xticks on the middle of the group bars
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Preference Score', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=-0.5, top= 0.6)
plt.yticks(ticks=[-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
           labels=['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
plt.axhline(y=0, linewidth=0.8, color='black')

#plotting of the scatters
bar_PS = plt.scatter(dataraw2[D]*3-3, dataraw2[T]-0.5, facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)

plt.tight_layout()
plt.show()

# Save figure as jpg
bar_PS.figure.savefig('Bar_PS.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for Movement
bar_movement =plt.bar(x3, mean[P], color= 'brown', width=barWidth, edgecolor='white', label='Control', 
            yerr=sem[P])       
 
# Find highest value
h1 = dataraw2[P].max()
highest = (h1+50)

# Add xticks on the middle of the group bars
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Time spent moving (s)', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)

#plotting of the scatters
bar_movement = plt.scatter(dataraw2[D]*3-3, dataraw2[P], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)

plt.tight_layout()
plt.show()

# Save figure as jpg
bar_movement.figure.savefig('Bar_movement.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for velocity
bar_velocity =plt.bar(x3, mean[Q], color= 'brown', width=barWidth, edgecolor='white', label='Control', 
            yerr=sem[Q])       
 
# Find highest value
h1 = dataraw2[Q].max()
highest = (h1+5)

# Add xticks on the middle of the group bars
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Velocity (m/s)', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)

#plotting of the scatters
bar_velocity = plt.scatter(dataraw2[D]*3-3, dataraw2[Q], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)

plt.tight_layout()
plt.show()

# Save figure as jpg
bar_velocity.figure.savefig('Bar_velocity.jpg', dpi=300, figsize=(16.0, 10.0))

# Make the plot for distance moved
bar_distance =plt.bar(x3, mean[F], color= 'brown', width=barWidth, edgecolor='white', label='Control', 
            yerr=sem[F])       
 
# Find highest value
h1 = dataraw2[F].max()
highest = (h1+200)

# Add xticks on the middle of the group bars
plt.xticks([r + 0 for r in x3], Stimuli_values, rotation='vertical')
plt.title('Distance moved (m)', fontweight = 'bold', fontsize = 16)
plt.ylim(bottom=0, top= highest)

#plotting of the scatters
bar_distance = plt.scatter(dataraw2[D]*3-3, dataraw2[F], facecolors=['black'], edgecolors='none',  
            s=10, alpha=1, linewidth=1, zorder=20)

plt.tight_layout()
plt.show()

# Save figure as jpg
bar_distance.figure.savefig('Bar_distance.jpg', dpi=300, figsize=(16.0, 10.0))

# Save all figures in one pdf file
# Create an imagelist that contains a list with all image filenames
im1 = Image.open('Bar_time.jpg')
im2 = Image.open('Bar_PS.jpg')
im3 = Image.open('Bar_visit.jpg')
im4 = Image.open('Bar_latency.jpg')
im5 = Image.open('Bar_movement.jpg')
im6 = Image.open('Bar_velocity.jpg')
im7 = Image.open('Bar_distance.jpg')
imagelist = [im2, im3, im4, im5, im6, im7]

# Create filename for saving
pdf1_filename = "Figures_result.pdf"
# Save the figures in one pfd
im1.save(pdf1_filename, "PDF" ,resolution=300.0, save_all=True, append_images=imagelist)

# Set up the axes for histograms
Histograms,((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(figsize =(15,20),
            ncols=2, nrows=4)

#Adjust the lay-out
Histograms.subplots_adjust(wspace=0.2, hspace=0.4)

# Take out the ax-figure that does not exist
ax8.axis('off')

# Plot the data of all columns in histograms one by one in nice format
# Time spent
ax1.hist(x=data[R], bins=30, color ='grey', alpha=0.7, rwidth=0.85, label=R)
ax1.hist(x=data[S], bins=30, color ='blue', alpha=0.7, rwidth=0.85, label=S)
ax1.set_title('Time spent near control versus stimulus', fontsize=16)
ax1.set_ylabel('Frequency', fontsize=16)
ax1.legend(loc='upper right')

#Histograms.savefig('Testfig.pdf')
# Preference score
ax2.hist(x=data[T], bins=30, color ='blue', alpha=0.7, rwidth=0.85, label=T)
ax2.set_title('Preference Score', fontsize=16)
ax2.set_ylabel('Frequency', fontsize=16)

# Visits
ax3.hist(x=data[U], bins=30, color ='grey', alpha=0.7, rwidth=0.85, label=U)
ax3.hist(x=data[V], bins=30, color ='blue', alpha=0.7, rwidth=0.85, label=V)
ax3.set_title('Visits to control versus stimulus', fontsize=16)
ax3.set_ylabel('Frequency', fontsize=16)
ax3.legend(loc='upper right')

# Latency
ax4.hist(x=data[W], bins=30, color ='grey', alpha=0.7, rwidth=0.85, label=W)
ax4.hist(x=data[X], bins=30, color ='blue', alpha=0.7, rwidth=0.85, label=X)
ax4.set_title('Latency to visit control versus stimulus', fontsize=16)
ax4.set_ylabel('Frequency', fontsize=16)
ax4.legend(loc='upper right')

# Movement
ax5.hist(x=data[P], bins=30, color ='brown', alpha=0.7, rwidth=0.85, label=P)
ax5.set_title('Movement', fontsize=16)
ax5.set_ylabel('Frequency', fontsize=16)

# Velocity
ax6.hist(x=data[Q], bins=30, color ='brown', alpha=0.7, rwidth=0.85, label=Q)
ax6.set_title('Velocity', fontsize=16)
ax6.set_ylabel('Frequency', fontsize=16)

# Distance moved
ax7.hist(x=data[F], bins=30, color ='brown', alpha=0.7, rwidth=0.85, label=F)
ax7.set_title('Distance moved', fontsize=16)
ax7.set_ylabel('Frequency', fontsize=16)

Histograms.savefig('Histograms_data.pdf')

