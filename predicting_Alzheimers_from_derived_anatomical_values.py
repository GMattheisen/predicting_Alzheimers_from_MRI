
# coding: utf-8

# # Oasis 3 Textual Data

# In[213]:


import random
import numpy as np
import random
random.seed(50)
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
from sklearn.ensemble import RandomForestClassifier
import scipy
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


# In[214]:


cd '/Users/glynismattheisen/Desktop/Final2'


# In[215]:


df_free = pd.read_csv('norm_vol_final.csv') # import freesurfer data
df_age = pd.read_csv('age.csv') # import age data
df_clin = pd.read_csv('gmattheisen_12_1_2018_15_52_40.csv') # import clinical data (includes diagnosis values)


# ## Adding Age to Freesurfer Data

# In[216]:


def get_SESS_ID(file_name): # parse out session id from MR or FS ID
    return file_name[:9] + file_name[-5:]

df_age['MR'] = df_age['MR ID'].apply(get_SESS_ID) # convert long form MR ID to session id
df_free['MR'] = df_free['FS_FSDATA ID'].apply(get_SESS_ID)# convert long form FS ID to session id


# In[217]:


def replace_null(column): # fill null values with MEAN +/- STD
    mean = column.mean()
    std = column.std()
    is_null = column.isnull().sum()
    rand_VAR = np.random.randint(mean - std, mean + std, size = is_null)
    VAR_slice = column.copy()
    VAR_slice[np.isnan(VAR_slice)] = rand_VAR
    column = VAR_slice
    column = column.astype(int)
    return column


# In[218]:


df_age['Age'] = replace_null(df_age['Age'])
assert df_age["Age"].isnull().sum() ==0  #confirm null values count = 0

del df_age['MR ID']
del df_age['Subject']

df_free.set_index('MR', inplace=True)
df_age.set_index('MR', inplace=True)

df_free = df_free.merge(df_age, left_index=True, right_index=True, how='inner')
df_free.set_index(['FS_FSDATA ID'], inplace=True)


# In[282]:


del df_single['ageAtEntry']


# In[281]:


df_single


# # Diagnosis Labeling

# In[219]:


df_clin['dx1'].fillna('empty', inplace=True) # two null elements are patients with no other clinical diagnostic data
diag_descriptors = df_clin['dx1'].unique() # list of all diagnostic values

diagnosis_dict = {} # create dictionary with key for each diagnostic option
for x in diag_descriptors:
    diagnosis_dict[x] = ''

healthy_diag  = ['Cognitively normal' ,
                 'No dementia', 
                 'uncertain dementia',
                 'Unc: ques. Impairment'] # list of healthy descriptors

alz_diag = ['AD Dementia','AD dem Language dysf prior',
            'AD dem Language dysf with','AD dem w/oth (list B) not contrib',
            'AD dem w/oth unusual features','AD dem visuospatial, after',
           'AD dem w/oth unusual features/demt on',
           'AD dem w/Frontal lobe/demt at onset',
           'AD dem/FLD prior to AD dem',
            'AD dem w/oth unusual feat/subs demt',
            'AD dem w/depresss- not contribut',
            'AD dem distrubed social- with',
            'AD dem distrubed social- prior',
            'AD dem w/CVD not contrib',
            'AD dem w/CVD contribut',
            'AD dem visuospatial- with',
            'AD dem visuospatial- prior',
            'AD dem Language dysf after',
            'AD dem w/oth (list B) contribut',
            'AD dem distrubed social- after',
            'AD dem w/depresss- contribut',
            'AD dem w/depresss  not contribut',
            'AD dem w/depresss  contribut',
            'AD dem w/PDI after AD dem contribut',
            'AD dem w/PDI after AD dem not contrib',
            'AD dem w/depresss, not contribut',
            'DAT w/depresss not contribut',
            'DAT' # dementia alzheimer's type
           ] # list of AD descriptors


misc_diag = ['DLBD, primary',
             'DLBD- primary',
             'DLBD- secondary',
             'Frontotemporal demt. prim',
             'Unc: impair reversible',
             'AD dem cannot be primary',
             'Non AD dem- Other primary',
             'Incipient Non-AD dem',
             'Incipient demt PTP', # mild cognitive impairment
             '.',
             'uncertain- possible NON AD dem',
             'Vascular Demt- primary',
             '0.5 in memory only',
             'Vascular Demt- secondary',
             'Vascular Demt  primary',
             'Dementia/PD- primary',
             'uncertain  possible NON AD dem',
             'empty'
            ] # list of misc descriptors (generally PD or vascular dementia)
             
for x in healthy_diag:
    diagnosis_dict[x] = 0 # assign 0 to keys in diagnosis_dict belonging to healthy descriptors list

for x in alz_diag:
    diagnosis_dict[x] = 1 # assign alzheimer's descriptors diagnosis 1

for x in misc_diag:
    diagnosis_dict[x]= 2 # assign non-healthy, non-alzheimer's diagnoses a 2

diags = [] # use the diagnostic dictionary to create list of all labels for elements in df_clin
for i in df_clin['dx1']:
    diags.append(diagnosis_dict[i])

df_clin['label'] = pd.DataFrame(diags) # add to df_clin as new column 


# ## Create Dataframe of Subjects Whose Diagnosis Changes During Course of Data Collection

# In[220]:


del df_clin['Age'] # mostly null
del df_clin['Date'] # mostly null


# In[221]:


Subjects = df_clin['Subject'].unique() # Determine unique subjects

for subj in range(0,1098):     # find the entries for each subject
    values = np.where(df_clin['Subject'] == Subjects[subj])[0]     # put entries in list
    diag_nums = []   # list of labels for each subject
    for i in values:
        diag_nums.append(df_clin['label'].iloc[i])
    if diag_nums.count(2) != 0 or diag_nums.count(1) != 0: 
        if diag_nums.count(0) == 0:
            if diag_nums.count(1) >= diag_nums.count(2): # if number of alz diagnoses greater than number of misc, diagnose alz
                for i in values:
                    df_clin['label'][i] = 1 # our new 'converted group'
            elif diag_nums.count(2) > diag_nums.count(1): # if number of misc diagnoses greater than number of alz, diagnose misc
                for i in values:
                    df_clin['label'][i] = 2
        elif diag_nums.count(0) != 0:
            if diag_nums.count(2) > diag_nums.count(1): # if number of alz diagnoses greater than number of misc, diagnose alz
                for i in values:
                    df_clin['label'][i] = 2 # our new 'converted group'
            elif diag_nums.count(1) >= diag_nums.count(2): # if number of misc diagnoses greater than number of alz, diagnose misc
                for i in values:
                    df_clin['label'][i] = 3
    
df_clin.to_csv('clinical_w_Conv_diag.csv') # write to file


# ## Make a Dictionary of Subjects and Diagnoses

# In[222]:


diag = [] # make file of just subjects with diagnoses
for subj in range(0,1098):
    values = np.where(df_clin['Subject'] == Subjects[subj])  # find the entries for each subject
    diag.append(df_clin['label'].iloc[values[0][0]])

Subject_dict = {}
for num in range(0,1098):
    Subject_dict[Subjects[num]] = diag[num] # dictionary with subjects as keys and diagnoses as values


# ## Alter Freesurfer File with Diagnostic Data

# In[223]:


diagnosis = [] 
for i in df_free['Subject']:
    diagnosis.append(Subject_dict[i])
    
diagnosis = pd.DataFrame(diagnosis)
df_free['Diagnosis'] = diagnosis.values # add column of diagnoses to freesurfer data

df_free_conv = df_free[df_free.Diagnosis == 3] # include only converted patients


# In[225]:


# create df without converted patients
df_wo_conv = df_free[df_free.Diagnosis != 2] # take out miscellaneous diagnoses
df_wo_conv = df_wo_conv[df_wo_conv.Diagnosis != 2] # take out conv diagnoses


# ## Change Converted Labels to Alzheimer's for Training

# In[12]:


for x,i in enumerate(df_free['Diagnosis']):
    if i == 3:
        df_free['Diagnosis'][x] = 1

df_free = df_free[df_free.Diagnosis != 2] # take out miscellaneous diagnoses


# In[13]:


len(df_free[df_free['Diagnosis'] == 1]), len(df_free[df_free['Diagnosis'] == 0])


# In[14]:


1431-452


# In[15]:


979 % 452


# In[16]:


alz_head = df_free.sort_values(by='Diagnosis', ascending = False).head(452) # all subjects diagnosed as alz


# In[17]:


df_free_bal = df_free.append(alz_head)


# In[18]:


df_free_bal = df_free_bal.append(alz_head)


# In[19]:


alz_head_short = df_free.sort_values(by='Diagnosis', ascending = False).head(75) # all subjects diagnosed as alz


# In[20]:


df_free_bal = df_free_bal.append(alz_head_short)


# In[21]:


assert len(df_free_bal[df_free_bal['Diagnosis'] == 1]) == len(df_free_bal[df_free_bal['Diagnosis'] == 0])


# ## Correlations

# In[168]:


corr = df_free.corr() # what values correlate with diagnosis?
corr


# In[169]:


columns = corr.columns.values


# In[173]:


z = [-0.412148,-0.402058,-0.409204,-0.392236,-0.392708,-0.324860,-0.318144,0.360485,1.000000]

list_n = [item*150 for item in z] 

abs_list = lambda x :abs(x) # normalize x values

for x, i in enumerate(list_n):
    list_n[x] = abs(i)


# In[175]:


import plotly.plotly as py
import plotly.graph_objs as go

z = [-0.412148,-0.402058,-0.409204,-0.392236,-0.392708,-0.324860,-0.318144,0.360485,1.000000]
trace0 = go.Scatter(
    x=[0,0,0,0,0,0,0,0,0],
    y=[0,1,2,3,4,5,6,7,8,9],
    text=columns,
    hoverinfo = 'y',
    textposition=['bottom center','top center','middle center','bottom center','bottom center','middle center','middle center','middle center','middle center'],
    mode='markers + text',
    marker=dict(
        color =[300,325,375,275,325,350,350,350,350],
        opacity=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        size=list_n,
        showscale=False

    )
)

layout = go.Layout(
    xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        range=[-25,25],
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False
    )
)

data = [trace0]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bubblechart-color')


# In[49]:


healthy = 'Cognitively Normal'
dementia = 'Alzheimer\'s Diagnosis'

fig = plt.figure(figsize=(30, 35))
grid = plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)

plt.subplot(grid[0, 0])
ax = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].lhCortexVol_norm_ICV, label = dementia, color='green', shade=True)
ax = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].lhCortexVol_norm_ICV, label = healthy, color ='blue', shade=True)
ax.legend(fontsize=15)
plt.xlabel('Left Hemisphere Cortex Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax.set_title('Diagnosis and Left Hemisphere Cortex Volume', fontsize=24, pad = 20)

plt.subplot(grid[0, 1])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].rhCortexVol_norm_ICV, label = dementia, color='green', shade=True)
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].rhCortexVol_norm_ICV, label = healthy, color='blue', shade=True)
ax1.legend(fontsize=15)
plt.xlabel('Right Hemisphere Cortex Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.set_title('Diagnosis and Right Hemisphere Cortex Volume', fontsize=24, pad = 20)

plt.subplot(grid[1, 0])
ax2 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].TotalGray_norm_ICV, label = dementia, color='green', shade=True)
ax2 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].TotalGray_norm_ICV, label = healthy, color='blue', shade=True)
ax2.legend(fontsize=15)
plt.xlabel('Total Gray Matter', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax2.set_title('Diagnosis and Total Gray Matter', fontsize=24, pad = 20)


plt.subplot(grid[1, 1])
ax = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].SupraTentorialVol_norm_ICV, label = dementia, shade=True, color='green')
ax = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].SupraTentorialVol_norm_ICV, label = healthy, shade=True, color ='blue')
plt.xlabel('SupraTentorial Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax.legend(fontsize=15)
ax.set_title('Diagnosis and SupraTentorial Volume', fontsize=24, pad = 20)

plt.subplot(grid[2, 0])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].lhCorticalWhiteMatterVol_norm_ICV, label = dementia, shade=True,color='green')
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].lhCorticalWhiteMatterVol_norm_ICV, label = healthy, shade=True,color='blue')
plt.xlabel('Left Hemisphere Cortical White Matter Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title('Diagnosis and Left Hemisphere Cortical White Matter Volume', fontsize=24, pad = 20)

plt.subplot(grid[2, 1])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].rhCorticalWhiteMatterVol_norm_ICV, label = dementia, shade=True,color='green')
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].rhCorticalWhiteMatterVol_norm_ICV, label = healthy, shade=True,color='blue')
plt.xlabel('Right Hemisphere Cortical White Matter Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title('Diagnosis and Right Hemisphere Cortical White Matter Volume', fontsize=24, pad = 20)


plt.subplot(grid[3, 0])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].Age, label = dementia, shade=True,color='green')
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].Age, label = healthy, shade=True,color='blue')
plt.xlabel('Age', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title('Diagnosis and Age', fontsize=24, pad = 20)


# In[68]:


healthy = 'Cognitively Normal'
dementia = 'Alzheimer\'s Diagnosis'
left_h = 'Left Hemisphere Healthy'
left_a = 'Left Hemisphere Alzheimer\'s'
right_h = 'Right Hemisphere Healthy'
right_a = 'Right Hemisphere Alzheimer\'s'

fig = plt.figure(figsize=(30, 35))
grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.5)

plt.subplot(grid[0, 0])

ax = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].lhCortexVol_norm_ICV, label = left_h, color =sns.xkcd_rgb["pastel green"], shade=True, legend='lh')
ax = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].rhCortexVol_norm_ICV, label = right_h, color=sns.xkcd_rgb["dark green"], shade=True, legend='rh')

ax = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].lhCortexVol_norm_ICV, label = left_a, color=sns.xkcd_rgb["pastel blue"], shade=True, legend='lh')
ax = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].rhCortexVol_norm_ICV, label = right_a, color=sns.xkcd_rgb["dark blue"], shade=True, legend='rh')
ax.legend(fontsize=15)
plt.xlabel('Hemisphere Cortex Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax.set_title('Diagnosis and Cortex Volume', fontsize=24, pad = 20)


plt.subplot(grid[0, 1])
ax2 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].TotalGray_norm_ICV, label = dementia, color='blue', shade=True)
ax2 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].TotalGray_norm_ICV, label = healthy, color='green', shade=True)
ax2.legend(fontsize=15)
plt.xlabel('Total Gray Matter', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax2.set_title('Diagnosis and Total Gray Matter', fontsize=24, pad = 20)


plt.subplot(grid[1, 0])
ax = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].SupraTentorialVol_norm_ICV, label = dementia, shade=True, color='blue')
ax = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].SupraTentorialVol_norm_ICV, label = healthy, shade=True, color ='green')
plt.xlabel('SupraTentorial Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax.legend(fontsize=15)
ax.set_title('Diagnosis and SupraTentorial Volume', fontsize=24, pad = 20)

plt.subplot(grid[1, 1])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].rhCorticalWhiteMatterVol_norm_ICV, label = right_h, shade=True,color=sns.xkcd_rgb["dark green"])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].lhCorticalWhiteMatterVol_norm_ICV, label = left_h, shade=True,color=sns.xkcd_rgb["pastel green"])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].lhCorticalWhiteMatterVol_norm_ICV, label = left_a, shade=True,color=sns.xkcd_rgb["pastel blue"])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].rhCorticalWhiteMatterVol_norm_ICV, label = right_a, shade=True,color=sns.xkcd_rgb["dark blue"])
plt.xlabel('Hemisphere Cortical White Matter Volume', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title('Diagnosis and Cortical White Matter Volume', fontsize=24, pad = 20)

plt.subplot(grid[2, 0])
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==1.0].Age, label = dementia, shade=True,color='blue')
ax1 = sns.kdeplot(df_free[df_free['Diagnosis']==0.0].Age, label = healthy, shade=True,color='green')
plt.xlabel('Age', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title('Diagnosis and Age', fontsize=24, pad = 20)


# # Model

# In[113]:


def get_model(df):
    RanFor = RandomForestClassifier() # initialize random forest

    x = df.drop('Subject', axis =1) # drop subject for test but maintain in df_free for later analysis
    X = x.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42) # split test and train

    random_trees =  random.sample(range(1,20),10)
    random_depths = random.sample(range(1,50),25)

    grid = GridSearchCV(RanFor, 
            param_grid={'n_estimators': random_trees, 'max_depth': random_depths},
            scoring='accuracy', 
            n_jobs=1,
            cv=5
            )

    grid.fit(X_train,y_train)

    final_model = grid.best_estimator_
    final_model.fit(X_train,y_train)

    print ("Training score: " + str(round(final_model.score(X_train,y_train)*100,2)) + '%')
    print ("Test score: " + str(round(final_model.score(X_test,y_test)*100,2)) + '%')
    print("Cross Validation Scores: " + str(cross_val_score(estimator = final_model, cv = 5, X = X_train, y= y_train)))

    results = final_model.score(X_test, y_test)
    accuracy = round(results*100,2)

    Y_pred = final_model.predict(X_test)

    recall = round((recall_score(y_test, Y_pred))*100, 2)
    precision = round(precision_score(y_test, Y_pred)*100, 2)

    return final_model, accuracy, recall, precision


# In[114]:


all_inst_model, accuracy, recall, precision = get_model(df_free_bal)


# In[115]:


accuracy, precision, recall


# ## Visualize the Decision Tree

# In[97]:


estimator = all_inst_model.estimators_[5]

tree = export_graphviz(estimator, out_file=None, 
                class_names=["Healthy", "Alzheimer's"],
                feature_names=['lhCortexVol_norm_ICV','rhCortexVol_norm_ICV','CortexVol_n_ICV','TotalGray_norm_ICV','SupraTentorialVol_norm_ICV','lhCorticalWhiteMatterVol_norm_ICV','rhCorticalWhiteMatterVol_norm_ICV','Age'],
                impurity=False,
                filled=True)

graph = graphviz.Source(tree)
graph.render('DecTree')  # creates PDF
graph  # in Jupyter


# # Predict First Instances of Converted Subjects

# In[28]:


def get_instance(df_free_conv, instance):
    conv_subj_ids = df_free_conv['Subject'].unique() # list of each converted subject id

    first_instance = [] # list instance for each subject in converted
    for i in conv_subj_ids:
        first_instance.append(np.where(df_free_conv['Subject'] == i)[0][instance]) 

    conv_data = [] # list data for instance of converted subj 
    for i in first_instance:
        conv_data.append(df_free_conv.iloc[i])

    return pd.DataFrame(conv_data), conv_subj_ids


# In[29]:


first_conv_FSdata, conv_subj_ids = get_instance(df_free_conv,instance = 0)


# In[30]:


def prediction_acc(df, model):

    x = df.drop('Subject', axis =1) # create data frame of just first instances FS data
    X = x.drop('Diagnosis', axis =1)

    prediction = model.predict(X) # predict diagnosis of first instances

    counts = Counter(prediction)

    return 'Accuracy of ' + str(round(counts[1] / (counts[1]+counts[0]) *100,2)) + '%'


# In[31]:


prediction_acc(first_conv_FSdata, all_inst_model)


# # Train on Just Last Instance

# In[184]:


conv_subj_ids = df_free_conv['Subject'].unique() # list of each converted subject id

first_instance = [] # list instance for each subject in converted
for i in conv_subj_ids:
    first_instance.append(np.where(df_free_conv['Subject'] == i)[0]) 
    
"""first_instance_2 = []
for i in first_instance:
    for x in i:
        first_instance_2.append(x)

conv_data = [] # list data for instance of converted subj 
for i in first_instance_2:
    conv_data.append(df_free_conv.iloc[i])

last_conv_FSdata = pd.DataFrame(conv_data)"""


# In[204]:


first_instance_2 = []
for i in first_instance:
    if len(i) == 1:
        first_instance_2.append(i)
    else:
        first_instance_2.append(i[1:])
        
first_instance_3 =[]
for i in first_instance_2:
    for x in i:
        first_instance_3.append(x)
        
conv_data = [] # list data for instance of converted subj 
for i in first_instance_3:
    conv_data.append(df_free_conv.iloc[i])

last_conv_FSdata = pd.DataFrame(conv_data)

df_w_last = df_wo_conv.append(last_conv_FSdata) # add back in conv patients

for x,i in enumerate(df_w_last['Diagnosis']): # change converted 3 to alzheimer's 2
    if i == 3:
        df_w_last['Diagnosis'][x] = 1


# In[37]:


"""instances_conv = [] # list indices in full data set that include last instance converted subjects
for i in conv_subj_ids:
    instances_conv.append(np.where(df_free_bal['Subject'] == i)[0])
    
instances_conv_2 = []
for i in instances_conv:
    for x in i:
        instances_conv_2.append(x)"""


# In[238]:


len(df_w_last[df_w_last['Diagnosis'] == 1]), len(df_w_last[df_w_last['Diagnosis'] == 0])


# In[239]:


1431-655


# In[240]:


776/655, 776 % 655


# In[241]:


alz_head = df_w_last.sort_values(by='Diagnosis', ascending = False).head(655) # all subjects diagnosed as alz
alz_head_rem = df_w_last.sort_values(by='Diagnosis', ascending = False).head(131) # all subjects diagnosed as alz

df_w_last = df_w_last.append(alz_head)
df_w_last = df_w_last.append(alz_head_rem)

len(df_w_last[df_w_last['Diagnosis'] == 1]), len(df_w_last[df_w_last['Diagnosis'] == 0])


# In[242]:


last_inst_model, accuracy, recall, precision = get_model(df_w_last)


# In[243]:


prediction_acc(first_conv_FSdata, last_inst_model)


# In[258]:


def color_red(val):
    color = 'red' if val == '97.93' else 'black'
    return 'color: %s' % color


# In[275]:


result_df = pd.DataFrame({
    '': ['Model','Pre-Diagnosis Prediction'],
    'Accuracy': [accuracy, '97.93'],
    'Precision': [precision, '-'],
    'Recall': [recall, '-']
 })
result_df = result_df.set_index('')
result_df.style.background_gradient(cmap='GnBu', low=1, high=0.75, axis=0).set_properties(**{'font-size':'15pt'}).applymap(color_red)

