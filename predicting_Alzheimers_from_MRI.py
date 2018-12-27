
# coding: utf-8

# In[392]:


import random
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from ann_visualizer.visualize import ann_viz;
from sklearn.tree import export_graphviz
import graphviz

from matplotlib.pyplot import imread
import matplotlib
import numpy as np
import os
import imageio
import cv2
from random import seed
seed(42)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import tensorflow
import keras
from keras import optimizers
from keras.models import Sequential, load_model, Model
from keras.callbacks import TensorBoard
from keras.activations import elu
from keras.initializers import he_normal
from keras.preprocessing import image
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D, Dropout, Input
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from numpy.testing import assert_allclose

import nibabel as nib
from nibabel.testing import data_path

from PIL import Image

import nilearn
from nilearn import image, plotting

import ggplot
from ggplot import aes, geom_point, ggtitle


# # Functions

# In[393]:


def get_subject(file_name): # parse out subjectids from file names
    return file_name[0:9]


# In[394]:


def get_slice(): # read in all oasis1 data across discs
    
    path = '/Users/glynismattheisen/Desktop/Final/Discs'
    disc_list = os.listdir(path) # get list of discs from directory
    MacOS_file = '.DS_Store'
    if MacOS_file in disc_list:
        disc_list.remove(MacOS_file) # remove MacOS file

    brain_list, total_subjects = [], [] # generate list of brain images for input to NN, all subjects used in study
    
    for disc in disc_list:
        path_ind_disc = f'{path}/{disc}'
        subject_list_p_disc = os.listdir(path_ind_disc) # generate list of subjects in each disc
        if MacOS_file in subject_list_p_disc:
            subject_list_p_disc.remove(MacOS_file) # remove MacOS file
            
        for subj_id in subject_list_p_disc:
            total_subjects.append(subj_id) # maintain a list of all subjects included in study for diagnosis labeling later
    
            path_n4 = path_ind_disc + f'/{subj_id}/FSL_SEG/{subj_id}_mpr_n4_anon_111_t88_masked_gfc_fseg_tra_90.gif'
            path_n3 = path_ind_disc + f'/{subj_id}/FSL_SEG/{subj_id}_mpr_n3_anon_111_t88_masked_gfc_fseg_tra_90.gif'
            path_n6 = path_ind_disc + f'/{subj_id}/FSL_SEG/{subj_id}_mpr_n6_anon_111_t88_masked_gfc_fseg_tra_90.gif'
            path_n5 = path_ind_disc + f'/{subj_id}/FSL_SEG/{subj_id}_mpr_n5_anon_111_t88_masked_gfc_fseg_tra_90.gif'
            
            path_list = [path_n4,path_n3, path_n6,path_n5]
            for i in path_list:
                if os.path.exists(i) == True:
                    brain_list.append(plt.imread(i)) # append if file format exists

    return disc_list, brain_list, total_subjects


# In[395]:


def get_diagnosis(total_subjects): # builds a dictionary of subjects and diagnoses
    oasis1 = pd.read_csv('/Users/glynismattheisen/Desktop/Final2/oasis_cross-sectional.csv') # read in summary file
    oasis1['CDR'].fillna(0, inplace=True) # null values are healthy diagnoses
    diagnosis_qual={0.:'normal', 0.5:'alz', 1.:'alz', 2.:'alz' } # convert to labels
    oasis1.replace({"CDR": diagnosis_qual}, inplace=True)
    diagnosis_quant={'normal':0,'alz':1} # convert back to numbers
    oasis1.replace({"CDR": diagnosis_quant}, inplace=True)
    
    subjects = []    # get subject names for each MRI ID
    for subj_id in total_subjects:
        subjects.append(get_subject(subj_id))
        
    subjects = pd.DataFrame(subjects, columns = ['Subject']) # convert to dataframe
    oasis1['Subject'] =oasis1['ID'].apply(get_subject)     # extract subject ID from MR ID
    
    diagnosis_dict= {}     # create a dictionary with subject and diagnosis
    for num in range(0,436):
        diagnosis_dict[oasis1['Subject'][num]] = oasis1['CDR'][num]
        
    diag = [] # create a list of diagnoses to append to dataframe of subjects
    for subj in subjects['Subject']:
        diag.append(diagnosis_dict[subj])

    subjects['Diagnosis'] = pd.DataFrame(diag)
    
    return subjects


# In[396]:


def balance_set(subjects, total_subjects, disc_list): # make a list of 236 alz patients in order to even out the lists
    alz_subjects = subjects.sort_values(by='Diagnosis', ascending = False).head(97) # all subjects diagnosed as alz
    alz_subjects = alz_subjects.append(alz_subjects)
    head = alz_subjects.head(42)
    alz_subjects = alz_subjects.append(head)    
    
    for subj in alz_subjects['Subject']:
        total_subjects.append(str(subj) + '_MR1') # maintain a list of all subjects included in study for diagnosis labeling later
    for disc in disc_list:
            path = f'/Users/glynismattheisen/Desktop/Final/Discs/{disc}'

            for i in alz_subjects['Subject']:

                path_n4 = path + f'/{i}_MR1/FSL_SEG/{i}_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg_tra_90.gif'
                path_n3 = path + f'/{i}_MR1/FSL_SEG/{i}_MR1_mpr_n3_anon_111_t88_masked_gfc_fseg_tra_90.gif'
                path_n6 = path + f'/{i}_MR1/FSL_SEG/{i}_MR1_mpr_n6_anon_111_t88_masked_gfc_fseg_tra_90.gif'
                path_n5 = path + f'/{i}_MR1/FSL_SEG/{i}_MR1_mpr_n5_anon_111_t88_masked_gfc_fseg_tra_90.gif'

                path_list = [path_n4,path_n3, path_n6,path_n5]
                for i in path_list:
                    if os.path.exists(i) == True:
                        brain_list.append(plt.imread(i)) # append if file format exists

    brain_array = np.asarray(brain_list)
    return brain_array, alz_subjects


# In[397]:


disc_list, brain_list, total_subjects = get_slice()


# In[398]:


subjects = get_diagnosis(total_subjects)


# In[399]:


brain_array, alz_subjects = balance_set(subjects, total_subjects, disc_list)
subjects = get_diagnosis(total_subjects)


# # Convoluted Neural Network

# In[400]:


x = brain_array # array of image values
y = np.asarray(subjects['Diagnosis'])
x = x.reshape(-1, 208, 176,1)

norm = lambda x :(x - np.min(x)) / (np.max(x) - np.min(x)) # normalize x values
x_norm = norm(x)
assert (np.min(x_norm), np.max(x_norm)) == (0.0, 1.0)

X_im_train, X_im_test, y_im_train, y_im_test = train_test_split(x_norm,y, random_state = 42)


# In[401]:


model = Sequential([
    Conv2D(filters=100,kernel_size=(3,3),strides=(10,10), input_shape=(208,176,1),padding ='same'),
    MaxPooling2D(pool_size=(2,2),padding='valid'),
    Activation('sigmoid'),
    Conv2D(filters=50,kernel_size=(3,3),strides=(5,5),padding ='same'),
    MaxPooling2D(pool_size=(2,2),padding='valid'),
    Activation('sigmoid'),
    Conv2D(filters=25,kernel_size=(3,3),strides=(1,1),padding ='same'),
    MaxPooling2D(pool_size=(1,1),padding='valid'),
    Activation('sigmoid'),
    Flatten(),
    Dense(1),
    Activation('sigmoid'),
])


# In[402]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_im_train,y_im_train,epochs=100, batch_size=75)


# In[403]:


def get_metrics(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    accuracy = round(results[1]*100,2)
    
    Y_pred = model.predict_classes(X_test)
    
    recall = round((recall_score(y_test, Y_pred))*100, 2)
    precision = round(precision_score(y_test, Y_pred)*100, 2)
    
    return accuracy, recall, precision


# In[404]:


model_accuracy, model_recall, model_precision = get_metrics(model,X_im_test, y_im_test)


# # Visualize with Tensorboard

# In[405]:


from keras.callbacks import TensorBoard
tboard = keras.callbacks.TensorBoard(log_dir='/Users/glynismattheisen/Desktop/Final/ouput',histogram_freq=5, write_graph=True, write_images=True)


# In[406]:


input_shape = (208,176,1)

network = Sequential()
network.add(Conv2D(100, kernel_size=(3, 3), strides=(10,10),
                 activation='sigmoid', padding ='valid',
                 input_shape=input_shape))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Conv2D(50, (3, 3), activation='sigmoid', strides=(5,5), padding ='same'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Conv2D(25, kernel_size=(3,3), activation='sigmoid', strides = (1,1),padding ='same'))
network.add(MaxPooling2D(pool_size=(1, 1)))
network.add(Flatten())
network.add(Dense(1, activation='sigmoid', name='preds'))


#from ann_visualizer.visualize import ann_viz;

#ann_viz(network, title="Neural Network for MRI Classification");

network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
network.fit(X_im_train,y_im_train,epochs=200, batch_size=75, callbacks=[tboard], validation_split=0.2)


# In[ ]:


#tensorboard --logdir=/Users/glynismattheisen/Desktop/Final/output


# # Visualize the Input

# In[408]:


def plot_test_images(subject,diagnosis, array):
    plt.figure(figsize=(20, 20))
    print("Test Images".center(os.get_terminal_size().columns))
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.title(str(subject[i]) +'  Classification: ' + str(diagnosis[i]), fontsize=24, pad = 20)
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(array[i], cmap='gray')
        plt.subplots_adjust(wspace=0.25)
    plt.show()  
    
plot_test_images(subjects['Subject'],subjects['Diagnosis'],brain_array)


# # T-SNE

# In[ ]:


x = x[:,:,:,0]
tsne = TSNE(n_components=2, init='pca', random_state=42)
assert x.shape == (666, 208, 176)
X_tsne = tsne.fit_transform(x.reshape((666,208*176)))


# In[ ]:


def run_tsne(X_tsne, y):
    X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_tsne,y, random_state = 42)
    he = keras.initializers.he_normal(seed=42)   
    model_t = Sequential([
    Dense(50, input_shape=(2,), kernel_initializer=he),
    Activation('sigmoid'),
    Dense(20),
    Activation('sigmoid'),
    Dense(10),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid'),
    ])
    
    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_t.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne

model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne= run_tsne(X_tsne,y)

# tboard = keras.callbacks.TensorBoard(log_dir='/Users/glynismattheisen/Desktop/Final',histogram_freq=5, write_graph=True, write_images=True)


# In[327]:


model_t.fit(X_t_train,y_t_train,epochs=100, batch_size=50, validation_split=0.2)


# In[328]:


layer_dict = dict([(layer.name, layer) for layer in model_t.layers]) # get the symbolic outputs of each "key" layer (we gave them unique names).


# In[329]:


tsne_accuracy, tsne_recall, tsne_precision = get_metrics(model_t,X_t_test, y_t_test)


# In[330]:


prediction = model_t.predict_classes(X_t_test)


# # Visualize the T-SNE

# In[391]:


X = x.reshape((666,208*176))
feat_cols = ['pixel'+str(i) for i in range(X.shape[1])] # name each column a pixel number
df = pd.DataFrame(X,columns=feat_cols) # convert X to dataframe
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i)) 
rndperm = np.random.permutation(df.shape[0])

plt.gray()
fig = plt.figure(figsize=(20,20))
for i in range(0,12):
    ax = fig.add_subplot(4,3,i+1, title='Classification: ' + str(df.loc[rndperm[i],'label']) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((208,176)).astype(float))
    plt.tight_layout()
plt.show()


# In[295]:


n_sne = 7000
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

ggplot.ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) + geom_point(size=70,alpha=0.2) + ggtitle("tSNE dimensions colored by digit")


# In[296]:


healthy = df_tsne.loc[df.label == "0"]
alz = df_tsne.loc[df.label == "1"]

g = sns.JointGrid(x="x-tsne", y="y-tsne", data=df_tsne, ratio = 2)
sns.kdeplot(healthy['x-tsne'], healthy['y-tsne'], cmap="Greens",
            shade=False, shade_lowest=False, ax=g.ax_joint)
sns.kdeplot(alz['x-tsne'], alz['y-tsne'], cmap="Reds",
            shade=False, shade_lowest=False, ax=g.ax_joint)
sns.kdeplot(healthy['x-tsne'], color="g", ax=g.ax_marg_x, shade = True, legend = False)
sns.kdeplot(alz['x-tsne'], color="r", ax=g.ax_marg_x, shade = True, legend = False)
sns.kdeplot(healthy['y-tsne'], color="g", ax=g.ax_marg_y, vertical=True, shade = True, legend = False)
sns.kdeplot(alz['y-tsne'], color="r", ax=g.ax_marg_y, vertical=True, shade = True, legend = False)
plt.show()


# In[28]:


# --logdir=/Users/glynisttheisen/Desktop/Final


# # PNCA

# In[332]:


X.shape


# In[333]:


m_PCA = PCA(n_components=10)
m_PCA.fit(X)
xt = m_PCA.transform(X)
xback = m_PCA.inverse_transform(xt)
assert X.shape == xback.shape
xback = xback.reshape((666,208,176))


# In[305]:


def draw(subject, input):
    plt.figure(figsize=(20,20))

    for i in range(6):
        plt.subplot(3, 3, i+1)
        plt.title(str(subject[i]) + ' Classification: ' + str(df.loc[rndperm[i],'label']), fontsize=24, pad = 20)
        plt.imshow(input[i], cmap=plt.cm.Greys)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25)
        plt.axis('off')
    plt.show()


# In[306]:


draw(subjects['Subject'], xback)


# In[334]:


X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(xt,y, random_state = 42)

he = keras.initializers.he_normal(seed=42)    

model_PCA = Sequential([
Dense(100, input_shape=(10,), kernel_initializer=he),
Activation('sigmoid'),
Dense(30),
Activation('sigmoid'),
Dense(15),
Activation('sigmoid'),
Dense(1),
Activation('sigmoid'),
])
optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_PCA.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[335]:


model_PCA.fit(X_PCA_train,y_PCA_train,epochs=100, batch_size=50, validation_split=0.2)


# In[336]:


PCA_accuracy, PCA_recall, PCA_precision = get_metrics(model_PCA,X_PCA_test, y_PCA_test)


# # Feeding in data

# In[36]:


sample = xt[1,].reshape(1,10) # take a sample from broad x set
sample.shape # shape of sample


# In[37]:


model_PCA.predict_classes(sample) # predict sample


# # correct and incorrect predictions

# In[337]:


wrong_predictions = []
correct_predictions = []
for i in range(167):
    if prediction[i][0] != y_t_test[i]:
        wrong_predictions.append(i)
    else:
        correct_predictions.append(i)


# In[338]:


plt.figure(figsize=(20,20))

for x,i in enumerate(wrong_predictions):
    plt.subplot(10, 5, x+1)
    plt.title('True Classification: ' + str(df.loc[rndperm[i],'label']))
    plt.imshow(xback[i], cmap=plt.cm.Greys)
    plt.axis('off')
    plt.tight_layout()
plt.show()


# In[339]:


len(correct_predictions)


# In[340]:


correct_predictions = correct_predictions[:20]


# In[341]:


plt.figure(figsize=(20,20))

for x,i in enumerate(correct_predictions):
    plt.subplot(10, 5, x+1)
    plt.title('True Classification: ' + str(df.loc[rndperm[i],'label']))
    plt.imshow(xback[i], cmap=plt.cm.Greys)
    plt.axis('off')
    plt.tight_layout()
plt.show()


# # Oasis 1 Textual Data

# In[342]:


df = pd.read_csv('oasis1_cross-sectional.csv')
Subj_ID = []

for i in df['ID']:
    Subj_ID.append(get_subject(i))

df['Subject'] = pd.DataFrame(Subj_ID)

del df['ID']
del df['Hand']
del df['Delay']


# In[343]:


df_demo = df[['Subject','M/F','Educ','SES','CDR','nWBV']].copy()


# In[344]:


# Identify percentage of each categories composed of missing values
total = df_demo.isnull().sum().sort_values(ascending=False)
percent = df_demo.isnull().sum()/df_demo.isnull().count()*100
pd.concat([total, percent], axis=1, keys=['Total', '%'],sort=True)


# In[345]:


df_demo['CDR'].fillna(0, inplace=True)
df_demo['CDR'].replace({0.5:1}, inplace=True) # replace 0.5 CDR values with 1 for Alz diagnosis
df_demo['CDR'].replace({2:1}, inplace=True) # replace 2 CDR values with 1 for Alz diagnosis


# In[346]:


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


# In[347]:


df_demo['Educ'] = replace_null(df_demo['Educ']) # fill missing Educ value with mean +/- SES
assert df_demo['Educ'].isnull().sum() == 0
df_demo['SES'] = replace_null(df_demo['SES']) # fill missing SES value with mean +/- SES
assert df_demo['SES'].isnull().sum() == 0

genders = {"M": 0, "F": 1}
df_demo['M/F'] = df_demo['M/F'].map(genders) # replace male = 0 and female = 1


# In[348]:


temp_df = pd.DataFrame() 
for i in alz_subjects['Subject']:
    New_df = df_demo[df_demo['Subject'] == i]
    temp_df = temp_df.append(New_df, ignore_index=True) #Moving the contents of newly created dataframe to the temporary dataframe
    
total_df = df_demo.append(temp_df)
total_df.corr(method= 'pearson')


# In[349]:


g = sns.PairGrid(temp_df)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);


# In[350]:


total_df = total_df.drop("M/F", axis=1)
total_df.set_index('Subject',inplace=True)

X_text = total_df.drop("CDR", axis=1)
y = total_df["CDR"]

assert X_text.shape == (666, 3)
assert y.shape == (666,)


# In[351]:


X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_text,y, random_state = 42)

model_txt = Sequential([
    Dense(150, input_shape=(3,)),
    Activation('relu'),
    Dropout(1.0),
    Dense(75),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])

optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_txt.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[352]:


model_txt.fit(X_text_train,y_text_train,epochs=100, batch_size=25)


# In[353]:


txt_accuracy, txt_recall, txt_precision = get_metrics(model_txt,X_text_test, y_text_test)


# # Random Forest

# In[354]:


RanFor = RandomForestClassifier(max_depth = 2, n_estimators = 15)
RanFor.fit(X_text_train,y_text_train)
print ("Training score: " + str(round(RanFor.score(X_text_train,y_text_train)*100,2)) + '%')
print ("Test score: " + str(round(RanFor.score(X_text_test,y_text_test)*100,2)) + '%')


# In[355]:


random_trees, random_depths =  random.sample(range(1,20),10), random.sample(range(1,50),25)


# In[356]:


grid = GridSearchCV(RanFor, 
        param_grid={'n_estimators': random_trees, 'max_depth': random_depths},
        scoring='accuracy', 
        n_jobs=1,
        cv=5
        )

grid.fit(X_text_train,y_text_train)
print("Best score: " + str(round(grid.best_score_*100,2)) + '%', str(grid.best_params_))


# In[357]:


final_model = grid.best_estimator_
final_model.fit(X_text_train,y_text_train)

print ("Training score: " + str(final_model.score(X_text_train,y_text_train)))
print ("Test score: " + str(final_model.score(X_text_test,y_text_test)))


# In[358]:


RF_txt_accuracy = round(final_model.score(X_text_test,y_text_test)*100,2)
txt_pred = final_model.predict(X_text_test)
RF_txt_recall = round((recall_score(y_text_test, txt_pred))*100, 2)
RF_txt_precision = round(precision_score(y_text_test, txt_pred)*100, 2)


# # Oasis 1 FS

# In[359]:


free = pd.read_csv('oasis1_FS.csv')
free.set_index('Subject', inplace=True)
subjects_x = subjects.set_index('Subject')

del free['FS_ASEGREGIONANALYSIS ID']
del free['MR ID']

subjects_x = subjects_x.head(423)
subjects_x = subjects_x.sort_index(ascending = True)

merge = free.merge(subjects_x, how='inner',left_index=True, right_index=True)


# In[360]:


norm_to_ICV = lambda x : x / merge['ICV']


# In[361]:


del merge['BrainMaskVol']
del merge['BrainSegNVox']
merge['BrainMaskNVox'] = norm_to_ICV(merge['BrainMaskNVox'])
merge['BrainSegVol'] = norm_to_ICV(merge['BrainSegVol'])


# In[362]:


merge.corr()


# In[363]:


temp_df = pd.DataFrame() 
for i in alz_subjects['Subject']:
    New_df = df_demo[df_demo['Subject'] == i]
    temp_df = temp_df.append(New_df, ignore_index=True) #Moving the contents of newly created dataframe to the temporary dataframe
    
total_df = df_demo.append(temp_df)

total_df.set_index('Subject',inplace=True)


# In[364]:


total_df = total_df.head(423)


# In[365]:


merge_2 = merge.merge(total_df, how='inner',left_index=True, right_index=True)


# In[366]:


merge_2.corr()


# In[367]:


import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[368]:


survived = 'healthy'
not_survived = 'dementia'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20, 6))
women = merge_2[merge_2['M/F']== 1]
men = merge_2[merge_2['M/F']== 0]
ax = sns.distplot(women[women['CDR']==1.0].SES, bins=5, label = survived, ax = axes[0], kde =True)
ax = sns.distplot(women[women['CDR']==0.0].SES, bins=5, label = not_survived, ax = axes[0], kde =True)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['CDR']==1].SES, bins=5, label = survived, ax = axes[1], kde = True)
ax = sns.distplot(men[men['CDR']==0].SES, bins=5, label = not_survived, ax = axes[1], kde = True)
ax.legend()
ax.set_title('Male')


# In[369]:


healthy = 'Cognitively Normal'
dementia = 'Alzheimer\'s Diagnosis'

fig = plt.figure(figsize=(30, 18))
grid = plt.GridSpec(2, 3, wspace=0.2, hspace=0.5)

plt.subplot(grid[0, 0])
ax = sns.kdeplot(merge_2[merge_2['Diagnosis']==1.0].SES, label = dementia, color='blue', shade=True)
ax = sns.kdeplot(merge_2[merge_2['Diagnosis']==0.0].SES, label = healthy, color ='green', shade=True)
ax.legend(fontsize=15)
plt.xlabel('Socioeconomic Status', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax.set_title('Diagnosis and SES', fontsize=24, pad = 20)

plt.subplot(grid[0, 1])
ax1 = sns.kdeplot(merge_2[merge_2['Diagnosis']==1.0]['M/F'], label = dementia, color='blue', shade=True)
ax1 = sns.kdeplot(merge_2[merge_2['Diagnosis']==0.0]['M/F'], label = healthy, color='green', shade=True)
ax1.legend(fontsize=15)
plt.xlabel('M: 0                            F:1', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.set_title('Diagnosis and Gender', fontsize=24, pad = 20)

plt.subplot(grid[0, 2])
ax2 = sns.kdeplot(merge_2[merge_2['Diagnosis']==1.0]['Educ'], label = dementia, color='blue', shade=True)
ax2 = sns.kdeplot(merge_2[merge_2['Diagnosis']==0.0]['Educ'], label = healthy, color='green', shade=True)
ax2.legend(fontsize=15)
plt.xlabel('Years of Education', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax2.set_title('Diagnosis and Education', fontsize=24, pad = 20)


plt.subplot(grid[1, 0])
ax = sns.kdeplot(merge_2[merge_2['Diagnosis']==1.0].BrainMaskNVox, label = dementia, shade=True, color='blue')
ax = sns.kdeplot(merge_2[merge_2['Diagnosis']==0.0].BrainMaskNVox, label = healthy, shade=True, color ='green')
plt.xlabel('BrainMaskNVox / ICV', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax.legend(fontsize=15)
ax.set_title('Diagnosis and BrainMaskNVox', fontsize=24, pad = 20)

plt.subplot(grid[1, 1])
ax1 = sns.kdeplot(merge_2[merge_2['Diagnosis']==1.0].BrainSegVol, label = dementia, shade=True,color='blue')
ax1 = sns.kdeplot(merge_2[merge_2['Diagnosis']==0.0].BrainSegVol, label = healthy, shade=True,color='green')
plt.xlabel('BrainSegVol / ICV', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title('Diagnosis and BrainSegVol', fontsize=24, pad = 20)

plt.subplot(grid[1, 2])
ax1 = sns.kdeplot(merge_2[merge_2['Diagnosis']==1.0].nWBV, label = dementia, shade=True,color='blue')
ax1 = sns.kdeplot(merge_2[merge_2['Diagnosis']==0.0].nWBV, label = healthy, shade=True,color='green')
plt.xlabel('nWBV / ICV', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title('Diagnosis and nWBV', fontsize=24, pad = 20)


# In[370]:


merge_2.head(4)


# In[371]:


del merge_2['CDR']
del merge_2['ICV']


# In[372]:


corr = merge_2.corr()


# In[373]:


corr


# In[374]:


healthy = merge_2[merge_2['Diagnosis'] ==0]
alz = merge_2[merge_2['Diagnosis'] ==1]
healthy.describe()


# In[375]:


alz.describe()


# In[376]:


columns = corr.columns.values


# In[377]:


z = [-0.447697,-0.581303,1, -0.032847,0.079117,0.340303,-0.649548]

list_n = [item*150 for item in z] 

abs_list = lambda x :abs(x) # normalize x values

for x, i in enumerate(list_n):
    list_n[x] = abs(i)


# In[378]:


import plotly.plotly as py
import plotly.graph_objs as go

z = [-0.447697,-0.581303,1, -0.032847,0.079117,0.340303,-0.649548]

trace0 = go.Scatter(
    x=[-3.5,-2.5, 0, -4.5, 2.25, 1.25, -1.5],
    y=[-0.447697,-0.581303,1, -0.032847,0.079117,0.340303,-0.649548],
    text=columns,
    hoverinfo = 'y',
    textposition=['bottom center','top center','middle center','bottom center','bottom center','middle center','middle center'],
    mode='markers + text',
    marker=dict(
        color =[300,325,375,275,325,350,350],
        opacity=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
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


# In[379]:


merge_2.head()


# In[380]:


X_text = merge_2.drop("Diagnosis", axis=1)
y = merge_2["Diagnosis"]
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_text,y, random_state = 42)


# In[381]:


RanFor = RandomForestClassifier(max_depth = 2, n_estimators = 15)
RanFor.fit(X_text_train,y_text_train)
print ("Training score: " + str(round(RanFor.score(X_text_train,y_text_train)*100,2)) + '%')
print ("Test score: " + str(round(RanFor.score(X_text_test,y_text_test)*100,2)) + '%')


# In[382]:


random_trees, random_depths =  random.sample(range(1,30),10), random.sample(range(1,50),25)


# In[383]:


random_depth =  random.sample(range(1,30),10)


# In[384]:


grid = GridSearchCV(RanFor, 
        param_grid={'max_depth': random_trees, 'n_estimators': random_depths},
        scoring='accuracy', 
        n_jobs=1,
        cv=5
        )

grid.fit(X_text_train,y_text_train)
print("Best score: " + str(round(grid.best_score_*100,2)) + '%', str(grid.best_params_))


# In[385]:


X_text_train.head(2)


# In[386]:


final_model = grid.best_estimator_
final_model.fit(X_text_train,y_text_train)

print ("Training score: " + str(final_model.score(X_text_train,y_text_train)))
print ("Test score: " + str(final_model.score(X_text_test,y_text_test)))


# In[387]:


RF_txt_vol_accuracy = round(final_model.score(X_text_test,y_text_test)*100,2)
txt_vol_pred = final_model.predict(X_text_test)
RF_txt_vol_recall = round((recall_score(y_text_test, txt_vol_pred))*100, 2)
RF_txt_vol_precision = round(precision_score(y_text_test, txt_vol_pred)*100, 2)


# In[388]:


estimator = final_model.estimators_[5]


# In[311]:


tree = export_graphviz(estimator, out_file=None, 
                class_names=["Healthy", "Alzheimer's"], 
                feature_names=['BrainMaskNVox','BrainSegVol','M/F','Educ','SES','nWBV'],
                impurity=False,
                filled=True)

graph = graphviz.Source(tree)
graph.render('DecTree')  # creates PDF
graph  # in Jupyter


# ## Results

# In[389]:


result_df = pd.DataFrame({
    'Model': ['NN','T-SNE + NN','PCA + NN','Demographic Data + RF'],
    'Accuracy': [model_accuracy,tsne_accuracy,PCA_accuracy, RF_txt_vol_accuracy],
    'Precision': [model_precision,tsne_precision,PCA_precision, RF_txt_vol_precision],
    'Recall': [model_recall,tsne_recall,PCA_recall, RF_txt_vol_recall]
 })
result_df = results.sort_values(by='Accuracy', ascending=False)
result_df = result_df.set_index('Model')
result_df


# In[390]:


result_df.style.background_gradient(cmap='GnBu', low=1, high=0.75, axis=0).set_properties(**{'font-size':'15pt'})

