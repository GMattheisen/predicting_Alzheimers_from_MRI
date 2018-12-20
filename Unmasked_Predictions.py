
# coding: utf-8

# In[19]:


import random
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import matplotlib
import numpy as np
import os
import imageio
from random import seed
seed(42)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import recall_score, precision_score
from sklearn.decomposition import PCA
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
from ggplot import aes
from ggplot import geom_point
from ggplot import ggtitle


# In[20]:


def get_subject(file_name): # parse out subjectids from file names
    return file_name[0:9]


# In[21]:


def get_slice(orientation):
    
    path = '/Users/glynismattheisen/Desktop/Final/Discs'
    disc_list = os.listdir(path) # get list of discs from directory
    MacOS_file = '.DS_Store'
    if MacOS_file in disc_list:
        disc_list.remove(MacOS_file) # remove annoying MacOS file

    brain_list, total_subjects = [], [] # generate list of brain images for input to NN, all subjects used in study
    
    for disc in disc_list:
        path_ind_disc = f'{path}/{disc}'
        subject_list_p_disc = os.listdir(path_ind_disc) # generate list of subjects in each disc
        if MacOS_file in subject_list_p_disc:
            subject_list_p_disc.remove(MacOS_file) # remove annoying MacOS file
            
        for subj_id in subject_list_p_disc:
            total_subjects.append(subj_id) # maintain a list of all subjects included in study for diagnosis labeling later

            path_n4 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n4_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'
            path_n3 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n3_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'
            path_n6 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n6_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'
            path_n5 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n5_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'

            
            if os.path.exists(path_n4) == True:
                brain_list.append(plt.imread(path_n4)) # append if file format exists
            elif os.path.exists(path_n3) == True:
                brain_list.append(plt.imread(path_n3)) # append if file format exists
            elif os.path.exists(path_n6) == True:
                brain_list.append(plt.imread(path_n6)) # append if file format exists
            elif os.path.exists(path_n5) == True:
                brain_list.append(plt.imread(path_n5)) # append if file format exists
    
    return disc_list, brain_list, total_subjects


# In[22]:


def get_diagnosis(total_subjects):
    oasis1 = pd.read_csv('/Users/glynismattheisen/Desktop/Final2/oasis_cross-sectional.csv') # read in summary file
    
    oasis1['CDR'].fillna(0, inplace=True) # null values are helathy diagnoses
    
    diagnosis={0.:'normal', 0.5:'alz', 1.:'alz', 2.:'alz' } # convert to labels
    oasis1 = oasis1.replace({"CDR": diagnosis})
    diagnosis={'normal':0,'alz':1} # convert back to numbers
    oasis1 = oasis1.replace({"CDR": diagnosis})
    
    subjects = []    # get subject names for each MRI from file names
    for subj_id in total_subjects:
        subjects.append(get_subject(subj_id))
        
    subjects = pd.DataFrame(subjects, columns = ['Subject'])
    
    oasis1['Subject'] =oasis1['ID'].apply(get_subject)     # extract subject ID from MR ID
    
    diagnosis_dict= {}     # create a dictionary with subject and diagnosis
    for num in range(0,436):
        diagnosis_dict[oasis1['Subject'][num]] = oasis1['CDR'][num]
        
    diag = [] # list of diagnoses
    for x in subjects['Subject']:
        diag.append(diagnosis_dict[x])

    subjects['Diagnosis'] = pd.DataFrame(diag) # append diagnoses to subjects
    
    return subjects # dataframe of subjects with their diagnoses


# In[23]:


def balance_set(subjects, total_subjects, disc_list, orientation):
    alz_subjects = subjects.sort_values(by='Diagnosis', ascending = False).head(97) # all subjects diagnosed as alz
    alz_subjects = alz_subjects.append(alz_subjects)
    head = alz_subjects.head(42)
    alz_subjects = alz_subjects.append(head) # extend alzheimer's subjects to make eqv to healthy subjects
    
    for i in alz_subjects['Subject']:
        total_subjects.append(str(i) + '_MR1') # maintain a list of all subjects included in study for diagnosis labeling later
    for disc in disc_list:
            path = f'/Users/glynismattheisen/Desktop/Final/Discs/{disc}'

            for i in alz_subjects['Subject']:

                path_n4 = path + f'/{i}_MR1/PROCESSED/MPRAGE/T88_111/{i}_MR1_mpr_n4_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'
                path_n3 = path + f'/{i}_MR1/PROCESSED/MPRAGE/T88_111/{i}_MR1_mpr_n3_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'
                path_n6 = path + f'/{i}_MR1/PROCESSED/MPRAGE/T88_111/{i}_MR1_mpr_n6_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'
                path_n5 = path + f'/{i}_MR1/PROCESSED/MPRAGE/T88_111/{i}_MR1_mpr_n5_anon_111_t88_gfc_{orientation[0]}_{orientation[1]}.gif'

            
                if os.path.exists(path_n4) == True:
                    brain_list.append(plt.imread(path_n4)) # append if file format exists
                elif os.path.exists(path_n3) == True:
                    brain_list.append(plt.imread(path_n3)) # append if file format exists
                elif os.path.exists(path_n6) == True:
                    brain_list.append(plt.imread(path_n6)) # append if file format exists
                elif os.path.exists(path_n5) == True:
                    brain_list.append(plt.imread(path_n5)) # append if file format exists

    brain_array = np.asarray(brain_list)
    return brain_array, alz_subjects


# In[24]:


disc_list, brain_list, total_subjects = get_slice(orientation = ['sag','95'])


# In[25]:


subjects = get_diagnosis(total_subjects)


# In[26]:


brain_array, alz_subjects = balance_set(subjects, total_subjects, disc_list, orientation = ['sag','95'])


# In[27]:


subjects = get_diagnosis(total_subjects)


# In[28]:


shape = brain_array.shape
input_shape = shape[1]*shape[2]
shape


# In[29]:


brain_array.shape


# In[30]:


x = brain_array
y = np.asarray(subjects['Diagnosis'])


# In[31]:


def norm(x):
    x = x.reshape(-1, 176,208,1)
    m = np.max(x)
    mi = np.min(x)
    x_norm = (x - mi) / (m - mi)
    assert (np.min(x_norm), np.max(x_norm)) == (0.0, 1.0)
    return x_norm


# In[32]:


x_norm = norm(x)


# In[33]:


X_im_train, X_im_test, y_im_train, y_im_test = train_test_split(x_norm,y, random_state = 42)


# In[34]:


model = Sequential([
    Conv2D(filters=100,kernel_size=(3,3),strides=(10,10), input_shape=(176,208,1),padding ='same'),
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


# In[35]:


"""
from ann_visualizer.visualize import ann_viz;

ann_viz(model, title="Neural Network for MRI Classification", view=);"""


# In[36]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[37]:


model.fit(X_im_train,y_im_train,epochs=100, batch_size=75)


# In[38]:


model.evaluate(X_im_test,y_im_test)


# In[39]:


def get_metrics(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    accuracy = round(results[1]*100,2)
    
    Y_pred = model.predict_classes(X_test)
    
    recall = round((recall_score(y_test, Y_pred))*100, 2)
    precision = round(precision_score(y_test, Y_pred)*100, 2)
    
    return accuracy, recall, precision


# In[40]:


sag_accuracy, sag_recall, sag_precision = get_metrics(model,X_im_test, y_im_test)


# In[41]:


sag_accuracy


# # Visualize the Input

# In[220]:


def visualize_input(input_array):
    plt.figure(figsize=(20, 20))
    print("Test Images")
    for i in range(0,12):
        plt.subplot(4, 3, i+1)
        plt.title(str(subjects['Subject'][i]) +'  Classification: ' + str(subjects['Diagnosis'][i]), fontsize=24, pad = 20)
        plt.tight_layout()
        plt.imshow(input_array[i], cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.25)
    plt.show()  
    
visualize_input(brain_array)


# # T-SNE

# In[193]:


he = keras.initializers.he_normal(seed=42)


# In[194]:


x.shape


# In[195]:


tsne = TSNE(n_components=2, init='pca', random_state=42)
assert x.shape == (666, 176, 208)
X = x.reshape((666,208*176))
X_tsne = tsne.fit_transform(X)


# In[196]:


def run_tsne(x, y):
    X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_tsne,y, random_state = 42)
    
    model_t = Sequential([
    Dense(60, input_shape=(2,), kernel_initializer=he),
    Activation('sigmoid'),
    Dense(30),
    Activation('sigmoid'),
    Dense(15),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid'),
    ])
    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_t.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return X, model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne


# In[197]:


X, model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne= run_tsne(x,y)


# In[198]:


# tboard = keras.callbacks.TensorBoard(log_dir='/Users/glynismattheisen/Desktop/Final',histogram_freq=5, write_graph=True, write_images=True)


# In[199]:


model_t.fit(X_t_train,y_t_train,epochs=100, batch_size=50, validation_split=0.2)


# In[200]:


model_t.evaluate(X_t_test,y_t_test)


# In[201]:


prediction = model_t.predict_classes(X_t_test)


# In[202]:


sag_t_accuracy, sag_t_recall, sag_t_precision = get_metrics(model_t,X_t_test, y_t_test)


# # Visualize the T-SNE

# In[203]:


feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

# x, y = None, None

print ('Size of the dataframe: {}'.format(df.shape))


# In[204]:


rndperm = np.random.permutation(df.shape[0])


# In[205]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the graph
plt.gray()
fig = plt.figure( figsize=(20,20) )
for i in range(0,12):
    ax = fig.add_subplot(4,3,i+1, title='Classification: ' + str(df.loc[rndperm[i],'label']) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((176,208)).astype(float))
    plt.tight_layout()
plt.show()


# In[206]:


n_sne = 7000
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


# In[207]:


df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

view = ggplot.ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) + geom_point(size=70,alpha=0.2) + ggtitle("tSNE dimensions colored by digit")


# In[208]:


view


# In[209]:


# --logdir=/Users/glynisttheisen/Desktop/Final


# # PCA

# In[210]:


m_PCA = PCA(n_components=10)


# In[211]:


X.shape


# In[212]:


m_PCA.fit(X)
xt = m_PCA.transform(X)
xback = m_PCA.inverse_transform(xt)


# In[213]:


xback.shape


# In[214]:


176*208


# In[215]:


xback = xback.reshape((666,176,208))


# In[216]:


xback.shape


# In[221]:


visualize_input(xback)


# # correct and incorrect predictions

# In[ ]:


wrong_predictions = []
correct_predictions = []
for i in range(167):
    if prediction[i][0] != y_t_test[i]:
        wrong_predictions.append(i)
    else:
        correct_predictions.append(i)


# In[ ]:


len(wrong_predictions)


# In[ ]:


len(correct_predictions)


# In[ ]:


plt.figure(figsize=(20,20))

for x,i in enumerate(wrong_predictions):
    plt.subplot(10, 5, x+1)
    plt.title('True Classification: ' + str(df.loc[rndperm[i],'label']))
    plt.imshow(xback[i], cmap=plt.cm.Greys)
    plt.axis('off')
    plt.tight_layout()
plt.show()


# In[ ]:


len(correct_predictions)


# In[ ]:


correct_predictions = correct_predictions[:20]


# In[ ]:


plt.figure(figsize=(20,20))

for x,i in enumerate(correct_predictions):
    plt.subplot(10, 5, x+1)
    plt.title('True Classification: ' + str(df.loc[rndperm[i],'label']))
    plt.imshow(xback[i], cmap=plt.cm.Greys)
    plt.axis('off')
    plt.tight_layout()
plt.show()


# # Cor

# In[222]:


disc_list, brain_list, total_subjects = get_slice(orientation = ['cor','110'])

subjects = get_diagnosis(total_subjects)

brain_array, alz_subjects = balance_set(subjects, total_subjects, disc_list, orientation = ['cor','110'])

subjects = get_diagnosis(total_subjects)

shape = brain_array.shape
input_shape = shape[1]*shape[2]
shape

x = brain_array
y = np.asarray(subjects['Diagnosis'])


# In[225]:


visualize_input(brain_array)


# In[226]:


x.shape


# In[227]:


def norm(x):
    x = x.reshape(-1, 176,176,1)
    m = np.max(x)
    mi = np.min(x)
    x_norm = (x - mi) / (m - mi)
    assert (np.min(x_norm), np.max(x_norm)) == (0.0, 1.0)
    return x_norm

x_norm = norm(x)


# In[228]:


X_im_train, X_im_test, y_im_train, y_im_test = train_test_split(x_norm,y, random_state = 42)


# In[229]:


model = Sequential([
    Conv2D(filters=100,kernel_size=(3,3),strides=(10,10), input_shape=(176,176,1),padding ='same'),
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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_im_train,y_im_train,epochs=150, batch_size=75)


# In[230]:


model.evaluate(X_im_test, y_im_test)


# In[231]:


corr_accuracy, corr_recall, corr_precision = get_metrics(model,X_im_test, y_im_test)


# # T-SNE

# In[232]:


he = keras.initializers.he_normal(seed=42)


# In[233]:


x.shape


# In[234]:


tsne = TSNE(n_components=2, init='pca', random_state=42)
assert x.shape == (666, 176, 176)
X = x.reshape((666,176*176))
X_tsne = tsne.fit_transform(X)


# In[235]:


def run_tsne(x, y):
    X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_tsne,y, random_state = 42)
    
    model_t = Sequential([
    Dense(60, input_shape=(2,), kernel_initializer=he),
    Activation('sigmoid'),
    Dense(30),
    Activation('sigmoid'),
    Dense(15),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid'),
    ])
    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_t.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return X, model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne


# In[236]:


X, model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne= run_tsne(x,y)


# In[237]:


# tboard = keras.callbacks.TensorBoard(log_dir='/Users/glynismattheisen/Desktop/Final',histogram_freq=5, write_graph=True, write_images=True)


# In[238]:


model_t.fit(X_t_train,y_t_train,epochs=100, batch_size=50, validation_split=0.2)


# In[239]:


model_t.evaluate(X_t_test,y_t_test)


# In[240]:


prediction = model_t.predict_classes(X_t_test)


# In[241]:


corr_t_accuracy, corr_t_recall, corr_t_precision = get_metrics(model_t,X_t_test, y_t_test)


# # Visualize the T-SNE

# In[242]:


feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

# x, y = None, None

print ('Size of the dataframe: {}'.format(df.shape))


# In[243]:


rndperm = np.random.permutation(df.shape[0])


# In[244]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the graph
plt.gray()
fig = plt.figure( figsize=(20,20) )
for i in range(0,10):
    ax = fig.add_subplot(3,5,i+1, title='Classification: ' + str(df.loc[rndperm[i],'label']) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((176,176)).astype(float))
    plt.tight_layout()
plt.show()


# In[245]:


n_sne = 7000
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


# In[246]:


df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

ggplot.ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) + geom_point(size=70,alpha=0.2) + ggtitle("tSNE dimensions colored by digit")


# # Tra

# In[247]:


disc_list, brain_list, total_subjects = get_slice(orientation = ['tra','90'])

subjects = get_diagnosis(total_subjects)

brain_array, alz_subjects = balance_set(subjects, total_subjects, disc_list, orientation = ['tra','90'])

subjects = get_diagnosis(total_subjects)

shape = brain_array.shape
input_shape = shape[1]*shape[2]
shape

x = brain_array
y = np.asarray(subjects['Diagnosis'])


# In[248]:


visualize_input(brain_array)


# In[99]:


x.shape


# In[100]:




def norm(x):
    x = x.reshape(-1, 208,176,1)
    m = np.max(x)
    mi = np.min(x)
    x_norm = (x - mi) / (m - mi)
    assert (np.min(x_norm), np.max(x_norm)) == (0.0, 1.0)
    return x_norm

x_norm = norm(x)


X_im_train, X_im_test, y_im_train, y_im_test = train_test_split(x_norm,y, random_state = 42)

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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[101]:


model.fit(X_im_train,y_im_train,epochs=150, batch_size=75)


# In[102]:


model.evaluate(X_im_test, y_im_test)


# In[103]:


tra_accuracy, tra_recall, tra_precision = get_metrics(model,X_im_test, y_im_test)


# # T-SNE
# 

# In[104]:


he = keras.initializers.he_normal(seed=42)


tsne = TSNE(n_components=2, init='pca', random_state=42)
assert x.shape == (666, 208, 176)
X = x.reshape((666,208*176))
X_tsne = tsne.fit_transform(X)


# In[105]:


def run_tsne(x, y):
    X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_tsne,y, random_state = 42)
    
    model_t = Sequential([
    Dense(60, input_shape=(2,), kernel_initializer=he),
    Activation('sigmoid'),
    Dense(30),
    Activation('sigmoid'),
    Dense(15),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid'),
    ])
    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_t.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return X, model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne

X, model_t, X_t_train, X_t_test, y_t_train, y_t_test, tsne= run_tsne(x,y)

# tboard = keras.callbacks.TensorBoard(log_dir='/Users/glynismattheisen/Desktop/Final',histogram_freq=5, write_graph=True, write_images=True)

model_t.fit(X_t_train,y_t_train,epochs=100, batch_size=50, validation_split=0.2)


# In[106]:


model_t.evaluate(X_t_test,y_t_test)


# In[107]:


prediction = model_t.predict_classes(X_t_test)


# In[108]:


tra_t_accuracy, tra_t_recall, tra_t_precision = get_metrics(model_t,X_t_test, y_t_test)


# # Visualize the T-SNE
# 

# In[109]:


feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

# x, y = None, None

print ('Size of the dataframe: {}'.format(df.shape))

rndperm = np.random.permutation(df.shape[0])

get_ipython().run_line_magic('matplotlib', 'inline')

# Plot the graph
plt.gray()
fig = plt.figure( figsize=(20,20) )
for i in range(0,10):
    ax = fig.add_subplot(3,5,i+1, title='Classification: ' + str(df.loc[rndperm[i],'label']) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((208,176)).astype(float))
    plt.tight_layout()
plt.show()


# In[110]:


n_sne = 7000
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

ggplot.ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) + geom_point(size=70,alpha=0.2) + ggtitle("tSNE dimensions colored by digit")


# In[149]:


results = pd.DataFrame({
   'Data Processing': ['SAG_Model','SAG_T','CORR_Model','CORR_t','TRA_Model','TRA_t'],
   'Accuracy': [sag_accuracy, sag_t_accuracy, corr_accuracy, corr_t_accuracy, tra_accuracy, tra_t_accuracy],
   'Precision': [sag_precision, sag_t_precision, corr_precision, corr_t_precision, tra_precision, tra_t_precision],
   'Recall': [sag_recall, sag_t_recall, corr_recall, corr_t_recall, tra_recall, tra_t_recall]
})
result_df = results.sort_values(by='Accuracy', ascending=False)
result_df = result_df.set_index('Data Processing')
result_df

