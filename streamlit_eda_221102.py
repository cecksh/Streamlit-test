import streamlit as st
import streamlit.components.v1 as stc
from datetime import datetime
import os
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import joblib
from sklearn.svm import OneClassSVM

matplotlib.use('Agg')


@st.cache
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def get_readable_time(mytime):
    return datetime.fromtimestamp(mytime).strftime('%Y%m%d-%H%M%S')

def get_file(file_name):
    file_name.drop(['Hopper_Temperature', 'Mold_Temperature_1', 'Mold_Temperature_2', 'Mold_Temperature_3',
            'Mold_Temperature_4', 'Mold_Temperature_5', 'Mold_Temperature_6', 'Mold_Temperature_7', 'Mold_Temperature_8',
            'Mold_Temperature_9', 'Mold_Temperature_10', 'Mold_Temperature_11', 'Mold_Temperature_12', 'Clamp_open_time'],
            axis = 1, inplace = True)
    moldset = file_name.iloc[:,2:30]

    moldset_9000R=moldset[moldset.Additional_Info_1.str.contains('09520 9000R')]
    moldset_PACKING =moldset[moldset.Additional_Info_1.str.contains('GURAD')]
    moldset_ROSSO =moldset[moldset.Additional_Info_1.str.contains('ROSSO')]

    labled = pd.DataFrame(moldset_9000R, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])
    labled_PACKING = pd.DataFrame(moldset_PACKING, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])
    labled_ROSSO = pd.DataFrame(moldset_ROSSO, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])

    Time_labled = pd.DataFrame(moldset_9000R, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure','Average_Back_Pressure', 'Filling_Time', 'TimeStamp'])

    model=IsolationForest(n_estimators=100, max_samples='auto', n_jobs=-1,
                        max_features=4, contamination=0.01)
    model.fit(labled.to_numpy())

    score = model.decision_function(labled.to_numpy())
    anomaly = model.predict(labled.to_numpy())
    Time_labled['scores']= score
    Time_labled['anomaly']= anomaly
    anomaly_data = labled.loc[Time_labled['anomaly']==-1]
    print(Time_labled['anomaly'].value_counts())

    return Time_labled

def get_file_OC(file_name):
    file_name.drop(['Hopper_Temperature', 'Mold_Temperature_1', 'Mold_Temperature_2', 'Mold_Temperature_3',
            'Mold_Temperature_4', 'Mold_Temperature_5', 'Mold_Temperature_6', 'Mold_Temperature_7', 'Mold_Temperature_8',
            'Mold_Temperature_9', 'Mold_Temperature_10', 'Mold_Temperature_11', 'Mold_Temperature_12', 'Clamp_open_time'],
            axis = 1, inplace = True)
    moldset = file_name.iloc[:,2:30]

    moldset_9000R=moldset[moldset.Additional_Info_1.str.contains('09520 9000R')]
    moldset_PACKING =moldset[moldset.Additional_Info_1.str.contains('GURAD')]
    moldset_ROSSO =moldset[moldset.Additional_Info_1.str.contains('ROSSO')]

    labled = pd.DataFrame(moldset_9000R, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])
    labled_PACKING = pd.DataFrame(moldset_PACKING, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])
    labled_ROSSO = pd.DataFrame(moldset_ROSSO, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])

    Time_labled = pd.DataFrame(moldset_9000R, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure','Average_Back_Pressure', 'Filling_Time', 'TimeStamp'])

    model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.003)
    model.fit(labled.to_numpy())

    score = model.decision_function(labled.to_numpy())
    anomaly = model.predict(labled.to_numpy())
    Time_labled['scores']= score
    Time_labled['anomaly']= anomaly
    anomaly_data = labled.loc[Time_labled['anomaly']==-1]
    print(Time_labled['anomaly'].value_counts())

    return Time_labled

def get_file_RF(file_name):
    file_name.drop(['Hopper_Temperature', 'Mold_Temperature_1', 'Mold_Temperature_2', 'Mold_Temperature_3',
            'Mold_Temperature_4', 'Mold_Temperature_5', 'Mold_Temperature_6', 'Mold_Temperature_7', 'Mold_Temperature_8',
            'Mold_Temperature_9', 'Mold_Temperature_10', 'Mold_Temperature_11', 'Mold_Temperature_12', 'Clamp_open_time'],
            axis = 1, inplace = True)
    moldset = file_name.iloc[:,2:30]

    moldset_9000R=moldset[moldset.Additional_Info_1.str.contains('09520 9000R')]
    moldset_PACKING =moldset[moldset.Additional_Info_1.str.contains('GURAD')]
    moldset_ROSSO =moldset[moldset.Additional_Info_1.str.contains('ROSSO')]

    labled = pd.DataFrame(moldset_9000R, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])
    labled_PACKING = pd.DataFrame(moldset_PACKING, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])
    labled_ROSSO = pd.DataFrame(moldset_ROSSO, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure', 'Average_Back_Pressure','Filling_Time'])

    Time_labled = pd.DataFrame(moldset_9000R, columns = ['Max_Injection_Pressure','Max_Switch_Over_Pressure','Max_Back_Pressure','Average_Back_Pressure', 'Filling_Time', 'TimeStamp'])

    return Time_labled



def main():
    menu = ['EDA', 'Autoencoder', 'OC-SVM', 'RandomForest']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'EDA':
        st.subheader('CSV files MetaData Extraction')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            moldset = df.iloc[:, 2:30]
            st.write(df)

            with st.expander('File Stats'):
                file_details = {'FileName':uploaded_file.name, 'FileSize':uploaded_file.size,
                                'FileType':uploaded_file.type, 'FileLength':len(df)}
                st.write(file_details)
                statinfo = os.stat(uploaded_file.readable())
                st.write(statinfo)
                stats_details = {
                    "Accessed_Time": get_readable_time(statinfo.st_atime),
                    "Creation_Time": get_readable_time(statinfo.st_ctime),
                    "Modified_Time": get_readable_time(statinfo.st_mtime),
                }
                st.write(stats_details)
            with st.expander('EDA'):
                product = list(moldset['Additional_Info_1'].unique())
                product_num = list()
                for i in product:
                    product_num.append(len(moldset.loc[moldset['Additional_Info_1'] == i]))
                product_df = pd.DataFrame({'product': product, 'number': product_num})
                product_df
                target = product[product_num.index(max(product_num))]
                st.write('Our target is : ', target)

                moldset_target = moldset[moldset.Additional_Info_1.str.contains(target)]

                moldset_target['TimeStamp'] = pd.to_datetime(moldset_target['TimeStamp'], format='%Y-%m-%d')
                moldset_target['y_m_d'] = moldset_target['TimeStamp'].dt.strftime("%Y-%m-%d")

                columns_name = moldset_target.columns.tolist()
                del columns_name[0:5]

                moldset_target = moldset_target[columns_name]
                st.write(moldset_target.describe())
                labeled_target_corr = moldset_target.corr(method='pearson')

                fig, ax = plt.subplots()
                sns.heatmap(labeled_target_corr, cmap='coolwarm_r', ax=ax)
                st.write(fig)

                fig, ax = plt.subplots(1, 5, figsize=[20, 4])
                sns.distplot(moldset_target['Filling_Time'], ax=ax[1], color='r')
                sns.distplot(moldset_target['Plasticizing_Time'], ax=ax[2], color='r')
                sns.distplot(moldset_target['Cycle_Time'], ax=ax[3], color='r')
                sns.distplot(moldset_target['Clamp_Close_Time'], ax=ax[4], color='r')
                st.write(fig)

                fig, ax = plt.subplots(1, 4, figsize=[20, 4])
                sns.distplot(moldset_target['Cushion_Position'], ax=ax[0], color='r')
                sns.distplot(moldset_target['Switch_Over_Position'], ax=ax[1], color='r')
                sns.distplot(moldset_target['Plasticizing_Position'], ax=ax[2], color='r')
                sns.distplot(moldset_target['Clamp_Open_Position'], ax=ax[3], color='r')
                st.write(fig)

                fig, ax = plt.subplots(1, 7, figsize=[20, 4])
                sns.distplot(moldset_target['Max_Injection_Speed'], ax=ax[0], color='r')
                sns.distplot(moldset_target['Max_Screw_RPM'], ax=ax[1], color='r')
                sns.distplot(moldset_target['Average_Screw_RPM'], ax=ax[2], color='r')
                sns.distplot(moldset_target['Max_Injection_Pressure'], ax=ax[3], color='r')
                sns.distplot(moldset_target['Max_Switch_Over_Pressure'], ax=ax[4], color='r')
                sns.distplot(moldset_target['Max_Back_Pressure'], ax=ax[5], color='r')
                sns.distplot(moldset_target['Average_Back_Pressure'], ax=ax[6], color='r')
                st.write(fig)

            with st.expander('RandomForest'):
                #데이터 프레임 moldset_target을 대상으로 라벨링 진행
                #isolation forest

                from sklearn.ensemble import IsolationForest
                labled = pd.DataFrame(moldset_target, columns=['Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
                                                              'Max_Back_Pressure', 'Average_Back_Pressure',
                                                              'Filling_Time'])
                Time_labled = pd.DataFrame(moldset_target, columns=['Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
                                                                   'Max_Back_Pressure', 'Average_Back_Pressure',
                                                                   'Filling_Time', 'y_m_d'])
                model = IsolationForest(n_estimators=100, max_samples='auto', n_jobs=-1,
                                        max_features=4, contamination=0.01)
                model.fit(labled.to_numpy())
                score = model.decision_function(labled.to_numpy())
                anomaly = model.predict(labled.to_numpy())
                labled['scores'] = score
                labled['anomaly'] = anomaly
                anomaly_data = labled.loc[labled['anomaly'] == -1]
                final_INNER_JOIN = pd.merge(Time_labled, anomaly_data, left_index=True, right_index=True, how='inner')
                final = final_INNER_JOIN.groupby('y_m_d').sum()
                labled.reset_index(drop=True)
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(n_estimators=50, max_samples=120, contamination=float(0.004),
                                      max_features=1.0, bootstrap=False, n_jobs=-1, random_state=None, verbose=0)
                #50개의 노드 수, 최대 50개의 샘플
                #0.4%의 outlier 색출.
                clf.fit(labled)
                pred = clf.predict(labled)
                labled['anomaly'] = pred
                outliers = labled.loc[labled['anomaly'] == -1]
                outlier_index = list(outliers.index)
                moldset_target.iloc[outlier_index, :].index
                isol_list = list(moldset_target.iloc[outlier_index, :].index)

                #LOF

                LOF_target = moldset_target[['Cushion_Position', 'Plasticizing_Position']]
                from sklearn.neighbors import LocalOutlierFactor
                # model specification
                model1 = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
                # model fitting
                y_pred = model1.fit_predict(LOF_target)
                # filter outlier index
                outlier_index = np.where(y_pred == -1)   #negative values are outliers and positives inliers
                # filter outlier values
                outlier_values = LOF_target.iloc[outlier_index]
                LOF_target.iloc[list(outlier_index[0]), :].index
                # LOF를 이용해 구한 9000R의 outlier index
                LOF_list = list(LOF_target.iloc[list(outlier_index[0]), :].index)

                #3시그마

                Cushion_Position_sig = float(LOF_target.describe().iloc[2, 0])
                Cushion_Position_mean = float(LOF_target.describe().iloc[1, 0])
                up = LOF_target.loc[LOF_target['Cushion_Position'] > Cushion_Position_mean + 2 * Cushion_Position_sig].index
                down = LOF_target.loc[LOF_target['Cushion_Position'] < Cushion_Position_mean - 2 * Cushion_Position_sig].index
                sig_list = list(up) + list(down)

                outlier = isol_list + LOF_list + sig_list
                x = []
                final = []

                for i in outlier:
                    if i not in x:
                        x.append(i)
                    else:
                        if i not in final:
                            final.append(i)     #fianl은 최종적으로 abnormal이라고 판단되는 index들의 리스트
                moldset_labeled = moldset_target.sort_index()
                moldset_labeled['Abnormal'] = np.nan
                moldset_labeled.fillna(value=0, inplace=True)
                moldset_labeled['Abnormal'] = moldset_labeled['Abnormal'].astype(int)
                for i in final:
                    moldset_labeled.loc[i, 'Abnormal'] = 1  #moldset_target 데이터 프레임에 labeling 완료
                moldset_labeled  #이 데이터프레임으로 RF돌린다

                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                import sklearn.metrics as metrics
                X_train, X_test, y_train, y_test = train_test_split(moldset_labeled[['Cushion_Position',
                                                                                     'Plasticizing_Position',
                                                                                     'Max_Injection_Pressure',
                                                                                     'Cycle_Time']],
                                                                    moldset_labeled['Abnormal'],
                                                                    test_size=0.25, random_state=0)
                clf = RandomForestClassifier(n_estimators=100)
                clf.fit(X_train, y_train)
                y_test_hat = clf.predict(X_test)
                st.write('accuracy : ', metrics.accuracy_score(y_test, y_test_hat))
                st.write('precision_score : ', metrics.precision_score(y_test, y_test_hat))
                st.write('recall_score : ', metrics.recall_score(y_test, y_test_hat))
                st.write('f1_score : ', metrics.f1_score(y_test, y_test_hat))



                
    elif choice == 'Autoencoder':
        st.subheader('Autoencoder')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding = 'utf-8', thousands = ',')
            Time_labled = get_file(df)
            moldset = df.iloc[:, 2:30]
            with st.expander('Plotting Normal/Abnormal'):
                product = list(moldset['Additional_Info_1'].unique())
                product_num = list()
                for i in product:
                    product_num.append(len(moldset.loc[moldset['Additional_Info_1'] == i]))
                product_df = pd.DataFrame({'product': product, 'number': product_num})
                target = product[product_num.index(max(product_num))]
                st.write('Our target is : ', target)

                moldset_target = moldset[moldset.Additional_Info_1.str.contains(target)]

                #moldset_target['TimeStamp'] = pd.to_datetime(moldset_target['TimeStamp'], format='%Y-%m-%d %hh %mm')
                #moldset_target['y_m_d'] = moldset_target['TimeStamp'].dt.strftime("%Y-%m-%d")

                columns_name = moldset_target.columns.tolist()
                del columns_name[0:5]

                moldset_target = moldset_target[columns_name]
                st.write(moldset_target.describe())
                labeled_target_corr = moldset_target.corr(method='pearson')

                loaded_model = load_model("models/autoencoder.pkl")
                #loaded_model.predict(moldset_target)
                #loaded_model.predict_proba(moldset_target)


                st.write('Our target is : ', target)
                
                fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)

                plt.scatter(Time_labled[Time_labled['anomaly']==1].index, Time_labled[Time_labled['anomaly']==1]['Max_Injection_Pressure'], c = 'b')
                plt.scatter(Time_labled[Time_labled['anomaly']==-1].index, Time_labled[Time_labled['anomaly']==-1]['Max_Injection_Pressure'], c = 'r')

                ax.set_title('Training Data', fontsize = 16)
                st.write(fig)

                product = list(Time_labled[Time_labled['anomaly']==-1].TimeStamp)
                product_injection = list(Time_labled[Time_labled['anomaly']==-1].Max_Injection_Pressure)
                product_switch = list(Time_labled[Time_labled['anomaly']==-1].Max_Switch_Over_Pressure)
                product_back = list(Time_labled[Time_labled['anomaly']==-1].Max_Back_Pressure)
                product_avgback = list(Time_labled[Time_labled['anomaly']==-1].Average_Back_Pressure)
                product_fill = list(Time_labled[Time_labled['anomaly']==-1].Filling_Time)
                product_df = pd.DataFrame({'product_time': product, 'product_injection': product_injection, 'product_switch': product_switch, 'product_back': product_back, 'product_avgback': product_avgback, 'product_fill': product_fill})
                product_df

    elif choice == 'OC-SVM':
        st.subheader('OC-SVM')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding = 'utf-8', thousands = ',')
            Time_labled = get_file_OC(df)
            moldset = df.iloc[:, 2:30]
            with st.expander('Plotting Normal/Abnormal'):
                product = list(moldset['Additional_Info_1'].unique())
                product_num = list()
                for i in product:
                    product_num.append(len(moldset.loc[moldset['Additional_Info_1'] == i]))
                product_df = pd.DataFrame({'product': product, 'number': product_num})
                target = product[product_num.index(max(product_num))]
                st.write('Our target is : ', target)

                moldset_target = moldset[moldset.Additional_Info_1.str.contains(target)]

                #moldset_target['TimeStamp'] = pd.to_datetime(moldset_target['TimeStamp'], format='%Y-%m-%d %hh %mm')
                #moldset_target['y_m_d'] = moldset_target['TimeStamp'].dt.strftime("%Y-%m-%d")

                columns_name = moldset_target.columns.tolist()
                del columns_name[0:5]

                moldset_target = moldset_target[columns_name]
                st.write(moldset_target.describe())
                labeled_target_corr = moldset_target.corr(method='pearson')

                loaded_model = load_model("models/autoencoder.pkl")
                #loaded_model.predict(moldset_target)
                #loaded_model.predict_proba(moldset_target)


                st.write('Our target is : ', target)
                
                fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)

                plt.scatter(Time_labled[Time_labled['anomaly']==1].index, Time_labled[Time_labled['anomaly']==1]['Max_Injection_Pressure'], c = 'b')
                plt.scatter(Time_labled[Time_labled['anomaly']==-1].index, Time_labled[Time_labled['anomaly']==-1]['Max_Injection_Pressure'], c = 'r')

                ax.set_title('Training Data', fontsize = 16)
                st.write(fig)

                product = list(Time_labled[Time_labled['anomaly']==-1].TimeStamp)
                product_injection = list(Time_labled[Time_labled['anomaly']==-1].Max_Injection_Pressure)
                product_switch = list(Time_labled[Time_labled['anomaly']==-1].Max_Switch_Over_Pressure)
                product_back = list(Time_labled[Time_labled['anomaly']==-1].Max_Back_Pressure)
                product_avgback = list(Time_labled[Time_labled['anomaly']==-1].Average_Back_Pressure)
                product_fill = list(Time_labled[Time_labled['anomaly']==-1].Filling_Time)
                product_df = pd.DataFrame({'product_time': product, 'product_injection': product_injection, 'product_switch': product_switch, 'product_back': product_back, 'product_avgback': product_avgback, 'product_fill': product_fill})
                product_df

    elif choice == 'RandomForest':
        st.subheader('RandomForest')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            moldset = df.iloc[:, 2:30]
            with st.expander('Plotting Normal/Abnormal'):
                st.write(df)
                product = list(moldset['Additional_Info_1'].unique())
                product_num = list()
                for i in product:
                    product_num.append(len(moldset.loc[moldset['Additional_Info_1'] == i]))
                product_df = pd.DataFrame({'product': product, 'number': product_num})
                target = product[product_num.index(max(product_num))]
                st.write('Our target is : ', target)

                moldset_target = moldset[moldset.Additional_Info_1.str.contains(target)]

                moldset_target['TimeStamp'] = pd.to_datetime(moldset_target['TimeStamp'], format='%Y-%m-%d')
                moldset_target['y_m_d'] = moldset_target['TimeStamp'].dt.strftime("%Y-%m-%d")

                columns_name = moldset_target.columns.tolist()
                del columns_name[0:5]

                moldset_target = moldset_target[columns_name]
                labeled_target_corr = moldset_target.corr(method='pearson')
                from sklearn.ensemble import IsolationForest
                labled = pd.DataFrame(moldset_target, columns=['Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
                                                            'Max_Back_Pressure', 'Average_Back_Pressure',
                                                            'Filling_Time'])
                Time_labled = pd.DataFrame(moldset_target, columns=['Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
                                                                'Max_Back_Pressure', 'Average_Back_Pressure',
                                                                'Filling_Time', 'y_m_d'])
                model = IsolationForest(n_estimators=100, max_samples='auto', n_jobs=-1,
                                        max_features=4, contamination=0.01)
                model.fit(labled.to_numpy())
                score = model.decision_function(labled.to_numpy())
                anomaly = model.predict(labled.to_numpy())
                labled['scores'] = score
                labled['anomaly'] = anomaly
                anomaly_data = labled.loc[labled['anomaly'] == -1]
                final_INNER_JOIN = pd.merge(Time_labled, anomaly_data, left_index=True, right_index=True, how='inner')
                final = final_INNER_JOIN.groupby('y_m_d').sum()
                labled.reset_index(drop=True)
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(n_estimators=50, max_samples=120, contamination=float(0.004),
                                    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=None, verbose=0)
                #50개의 노드 수, 최대 50개의 샘플
                #0.4%의 outlier 색출.
                clf.fit(labled)
                pred = clf.predict(labled)
                labled['anomaly'] = pred
                outliers = labled.loc[labled['anomaly'] == -1]
                outlier_index = list(outliers.index)
                isol_list = list(moldset_target.iloc[outlier_index, :].index)

                #LOF

                LOF_target = moldset_target[['Cushion_Position', 'Plasticizing_Position']]
                from sklearn.neighbors import LocalOutlierFactor
                # model specification
                model1 = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
                # model fitting
                y_pred = model1.fit_predict(LOF_target)
                # filter outlier index
                outlier_index = np.where(y_pred == -1)   #negative values are outliers and positives inliers
                # filter outlier values
                outlier_values = LOF_target.iloc[outlier_index]
                # LOF를 이용해 구한 9000R의 outlier index
                LOF_list = list(LOF_target.iloc[list(outlier_index[0]), :].index)

                #3시그마

                Cushion_Position_sig = float(LOF_target.describe().iloc[2, 0])
                Cushion_Position_mean = float(LOF_target.describe().iloc[1, 0])
                up = LOF_target.loc[LOF_target['Cushion_Position'] > Cushion_Position_mean + 2 * Cushion_Position_sig].index
                down = LOF_target.loc[LOF_target['Cushion_Position'] < Cushion_Position_mean - 2 * Cushion_Position_sig].index
                sig_list = list(up) + list(down)

                outlier = isol_list + LOF_list + sig_list
                x = []
                final = []

                for i in outlier:
                    if i not in x:
                        x.append(i)
                    else:
                        if i not in final:
                            final.append(i)     #fianl은 최종적으로 abnormal이라고 판단되는 index들의 리스트
                moldset_labeled = moldset_target.sort_index()
                moldset_labeled['Abnormal'] = np.nan
                moldset_labeled.fillna(value=0, inplace=True)
                moldset_labeled['Abnormal'] = moldset_labeled['Abnormal'].astype(int)
                for i in final:
                    moldset_labeled.loc[i, 'Abnormal'] = 1  #moldset_target 데이터 프레임에 labeling 완료

                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                import sklearn.metrics as metrics
                X_train, X_test, y_train, y_test = train_test_split(moldset_labeled[['Cushion_Position',
                                                                                    'Plasticizing_Position',
                                                                                    'Max_Injection_Pressure',
                                                                                    'Cycle_Time']],
                                                                    moldset_labeled['Abnormal'],
                                                                    test_size=0.25, random_state=0)
                clf = RandomForestClassifier(n_estimators=100)
                clf.fit(X_train, y_train)
                y_test_hat = clf.predict(X_test)

                
                fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)

                plt.scatter(moldset_labeled[moldset_labeled['Abnormal']==0].index, moldset_labeled[moldset_labeled['Abnormal']==0]['Max_Injection_Pressure'], c = 'b')
                plt.scatter(moldset_labeled[moldset_labeled['Abnormal']==1].index, moldset_labeled[moldset_labeled['Abnormal']==1]['Max_Injection_Pressure'], c = 'r')

                ax.set_title('Training Data', fontsize = 16)
                st.write(fig)

                product_injection = list(moldset_labeled[moldset_labeled['Abnormal']==1].Max_Injection_Pressure)
                product_switch = list(moldset_labeled[moldset_labeled['Abnormal']==1].Max_Switch_Over_Pressure)
                product_back = list(moldset_labeled[moldset_labeled['Abnormal']==1].Max_Back_Pressure)
                product_avgback = list(moldset_labeled[moldset_labeled['Abnormal']==1].Average_Back_Pressure)
                product_fill = list(moldset_labeled[moldset_labeled['Abnormal']==1].Filling_Time)
                product_df = pd.DataFrame({'product_injection': product_injection, 'product_switch': product_switch, 'product_back': product_back, 'product_avgback': product_avgback, 'product_fill': product_fill})
                product_df


            

if __name__ == '__main__':
    main()
