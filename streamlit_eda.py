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


def main():
    menu = ['CsvFiles', 'ml_app']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'CsvFiles':
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


                
    else:
        st.subheader('ml_app')
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

if __name__ == '__main__':
    main()
