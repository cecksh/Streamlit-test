
import streamlit as st
import streamlit.components.v1 as stc
from datetime import datetime
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')


def get_readable_time(mytime):
    return datetime.fromtimestamp(mytime).strftime('%Y%m%d-%H%M%S')




def main():
    menu = ['CsvFiles', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'CsvFiles':
        st.subheader('CSV files MetaData Extraction')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            moldset = df.iloc[:, 2:30]
            st.write(df)
            with st.expander('File Stats'):
                file_details = {'FileName': uploaded_file.name, 'FileSize': uploaded_file.size,
                                'FileType': uploaded_file.type, 'FileLength': len(df)}
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
                moldset_target

                moldset_target['TimeStamp'] = pd.to_datetime(moldset_target['TimeStamp'], format='%Y-%m-%d')
                moldset_target['y_m_d'] = moldset_target['TimeStamp'].dt.strftime("%Y-%m-%d")

                columns_name = moldset_target.columns.tolist()
                del columns_name[0:5]

                moldset_target = moldset_target[columns_name]
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
        st.subheader('About')


main()
