import streamlit as st
import numpy as np
import scipy
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.graph_objects as pxx
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
import sympy as smp
import sklearn
from sklearn.linear_model import LogisticRegression
import pandera as pa
from pandera.typing import DataFrame, Series
import seaborn as sns
from pandera import Column, DataFrameSchema
from sklearn.ensemble import RandomForestRegressor
#st.session_state["validation"] is the data frame used to store the number of data with errors based on data validation
if "validation" not in st.session_state:
     st.session_state["validation"] = []
#st.session_state["type"] is the data frame used to store the number of data with errors in the data type (used for Completeness)
if "type" not in st.session_state:
     st.session_state["type"] = []
def remove_outliers(data, column, z_threshold=3):
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    return data[z_scores < z_threshold]
def remove_duplicates(data):
    return pd.DataFrame.drop_duplicates(data)
def remove_nulls(S1):
    return S1.drop(S1[S1.isna().any(axis=1)].index, inplace = True)
def read_csv(source_data):
    df = pd.read_csv(source_data)
    return df 
def read_excel(source_data):
    df = pd.read_excel(source_data)
    return df
def hig(failure_cases,df,i):#make a dataframe that highlights wrong data values and a table shows failure cases
    col1,col2 = st.columns([1,3])#adjust the positions of dataframe and table
    with col1: #table shows failure cases
        st.dataframe(failure_cases.iloc[:,[-2,-1,-4,-3]])
    failure = failure_cases["failure_case"]
    validation_check = []
    # Summarize the number of wrong values
    if len(failure)==0: 
        st.write("There are ",0,"rows in our error result table(left)")
    else:
        for ww in range(len(failure_cases["check"].unique())):
            st.write(failure_cases["check"].unique()[ww],": There are" ,
                     len(failure_cases.loc[failure_cases["check"]==failure_cases["check"].unique()[ww]]),"wrong rows")
            validation_check.append(len(failure_cases.loc[failure_cases["check"]==failure_cases["check"].unique()[ww]]))
    def highlight_right(df): #hightlight all rows include wrong values except the column include wrong values 
        if df[i] in list(failure):
            return ['background-color: yellow'] * len(df)
        else:
            return ['background-color: white'] * len(df)
    def highlight_wrong(df): #only hightlight the wrong values
        if df in list(failure) :
            color = "green"
        else:
            color = 'white'
        return f'background-color: {color}'
    with col2: #dataframe that highlights wrong data values 
        openn = df.style.apply(highlight_right, axis=1)
        st.dataframe(openn.applymap(highlight_wrong, subset=[i]))
    st.session_state["validation"].append(sum(validation_check))#Record the number of wrong values in st.sessoin_state
                                                                #                    in st.sessoin_state["validaition"] 
def main():
    df = None
    Non = None
    with st.sidebar.header("Source Data Selection"): #Upload file
        selection = ["csv",'excel'] # create selectbox for uploading file
        selected_data = st.sidebar.selectbox("Please select your dataset format:",selection)
        if selected_data is not None:
            if selected_data == "csv":# Upload csv
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.csv) data", type = ["csv"])
                if source_data is not None: 
                    df = pd.read_csv(source_data) # store the file as df      
            elif selected_data == "excel": # Upload excel
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.xlsx) data", type = ["xlsx"])
                if source_data is not None:
                    df = pd.read_excel(source_data)# store the file as df      
                   
        
       

        st.header("Dataset")   
        
    if df is not None:
        user_choices = ["Data Quality","Data Validation","Score"]#menu
        selected_choices = st.sidebar.selectbox("Please select your choice:",user_choices)
        word = []#if there are classification columns in the dataset, their name will store here
        dff = df.copy()
        #sidebar for choosing classification columns, and choose if O will conisder to be missing values
        st.sidebar.write("Would you consider 0 as missing value in the dataset (except classification column)?",key=f"MyKey{13}")
        y =  st.sidebar.checkbox("Yes",key=f"MyKey{3223}")
        n =  st.sidebar.checkbox("No",key=f"MyKey{33333}")
        if y:# conisder the situation where 0 is missing values
            st.sidebar.write("Is there any columns for classification in the dataset?",key=f"MyKey{123}")
            Y = st.sidebar.checkbox("Yes")
            N = st.sidebar.checkbox("No")
            select = df.keys()
            if Y:
                st.sidebar.write("Which columns are classification columns?",key=f"MyKey{133}")
                for i in select:
                    X = st.sidebar.checkbox(i)
                    if X:
                        word.append(i)#store classification columns' name in "word" list 
                    elif not X:
                        df[i].replace(0,np.nan,inplace=True)#Replace 0 to NaN
            elif N:
               
                df.replace(0,np.nan,inplace=True)#Replace 0 to NaN                 
        elif n:# conisder the situation where 0 isn't missing values
            df=dff
            st.sidebar.write("Are there any columns for classification in the dataset?",key=f"MyKey{123}")
            Y = st.sidebar.checkbox("Yes")
            N = st.sidebar.checkbox("No")
            select = df.keys()
            if Y:
                st.sidebar.write("Which columns are classification columns?",key=f"MyKey{3433}")
                for i in select:
                    X = st.sidebar.checkbox(i)
                    if X:
                         word.append(i)#store classification columns' name in "word" list  
        
        if "dataset" not in st.session_state:
                st.session_state["dataset"] = df #Store df into st.session_state["dataset"]
        col1,col2 = st.columns([10,1])#put "restart" buttion and "Data Sample" expander together,
                                      # and adjust the positions for the buttion and  expander                                                 
        st.markdown(  #adjust the size of "restart" buttion,
            """
<style>
button {
    height: auto;
    padding-top: 10.5px !important;
    padding-bottom: 10.5px !important;
}
</style>
""",
    unsafe_allow_html=True,
            )
        #Data Sample Expander
        with col1.expander("Data Sample"):
            st.info("Selected dataset has "+str(st.session_state["dataset"].shape[0])+" rows and "+str(st.session_state["dataset"].shape[1])+" columns.")
            st.write(st.session_state["dataset"])
        # restart button, let everything go back to original version
        if col2.button("Restart"):
            st.session_state["dataset"] = df
            st.session_state["type"] = []
            st.session_state["validation"]=[]
   
        if selected_choices is not None:                     
            if  selected_choices == "Data Quality":#Data quality module
                st.subheader("Data Quality")
                box = ["Data types","Descriptive statistics","Missing values","Duplicate records",
                     "Correlation", "Outliers","Data distribution","Random Forest"]
                selection = st.selectbox("Data Quality Selection",box,key=f"MyKey{4}") 
                
                if selection is not None:
                    if selection == "Data types": #Data types
                        types = pd.DataFrame(df.dtypes)
                        types.rename(columns={0:"Type of Data"},inplace = True)
                        df_types = types.astype(str)
                        st.write(df_types)
                    elif selection == "Descriptive statistics": #Descriptive statistics
                        describe = pd.DataFrame(df.describe())
                        # a multiselect box to check specific column's information
                        selection = st.multiselect("Please select any data you want to check",describe.keys(),key=f"MyKey{555}")
                        df_describe = None
                        if len(selection)>0:
                            df_describe = pd.DataFrame({})
                            for i in selection:
                                 df_describe[i]=describe[i]
                       
                            st.write(df_describe.T)# for certain coulmn's describtion
                        else:
                            st.table(describe.T)# for entire dataset describtion
                            
                    elif selection == "Missing values":#Missing values  
                        dff = st.session_state["dataset"] #read st.session_state["dataset"] as dff
                        summary_null = pd.DataFrame(dff.isnull().sum())           
                        summary_null.rename(columns = {0:'Summary for Missing Values'}, inplace = True)
                        df_summary_null = summary_null.astype(str)            
                        st.write(df_summary_null.T)# a table summarize the missing values for each column
                        #visualize on heatmap
                        fig = plt.figure(figsize=(15,10))
                        if sum(dff.isnull().sum().values)!=0:
                            graph=sns.heatmap(dff.isna().transpose(),cmap="YlGnBu",cbar=False,cbar_kws={'label': 'Missing Data'})
                            graph.set_yticklabels(graph.get_ymajorticklabels(), fontsize = 16)
                            plt.savefig("visualizing_missing_data_with_heatmap_Seaborn_Python.png", dpi=100)
                            plt.title("Visualizing Missing Data with Heatmap Seaborn", fontdict={'fontsize': 24})
                            plt.yticks(rotation=0) 
                            st.pyplot(fig)
                        #List missing values for different columns
                        box = dff.keys()
                        select = st.selectbox("Show missing values",box,key=f"MyKey{5}")
                        for i in box:
                            if select == i:
                                st.write(dff[pd.isnull(dff[i])])
                        #Remove missing values
                        if st.checkbox("Remove Rows with Missing Values"):#Remove missing rows
                            st.session_state["dataset"] = dff.dropna(axis=0)
                            st.success("Missing values has successfully removed！")
                       
                        if st.checkbox("Remove Columns with Missing Values"):#Remove missing columns
                            selection = st.multiselect("Please select column you want to drop",dff.keys(),key=f"MyKey{550}")
                            if len(selection)!=0:
                                if st.button("drop"):#Remove certain columns you want
                                    st.session_state["dataset"] = dff.drop(columns=selection)
                                    st.success("Missing values has successfully removed！")
                            if st.button("Remove all columns with Missing Values"):#Remove missing columns
                                st.session_state["dataset"] = dff.dropna(axis=1)
                                st.success("Missing values has successfully removed！")
                  
                    elif selection == "Duplicate records":#Duplicate records
                        dff = st.session_state["dataset"]#read st.session_state["dataset"] as dff
                        duplicate = dff[dff.duplicated()]
                        df_duplicate = duplicate.astype(str)
                        #visualize on heatmap
                        fig = plt.figure(figsize=(15,10))
                        if len(duplicate)!=0:
                            data_duplicate = pd.DataFrame(dff.duplicated())
                            graph=sns.heatmap(data_duplicate,cmap="YlGnBu",xticklabels=False,cbar=False)
                            plt.title("Visualizing Duplicate Data with Heatmap Seaborn", fontdict={'fontsize': 24})
                            plt.savefig("visualizing_duplicate_data_with_heatmap_Seaborn_Python.png", dpi=100)
                            plt.yticks(rotation=0) 
                            plt.xlabel("Columns",fontdict={'fontsize': 17})
                            plt.ylabel("Index",fontdict={'fontsize': 17})
                            fig.tight_layout()
                            st.pyplot(fig)
                        #Summarize duplicated rows, and list all duplicated rows
                        st.write("The number of duplicated rows is ",len(duplicate))
                        st.write(df_duplicate)
                        #Remove duplicated rows
                        if st.button("Remove Duplicate Values"):
                            st.session_state["dataset"] = remove_duplicates(st.session_state["dataset"])
                            st.success("Duplicate values has successfully removed！")
                    elif selection == "Outliers":#Outliers
                        dff = st.session_state["dataset"]
                        #visualize on barplot
                        fig = plt.figure(figsize=(15,20))
                        box = dff.select_dtypes(include=['int',"float"])
                        for i in range(len(box.keys())):
                            plt.subplot(len(box.keys()),1,i+1)
                            sns.boxplot(x=dff[box.keys()[i]])
                            plt.xlabel(box.keys()[i],fontsize=18)  
                        fig.tight_layout()
                        st.pyplot(fig)
                        #an expander which include all outliers in the dataset 
                        with st.expander("See outliers"):
                            for i in range(len(box.keys())):
                                z_scores = np.abs((box[box.keys()[i]] - box[box.keys()[i]].mean()) / box[box.keys()[i]].std())
                                st.write(box.keys()[i])
                                st.write(box[z_scores >= 3])
                                if st.button("Remove",key=f"MyKey{1000+i}"):
                                    st.session_state["dataset"] = remove_outliers(st.session_state["dataset"], box.keys()[i])
                             
                        #Remove exterme values 
                        if st.button("Remove All Outlier"):
                                for i in box.keys():
                                    st.session_state["dataset"] = remove_outliers(st.session_state["dataset"], i)
                                st.success("Outlier Values has successfully removed!")
                                   
                    elif selection == "Data distribution": #Data Distribution
                        df =  st.session_state["dataset"]
                        data = df.select_dtypes(include=['int',"float"])#only include int/float column to make graph
                        box = df.keys()
                        selection = st.selectbox("Select which column you want to check",box,key=f"MyKey{6}")
                        for i in box:
                            #if this column is not classification column, make histogram, scatter, and line graph
                            if selection == i and selection not in word:
                                tab1,tab2,tab3 = st.tabs(["Histogram graph","Scatter graph","Line graph"])
                                with tab1:#Histogram graph
                                    fig = plt.figure(figsize=(4,3))  
                                    x0 = None
                                    graph_range = st.checkbox("Change graph range",key=f"MyKey{67551}")
                                    if graph_range:#set up x and y axis range
                                            x0,x1 = st.slider("If you want to change the range of the graph, tell me the x axis range",min(df[i]), max(df[i]),(min(df[i]), max(df[i])),key=f"MyKey{67}")
                                            y0,y1 = st.slider("If you want to change the range of the graph, tell me the y axis range",0,max(df[i].value_counts()*4),(0,max(df[i].value_counts())),key=f"MyKey{700}")
                                    if word != []:#classify graph by classificaiton column
                                        select = st.selectbox("What classification condition do you want?",word,key=f"MyKey{321}")
                                        for j in word: 
                                            if select == j: 
                                                sns.histplot(data = df,x=i,binwidth=3,kde=True,hue=j)
                                                if x0 is not None:
                                                    plt.xlim(x0, x1)
                                                    plt.ylim(y0, y1)    
                                                st.pyplot(fig)
                                                
                                    elif word ==[]:#make graph without any claddification
                                        sns.histplot(data = df,x=i,binwidth=3,kde=True)
                                        if x0 is not None:# change range of x, y axis
                                            plt.xlim(x0, x1)
                                            plt.ylim(y0, y1)    
                                        st.pyplot(fig)
                                                      
                
                                with tab2:#Scatter graph
                                    fig = plt.figure(figsize=(4,3))
                                    x0 = None
                                    df[" "]=np.arange(len(df))
                                    graph_range = st.checkbox("Change graph range",key=f"MyKey{6755}")
                                    if graph_range:#set up x and y axis range
                                            x0,x1 = st.slider("If you want to change the range of the graph, tell me the x axis range",min(df[i]), max(df[i]),(min(df[i]), max(df[i])),key=f"MyKey{699}")
                                            y0,y1 = st.slider("If you want to change the range of the graph, tell me the y axis range",min(df[" "]),max(df[" "]),(min(df[" "]),max(df[" "])),key=f"MyKey{7000}")
                                    if word != []:#classify graph by classificaiton column
                                        select = st.selectbox("What classification condition do you want?",word,key=f"MyKey{1128}")
                                        for j in word: 
                                            if select == j: 
                                                sns.scatterplot(data = df,x=i,y=" ",hue=j)
                                                if x0 is not None:
                                                    plt.xlim(x0, x1)
                                                    plt.ylim(y0, y1)
                                                st.pyplot(fig)
                                    elif word ==[]:#make graph without any claddification
                                        sns.scatterplot(data = df,x=i,y=" ")
                                        if x0 is not None:# change range of x, y axis
                                            plt.xlim(x0, x1)
                                            plt.ylim(y0, y1)
                                        st.pyplot(fig)
                                with tab3:#Line graph
                                    fig = plt.figure(figsize=(4,3))
                                    df["range"]=np.arange(len(df))
                                    x0 = None
                                    graph_range = st.checkbox("Change graph range",key=f"MyKey{65}")
                                    if graph_range:#set up x and y axis range         
                                            x0,x1 = st.slider("If you want to change the range of the graph, tell me the x axis range",min(df["range"]),max(df["range"]),(min(df["range"]),max(df["range"])),key=f"MyKey{701}")
                                            y0,y1 = st.slider("If you want to change the range of the graph, tell me the y axis range",min(df[i]), max(df[i]),(min(df[i]), max(df[i])),key=f"MyKey{67055}")
                                    if word != []:#classify graph by classificaiton column
                                        select = st.selectbox("What classification condition do you want?",word,key=f"MyKey{1122}")
                                        for j in word:
                                            if select == j: 
                                                sns.lineplot(data = df,x="range",y=i,hue=j)
                                                if x0 is not None:
                                                    plt.xlim(x0, x1)
                                                    plt.ylim(y0,y1)
                                                st.pyplot(fig)
                                    elif word ==[]:#make graph without any claddification
                                        sns.lineplot(data = df,x="range",y=i)
                                        if x0 is not None:# change range of x, y axis
                                            plt.xlim(x0, x1)
                                            plt.ylim(y0,y1)
                                        st.pyplot(fig)
                            #if this column is classification column, make pie graph  
                            elif selection ==i and selection in word:
                                tab1,tab2 = st.tabs(["Pie","  "])#keep pie graph with same size as line/scatter/hist graph
                                with tab1:
                                    fig = plt.figure(figsize=(3,3))
                                    pie_plot=df.groupby([i]).size().plot(kind='pie', y='counts',autopct='%1.0f%%')
                                    pie_ploy.set_ylabel('Counts', size=11)
                                    st.pyplot(fig)
                               
                       
                    elif selection == "Correlation":#Correlation
                        df =  st.session_state["dataset"]
                        new_df = df.select_dtypes(include=['int',"float"])#only include int/float columns to do correlation
                        box = new_df.keys()
                        new = {}
                        new = pd.DataFrame(new)
                        container = st.container()
                        all = st.checkbox("Select all")
                        #make multiselection module
                        if all:
                            selected_options = container.multiselect("Select one or more options:",
                            list(box),list(box))#select all selections
                        else:
                            selected_options =  container.multiselect("Select one or more options:",
                            list(box))#select certain selections
                 
                        #Make correlation graph
                        for i in range(len(selected_options)):
                            new[selected_options[i]]=df[selected_options[i]]
                        
                        if new.empty == False:
                            fig,ax = plt.subplots()
                            if len(new.keys())<6:
                                sns.heatmap(new.corr(),annot = True,ax=ax)
                            else:
                                 sns.heatmap(new.corr(),ax=ax)
                            st.pyplot(fig)
                       #Explain correlation in words
                        for i in range(len(new.corr().keys())):
                            for j in range(len(new.corr().index.values)):
                                if new.corr().keys()[i]!=new.corr().index.values[j]:
                                    # based on the score of corr graph to identify if two columns have strong/week correlation
                                    if new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] >=0.8:
                                        st.write(new.corr().keys()[i],"and",new.corr().index.values[j],"are strongly positively correlated")
                                    elif new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] >=0.5 and new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] <0.8:
                                        st.write(new.corr().keys()[i],"and",new.corr().index.values[j],"are positively correlated")
                                    elif new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] >=0. and new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] <0.5:
                                        st.write(new.corr().keys()[i],"and",new.corr().index.values[j],"are weakly positively correlated")
                                    elif new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] <=-0.8:
                                        st.write(new.corr().keys()[i],"and",new.corr().index.values[j],"are strongly negatively correlated")
                                    elif new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] <=-0.5 and new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] >-0.8:
                                        st.write(new.corr().keys()[i],"and",new.corr().index.values[j],"are negatively correlated")
                                    elif new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] <=0. and new.corr()[new.corr().keys()[i]][new.corr().index.values[j]] >-0.5:
                                        st.write(new.corr().keys()[i],"and",new.corr().index.values[j],"are weakly negatively correlated")
   
                        
                    elif selection == "Random Forest":#Random Forest
                        df =  st.session_state["dataset"]
                        new_df = df.select_dtypes(include=['int',"float"])
                        dfff = new_df.dropna(axis=1)#new dataframe which removed nan values
                        select = st.selectbox("Please choice dependent value",dfff.keys())
                        for i in dfff.keys():
                            #make random forest depend on certain column
                            if select == i:
                                y,X = dfff[i],dfff.drop(columns=[i])
                                x_train,x_test,y_train,y_test = train_test_split(X.values,y.values,test_size=0.3,random_state=0)
                                regressor = RandomForestRegressor(n_estimators=100,
                                  random_state=0)
                                regressor.fit(x_train, y_train)
                                importances = regressor.feature_importances_
                                indices = np.argsort(importances)[::-1]
                                fig = plt.figure(figsize=(4,3))
                                plt.ylabel("Feature importance")
                                plt.bar(range(x_train.shape[1]),importances[indices],align="center")
                                feat_labels = X.columns[0:]
                                plt.xticks(range(x_train.shape[1]),feat_labels[indices],rotation=60)
                                plt.xlim([-1,x_train.shape[1]])
                                st.pyplot(fig)
                                
            elif selected_choices == "Data Validation": #Data Validation module
                st.subheader("Data Validation")
                df =  st.session_state["dataset"]
                box = df.keys()
                select = st.selectbox("Choose columns",box,key=f"MyKey{12}")
                for i in box:  
                    if select is not None :
                        #do data validation when this column isn't classification column
                        if select == i and i not in word:
                            #check if this column is empty
                            if df[i].isnull().sum()==len(df[i]):
                                st.error("this column is empty")
                            else:
                                st.write("What type of data is this in the column?")
                                #check data validation for int/float data
                                check_int = st.checkbox('Int/Float')
                                case = [] #sum of failure_cases will be add there
                        
                                if check_int:
                                    int_float = df[i].apply(pd.to_numeric, errors='coerce')#check if all values in the column  
                                                                                           #are int/float
                                    df[i] = int_float
                                    st.session_state["type"].append(len(int_float[int_float.isna()]))#record number of wrong 
                                                                                           #values into st.session_state["type"]
                              
                                    st.write("What is the range of your data?")
                                    minimum = st.number_input('Insert the minimum value')#set up max of the range
                                    
                                    maximum = st.number_input('Insert the maximum value')#set up min of the range
                                   
                         
                                    #check error result
                                    if st.button("Error Result"):
                                        schema = pa.DataFrameSchema({
                                      i: pa.Column(float, pa.Check.in_range(minimum,maximum),coerce=True)                                                                                                                       })
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                                failure_cases = err.failure_cases
                                                case.append(len(err.failure_cases))
                                                hig(failure_cases,df,i)#create table and graph shows error values
                                        if len(case)==0: #if error result is 0,report this
                                              st.write("There are ",0,"rows out of range")
                                       
                                #check data validation for str data
                                check_str = st.checkbox('Str')
                                if check_str:
                                    #check if all values in column are str, use same method as checking int/float 
                                    # but record NaN/wrong rows as right values
                                    Not_Str = df[i].apply(pd.to_numeric, errors='coerce')
                                    st.session_state["type"].append(len(df)-len(Not_Str[Not_Str.isna()]))#Record number of 
                                                                                                         #not str values
                      
                   
                                    st.write("What is the range of your data?")
                                    minimum = st.number_input('Insert the minimum length of your string')-1#set up max of the range
                                    maximum = st.number_input('Insert the maximum length of your string')+1#set up min of the range
                                    text = st.text_input("If you want, type the Regular Expression of your data")
                                    #check error result
                                    if st.button("Error Result",f"MyKey{43}"):
                                        case = []
                                        schema = pa.DataFrameSchema({
                                        i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                                         pa.Check(lambda x: len(x) < maximum, element_wise=True),
                                                          pa.Check.str_matches(text)])})
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                            failure_cases = err.failure_cases
                                            case.append(err.failure_cases["failure_case"].duplicated().sum())
                                            hig(failure_cases,df,i)#create table and graph shows error values
                                        if len(case)==0: #if error result is 0,report this
                                             st.write("There are ",0,"rows out of range")
                                                
                                #check data validation for datetime data       
                                date_check = st.checkbox('Date/Time')
                                if date_check:
                                    #check if all values in column are datetime
                                    df[i]=pd.to_datetime(df[i],format="mixed",errors="coerce").dt.strftime("%Y-%m-%d")
                                    st.session_state["type"].append(len(df[i][df[i].isna()]))#Record number of 
                                                                                             #not datetime values
                                    st.write("What is the range of your data?")
                                    minimum = st.date_input('Insert the minimum date')#set up max of the range
                                    maximum = st.date_input('Insert the maximum date')#set up min of the range
                                    
                                    #Error Result
                                    if st.button("Error Result",key=f"MyKey{34333}"):
                                            df[i]= df[i].astype('datetime64[ns]').dt.tz_localize(None)
                                            case =[]
                                            schema = pa.DataFrameSchema({
                                            i: pa.Column("datetime64[ns]",[pa.Check(lambda x: x >         
                                                      pd.to_datetime(minimum),element_wise=True),
                                                      pa.Check(lambda x: x < pd.to_datetime(maximum),element_wise=True) ]               
                                                             )})  
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err:
                                                failure_cases = err.failure_cases
                                                case.append(len(err.failure_cases))
                                                hig(failure_cases,df,i)#create table and graph shows error values
                                            if len(case)==0:#if error result is 0,report this
                                                st.write("There are ",0,"rows out of range")
                                                
                                                
                                #check if all values in this column are unique     
                                st.write("Are all values in the column unique？") 
                                Y = st.checkbox("Yes",key=f"MyKey{32}")
                             
                                if Y:
                                    #report duplicated result
                                    st.write("In",i,"there are ",len(df[df[i].duplicated()]),"rows with repeated values.")
                                    st.write(df[df[i].duplicated()])
                        #do data validation when this column is classification column
                        elif select == i and i in word:
                            #check if this column is empty
                            if df[i].isnull().sum()==len(df[i]):
                                st.error("this column is empty")
                            else: 
                                options = st.radio("What sorting options are in this column？",
                                                         ("number","text"))
                                # when it is number classification
                                if options == "number":
                                    case = []
                                    number = st.number_input('How many categories are there？')
                                    array = np.arange(int(number))#base on the number, list all classification numbers
                                    if st.button("Error Result",key=f"MyKey{52}"):
                                            schema = pa.DataFrameSchema({
                                             i: pa.Column(int,pa.Check.isin(array), coerce=True)                                                                                                                       })
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err: 
                                                failure_cases = err.failure_cases
                                                case.append(len(err.failure_cases))
                                                hig(failure_cases,df,i)#create table and graph shows error values
                                            if len(c)==0:
                                                st.write("There are ",0,"rows out of range")
                                # when it is text classification           
                                elif options == "text":
                                    array_text = []#all classification text will be list here
                                    case = []
                                    number = st.number_input('How many categories are there？')
                                    for ii in range(int(number)):
                                        text = st.text_input("insert name of category",key=f"MyKey{ii}")
                                        array_text.append(text)
                                    if st.button("Error Result",f"MyKey{42}"):
                                        schema = pa.DataFrameSchema({
                                         i: pa.Column(str,pa.Check.isin(barr), coerce=True)                                                                                                                       })
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                            failure_cases = err.failure_cases
                                            case.append(len(err.failure_cases))
                                            hig(failure_cases,df,i)#create table and graph shows error values
                                          
                                        if len(c)==0:
                                            st.write("There are ",0,"rows out of range")
            elif  selected_choices == "Score": #Score module
                        #Data Quality Score
                        st.subheader("Data Quality Score")
                        df =  st.session_state["dataset"]
                        extreme_values = []
                        box = df.keys()
                        missing_value =len(df[df.isna().any(axis=1)])
                        duplicated_rows = df.duplicated().sum()
                        for i in box:
                            if df[i].dtypes == "int64" or  df[i].dtypes == "float64":
                                z_scores = np.abs((df[i] - df[i].mean()) / df[i].std())
                                extreme_values.append(len(df[z_scores >= 3]))
                        error = sum(extreme_values)+missing_value+duplicated_rows
                        #Summarize the data quality score
                        st.write("number of missing values in the dataset is",missing_value)
                        st.write("number of duplicated rows in the dataset is",duplicated_rows)
                        st.write("number of extreme values in the dataset is", sum(extreme_values))
                        st.write("the dataset has",len(df),"rows")
                        if round(100*(1-error/len(df)))<0:#if score is negative, then it means 0
                            st.write("Overall, the score of data is ",0)
                        else:
                            st.write("Overall, the score of data is ",round(100*(1-error/len(df))))
                        #equation about how we calculate the score
                        st.latex(r'''score = (missing+extreme+duplication)/total''')
                        #Data Validation Score
                        st.subheader("Data Validation Score")
                        st.info("The result is calculated by the rules you set in data validation, if you do not make any adjustments in data validation, the result will be 100 ")
                        non_classify = df.drop(columns=word)
                        non_classify = non_classify.dropna(axis=1)#clean data,used to calculate accuracy score
                        new_df = pd.DataFrame({})
                        accuracy = []
                        if word != []:
                            #calculate accuracy score
                            for i in word:
                                new_df[i]=df[i]
                                X_train, X_test, y_train, y_test = train_test_split(non_classify, new_df[i], test_size=0.3, random_state=42)
                                clf = LogisticRegression(random_state=42)
                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                accuracy.append(accuracy_score(y_test, y_pred))
                        if len(accuracy) != 0:
                            final_accuracy = sum(accuracy)/len(accuracy)#accuracy score
                        else:
                            final_accuracy = 1
                        #set up different score's weight
                        accuracys_weight = st.number_input("[1] Weight of Accuracy Score")
                        completeness_weight = st.number_input("[2] Weight of Completeness Score")
                        validation_weight = st.number_input("[3] Weight of Validation Score")
                        #calculate final score
                        score=round(final_accuracy*accuracys_weight*100)+completeness_weight*round(100*(1-missing_value/len(df)-sum(st.session_state["type"])/len(df)))+round(100*(1-sum(st.session_state["validation"])/len(df)))*validation_weight
                        st.latex(r'''score = \frac{[1]*Accuracy Score+[2]*CompletenessScore+[3]*ValidationScore}{total weight}*100''')
                        if st.button("Final Score"):
                           
                            #st.write(st.session_state["validation"])
                            st.write("Accuracy :",round(final_accuracy*100))
                            st.write("Completeness :",round(100*(1-missing_value/len(df)-sum(st.session_state["type"])/len(df))))
                            st.write("Validation :",round(100*(1-sum(st.session_state["validation"])/len(df))))
                            st.write("Final Score :",score/(100*(accuracys_weight+completeness_weight+validation_weight))*100)
                           
         
         
    
    else:
        st.error("Please select your data to started")

main()
