from tensorflow import keras
import numpy as np
import geopandas as gpd
import pandas as pd
import dask_geopandas as dask_gpd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from itertools import chain
from sklearn import metrics
import matplotlib.pyplot as plt
import time

#time the script running time
start_time = time.time()

#file that has species points to include how many observations fall into certain polygon
filename_species = r"YourFilePath"
ddf_species = dask_gpd.read_file(filename_species, npartitions=4)
speciespoints = ddf_species.compute()

#file that includes OGF overlap percentage to asses similarity
filename = r"YourFilePath"

#read as dask (faster) and convert to geopandas for processing
ddf = dask_gpd.read_parquet(filename, npartitions=2)
#ddf = dask_gpd.read_file(filename, npartitions=2) # if other than parquet
df = ddf.compute()

#Calculate the percentage of null values per column and drop the ones that contain over 99% nulls
filter = (df.isnull().mean() * 100)>99
drop_me = filter[filter].index.to_list()
df = df.drop(columns=drop_me)
len(df)

#fill nulls with zeros as null means no observations in certain columns
Columns = ['Kasvupaikka','Kiinteisto1_5haRakennusPerc','Rakennukset30mPerc','Ojitus_Perc','All90PercEtela4KL_BTKUVuusiA_Peitto','GFW_Perc','MKI_Perc','MantyVol_Osuus','LehtipVol_Osuus','MantyVol_Mean','LehtipVol_Mean']
Columns_drop = ['Natura_Perc','Suojelu_Perc'] #dont want to teach correlation between presservation and OGF as non protected are the main interest, in theory, no correlation either
Columns_drop2 = ['Omistaja040222','MH_KiinteistoKP'] # in theory, ownership is not correlated with OGF formation
df[Columns] = df[Columns].fillna(0)

#additional string-typed parameters can be converted to int-type to be used as classification input
#here forest vegetation zones are converted to numeric values
#in the first alternative each vegetation zone has their own subclass
#in the second alternative vegetations are divided according to main vegetation zones (choicelist differs)
condlist = [df['MKV']=='1a', 
            df['MKV']=='1b',
            df['MKV']=='2a', 
            df['MKV']=='2b',
            df['MKV']=='3a',
            df['MKV']=='3b',
            df['MKV']=='3c',
            df['MKV']=='4a',
            df['MKV']=='4b',
            df['MKV']=='4c',
            df['MKV']=='4d']
choicelist = ['1','1','2','2','3','3','3','4ab','4ab','4cd','4cd']
df['MKV_MainClass'] = np.select(condlist, choicelist)

#choicelist = ['1','1','1','1','1','1','1','4ab','4ab','4cd','4cd'] #for north and south class
#choicelist = [0,1,2,3,4,5,6,7,8,9,10] #for each subclass separately
#df['MKV_class'] = np.select(condlist, choicelist)


#define classes that you want to predict, e.g. here overlap with area that is classified correctly by experts
#e.g. when overlap over 50%, define it as 1, where 1 = OGF and 0 = is not OGF
#Neg_columns = ['MKI_Perc', 'GFW_Perc', 'Ojitus_Perc','Ihmispaine16m_Mean']
#if certain negative thresholds get fulfilled, assign 0 even if "a certain OGF", where “certain” means high overlap
condition1 = df['AllReferencePolygons_pc']>50 #overlap with reference over x%
condition2 = df['Ihmispaine16m_Mean']<20 #human impact less than x
condition3 = df['MKI_Perc']<20 #forest use declaration less than x
condition4 = df['Ojitus_Perc']<20 #drainage percentage less than x
condition5 = df['GFW_Perc']<20 #global forest loss percentage less than x

"""
EXAMPLE this part not utilized in script
0 class might be too varied and there might be a need to curate it to lessen the amount of 50/50 classifications
no certain expected results with this polarization, but increase in correlation should be expected
ccondition1 = df['OGFReference_pc']==0
ccondition2 = df['Ihmispaine16m_Mean']>50
ccondition3 = df['MKI_Perc']>50
ccondition4 = df['Ojitus_Perc']>50
ccondition5 = df['GFW_Perc']>50
"""

#Combine the conditions
all_conditions_met = condition1 & condition2 & condition3 & condition4 & condition5 #to include multiple assesment factors
all_conditions_met = condition1 #to single out certain criterium

# Convert the boolean array to 0s and 1s using astype(int)
df['OGF'] = all_conditions_met.astype(int)


#Columns to use in the equation
Pos_columns = ['Kasvupaikka','Ika_MaxYhdistetty_Mean', 'KKorkeus8m_Mean', 'KokTilavuusUusi_Mean', 'MZonationValt2_Mean','All90PercEtela4KL_BTKUVuusiA_Peitto','MantyVol_Osuus','LehtipVol_Osuus','MantyVol_Mean','LehtipVol_Mean']
Neg_columns = ['MKI_Perc', 'GFW_Perc', 'Ojitus_Perc','Ihmispaine16m_Mean','Kiinteisto1_5haRakennusPerc','Rakennukset30mPerc']
Maybe_columns = ['KKL2m_Median', 'KKL2m_Mean', 'KKL2m_Std','Ika_MaxYhdistetty_Std']
Other_columns = ['MKV_class','Rakenne_Mean','Rakenne_Std','KKL2m_Median', 'KKL2m_Mean', 'KKL2m_Area', 'KKL2m_Std', 'MantyVol_Osuus', 'LehtipVol_Osuus','Ika_MaxYhdistetty_Std']
All_columns = Pos_columns + Neg_columns + Maybe_columns
print(f"number of variables used in the equation is: {len(All_columns)}")

def normalize_column(column):
    min_val = min(column)
    max_val = max(column)
    normalized_column = [(x - min_val) / (max_val - min_val) for x in column]
    return normalized_column
    

X = df[All_columns]
normalized_x = []
for column in X:
    #variable names
    name = column + '_norm'
    name2 = column + '_imputed'
    name3 = column + '_z'
    #print(name)
    normalized_x.append(name)

    #if nodatavalue 999, then assign as median
    df[column]=df[column].replace(999, df[column].quantile(0.5))

    #missing data filling. nulls are already filled with zeroes on line 42, so here no impact, comment it out for this to take effect
    imputer = SimpleImputer(strategy='median')  # or 'mean', median', 'most_frequent', 'constant'.
    df[name2] = imputer.fit_transform(df[[column]])

    #outlier detection with z-score and their removal
    df[name3] = stats.zscore(df[name2])
    outlier_threshold = 3
    outliers = (df[name3] > outlier_threshold) #|(df[name3] < -outlier_threshold) if lower outliers need to be processed
    np.where(outliers, max(column),column) #if outlier, replace with max value

    #normalization function
    df[name] = normalize_column(df[name3])
    #df[name] = normalize_column(df[column])

print(f"columns used and normalized: {normalized_x}")

# Access a specific layer as a GeoDataFrame using geopandas
files = r"YourFilePath"
layer_names = ['Kymenlaakso_Kuviot_Single', 'Keski_Pohj_Kuviot_Single']
# list unique MKV classes to iterate
MKVclasses = []
MKVmainclasses = []
for layername in layer_names:
    ddf = dask_gpd.read_file(files, layer=layername, npartitions=1)
    dfname = f"df_{layername}"
    globals()[dfname] = ddf.compute() #by using globals you can directly write a new variable

    #already remove rows with high human impact
    ccondition2 = globals()[dfname]['Ihmispaine16m_Mean']<80
    ccondition3 = globals()[dfname]['MKI_Perc']<80
    ccondition4 = globals()[dfname]['Ojitus_Perc']<80
    ccondition5 = globals()[dfname]['GFW_Perc']<80
    globals()[dfname] = globals()[dfname][ccondition2 & ccondition3 & ccondition4 & ccondition5]

    #create MKV mainclass for grouping
    condlist = [globals()[dfname]['MKV']=='1a', 
            globals()[dfname]['MKV']=='1b',
            globals()[dfname]['MKV']=='2a', 
            globals()[dfname]['MKV']=='2b',
            globals()[dfname]['MKV']=='3a',
            globals()[dfname]['MKV']=='3b',
            globals()[dfname]['MKV']=='3c',
            globals()[dfname]['MKV']=='4a',
            globals()[dfname]['MKV']=='4b',
            globals()[dfname]['MKV']=='4c',
            globals()[dfname]['MKV']=='4d']
    choicelist = ['1','1','2','2','3','3','3','4ab','4ab','4cd','4cd']
    globals()[dfname]['MKV_MainClass'] = np.select(condlist, choicelist)   

    MKVclass = globals()[dfname]['MKV'].unique().tolist()
    MKVmainclass = globals()[dfname]['MKV_MainClass'].unique().tolist()
    MKVclasses.append(MKVclass)
    MKVmainclasses.append(MKVmainclass)
    print(f"{layername} has classes {MKVclass}")
    
    #to remove nested list and combine identical values
    MKVclasses_flat = list(chain.from_iterable(MKVclasses))
    MKVclasses_flat = pd.unique(MKVclasses_flat)
    MKVmainclasses_flat = list(chain.from_iterable(MKVmainclasses))
    MKVmainclasses_flat = pd.unique(MKVmainclasses_flat)


#run something for each forest vegetation zone separately
grouped_training = df.groupby(by='MKV_MainClass')
groups = MKVmainclasses_flat


for group in groups:
    subset_training = grouped_training.get_group(group)
    print(f"trainingset {group} has a size of {len(subset_training)} with OGF percentage being {(subset_training['OGF']==1).sum()/len(subset_training)*100}")
    #define target variable Y or dependent variable as x is independent -> y = x1 + x2 -x3 equation
    Y = subset_training['OGF']
    #define independent variables
    X = subset_training[normalized_x]
    #split data into train and test sections
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    classifier = RandomForestClassifier(n_estimators=20,criterion='gini', random_state= 0, n_jobs=-1) #gini, entropy
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    y_pred_perc = classifier.predict_proba(X_test)[:,1] #[:,0]is probability of class 0, [:,1] of class 1
    print(classification_report(y_test,y_pred))
    # ROC curve
    fpr, tpr, thresholds  = metrics.roc_curve(y_test, y_pred_perc)
    plt.figure()
    plt.title(f"AUC score {metrics.roc_auc_score(y_test, y_pred_perc)}")
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(rf'YourFilePath\ROC_{group}.png')
    #plt.show()
    
    for layername in layer_names:
            dfname = f"df_{layername}"
            grouped2 = globals()[dfname].groupby(by='MKV_MainClass')
            try:
                subset = grouped2.get_group(group).copy()
                print(f"{layername} class {group} is of size {len(subset)}")
                normalized_x = []
                X = globals()[dfname][All_columns]
                for column in X:
                    #variable names
                    name = column + '_norm'
                    name2 = column + '_imputed'
                    #print(name)
                    normalized_x.append(name)

                    #missing data filling
                    imputer = SimpleImputer(strategy='median')  # or 'mean', median', 'most_frequent', 'constant'.
                    subset[name2] = imputer.fit_transform(subset[[column]])

                    #normalization function
                    try:
                        subset[name] = normalize_column(subset[name2])
                    except ZeroDivisionError:
                        subset[name] = 0  # Filling the result with 0 when division by zero occurs
                X = subset[normalized_x]
                subset['predicted_OGF'] = classifier.predict(X)
                subset['predicted_OGF_probablity'] = classifier.predict_proba(X)[:,1] #0 for class 0, 1 for 1
                condition = subset['predicted_OGF'] == 1 #if need to subset only OGFs
                condition = subset['predicted_OGF_probablity'] > 0.4 #if need to subset only polygons with odds over 40%
                print(f"{len(subset[condition])} OGF polygons in {layername}_{group}")
                subset_selected = subset[condition]
                print(f"probability of class estimate being correct is around {subset_selected['predicted_OGF_probablity'].describe()}")


                # Perform a spatial join of species data
                joined = gpd.sjoin(subset_selected, speciespoints)
                # Count the number of points in each polygon. use an unique id column to group
                print(f"groupingvariable (FID_Alue) is unique {subset_selected.FID_Alue.is_unique}")
                counts = joined.groupby('FID_Alue').size()
                # Add the counts back to the original DataFrame
                df_poly = subset_selected.merge(counts.rename('speciesobservations').reset_index(), how='left')
                #df_poly['speciesobservations'] = df['speciesobservations'].fillna(0)
                print(f"on average {df_poly.speciesobservations.mean()} observations per polygon")

                #this script produces additional link columns named 'Paikkatietoikkuna' and 'VanhatKartat'
                #these additional map services include historical aerial images and basemaps for visual
                #inspection of time dependent changes

                #read data that will get additional columns describink links to external services such as paikkatietoikkuna and vanhat kartat
                df = df_poly

                #PAIKKATIETOIKKUNA (https://kartta.paikkatietoikkuna.fi/)
                #produces a centroid coulumn used for zooming
                df['centroid']=df.centroid
                #link in parts
                part1=r'https://kartta.paikkatietoikkuna.fi/?zoomLevel=10&coord='
                centroidstring=df['centroid'].astype(str)
                part3='_'
                part2=centroidstring.str.extract('(\\d\\d\\d\\d\\d\\d)')
                part4=centroidstring.str.extract('( \\d\\d\\d\\d\\d\\d\\d)')
                #shortened version of a too long url
                part5=r'&mapLayers=801+100+,3400+100+ortokuva:indeksi,90+100+,99+100+,2622+61+,722+50+,511+50+&timeseries=1950&noSavedState=true&showMarker=true&showIntro=false'
                #alternative ways
                #shortened version of too long url,without map marker
                #part5=r'&mapLayers=801+100+default,3400+100+ortokuva:indeksi,90+100+default,99+100+default,2622+61+default,722+50+default,511+50+default&timeseries=1950&noSavedState=true&showIntro=false'
                #with other maps on top
                #part5=r'&mapLayers=801+100+default,3400+100+ortokuva:indeksi,90+100+default,99+100+default,2622+61+default,722+50+default,511+50+default&timeseries=1950&noSavedState=true&showMarker=true&showIntro=false'
                #without map marker
                #part5=r'&mapLayers=801+100+default,722+100+default,511+100+default,2622+100+default,90+100+default,99+100+default,3400+100+ortokuva:indeksi&timeseries=1950&noSavedState=true&showIntro=false'
                #with map marker
                #part5=r'&mapLayers=801+100+default,722+100+default,511+100+default,2622+100+default,90+100+default,99+100+default,3400+100+ortokuva:indeksi&timeseries=1950&noSavedState=true&showMarker=true&showIntro=false'
                #without extra layers
                #part5=r'&mapLayers=801+100+default,3400+100+ortokuva:indeksi&timeseries=1950&noSavedState=true&showIntro=false&lang=fi'
                df['Paikkatietoikkuna']=part1+part2+part3+part4+part5
                df['Paikkatietoikkuna']=df['Paikkatietoikkuna'].str.replace(" ","")



                # VANHAT KARTAT (https://vanhatkartat.fi/#12.3/65.00854/25.46662)
                #vanhatkartat.fi is in wgs84, so the data is reprojected to wgs84
                df=df.to_crs(4326)
                df['centroid']=df.centroid
                centroidstring_wgs84=df['centroid'].astype(str)
                #link in parts
                part1=r'https://vanhatkartat.fi/#13/'
                part2=centroidstring_wgs84.str.extract('( \\d\\d\\.\\d\\d\\d\\d\\d)')
                part3='/'
                part4=centroidstring_wgs84.str.extract('(\\d\\d\\.\\d\\d\\d\\d\\d)')
                df['VanhatKartat']=part1+part2+part3+part4
                df['VanhatKartat']=df['VanhatKartat'].str.replace(" ","")

                #two geometry columns creates problems
                df=df.drop(columns='centroid')
                #convert back to eureffin (most common projection in Finland)
                df=df.to_crs(3067)

                #save to file
                out_file = rf"YourFilePath\{layername}_{group}.parquet"
                df.to_parquet(out_file)
                print(f"saving to {out_file}")

            except:
                print(f"no polygons in subset {layername}_{group}")

end_time = time.time()
execution_time = end_time - start_time
execution_time_minutes = execution_time / 60
print("Total time used: {:.2f} minutes".format(execution_time_minutes))