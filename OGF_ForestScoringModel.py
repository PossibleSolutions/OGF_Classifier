#General settings, imports and filepaths
import geopandas as gpd
import pandas as pd
import numpy as np
import dask_geopandas as dask_gpd
import dask.dataframe as dd
import time

#time the script running time
start_time = time.time()

#file that has species points to include how many observations fall into certain polygon
filename_species = r"PathToYourFile"
ddf_species = dask_gpd.read_file(filename_species, npartitions=4)
speciespoints = ddf_species.compute()

#Columns to use in the equation and their weights
Pos_columns = ['Ika_MaxYhdistetty_Mean', 'KKorkeus8m_Mean', 'KokTilavuusUusi_Mean', 'MZonationValt2_Mean','Rakenne_Mean']
Bonus_columns = ['speciesobservations','Pohjantikka_Mean','Kanahaukka_Mean','Valkoselkatikka_Mean','All90PercEtela4KL_BTKUVuusiA_Peitto']
Pos_columns = Pos_columns + Bonus_columns #comment this out if no need for "bonus" variables
Neg_columns = ['MKI_Perc', 'GFW_Perc', 'Ojitus_Perc','Ihmispaine16m_Mean']
All_columns = Pos_columns + Neg_columns
weigths=[50,5,10,30,5,5,5,5,3,5,5,10,10,30]
#Test different modelling inputs and weights for possible best outcome
Cor_columns = ['Ika_MaxYhdistetty_Mean','KKorkeus8m_Mean','MZonationValt2_Mean','MKI_Perc','GFW_Perc','Ojitus_Perc','Ihmispaine16m_Mean']
All_columns= Cor_columns
weigths = [50,15,35,10,15,15,20]

#import scoring function
from ScoringFunction import Scorer

#Access a specific layer as a GeoDataFrame using geopandas
files = r"PathToYourFile"
layer_names = ['Keski_Pohj_Kuviot_Single']

#Access reference data for result validation
filename = r"PathToYourFile"
overlay_dask = dask_gpd.read_file(filename, npartitions=2)
overlay = overlay_dask[['geometry']].compute()
print(f"Total number of national reference polygons is {len(overlay)}") 

for layername in layer_names:
    ddf = dask_gpd.read_file(files, layer=layername, npartitions=2)
    dfname = f"df_{layername}"
    globals()[dfname] = ddf.compute() #by using globals you can directly write a new variable
    print(f"layer {layername} has {len(globals()[dfname])} rows")
    joined = gpd.sjoin(overlay, globals()[dfname], how="inner", predicate="intersects")
    max_intersects = len(joined.index.unique())
    print(f"The number of unique reference polygons that overlap with area polygons is {max_intersects}")

    # Perform a spatial join of species data
    joined = gpd.sjoin(globals()[dfname], speciespoints)
    # Count the number of points in each polygon. use an unique id column to group
    print(f"groupingvariable (FID_Alue) is unique {globals()[dfname].FID_Alue.is_unique}")
    counts = joined.groupby('FID_Alue').size()
    # Add the counts back to the original DataFrame
    globals()[dfname] = globals()[dfname].merge(counts.rename('speciesobservations').reset_index(), how='left')
    globals()[dfname]['speciesobservations'] = globals()[dfname]['speciesobservations'].fillna(0)
    print(f"on average {globals()[dfname].speciesobservations.mean()} species observations per polygon")


    #Column minmax value definition for normalization purposes. Top and bottom x% excluded to focus scaling (remove outliers) 
    #It is later defined if value/x percentile>1, then 1.
    All_variable_max=[]
    All_variable_max_absolute=[]
    All_variable_min=[]
    for column in All_columns:
        valuemax=globals()[dfname][column].quantile(0.95)
        valuemaxabs=globals()[dfname][column].max()
        valuemin=globals()[dfname][column].quantile(0.05)
        #print(column,value)
        All_variable_max.append(valuemax)
        All_variable_max_absolute.append(valuemaxabs)
        All_variable_min.append(valuemin)


    #calculate score of potentiality with Scorer (dataframe, variables, variable_maxes, variable_maxes_abs, variable_mins, weigths, Pos_columns)
    Scorer(globals()[dfname], All_columns, All_variable_max, All_variable_max_absolute, All_variable_min, weigths, Pos_columns)

    #score spread/potentiality stats
    print(globals()[dfname].Score.describe())

    #Save top x% (here 25%) features or use aby other arbitrary threshold. we could e.g. assume that over 25% cant represent the best OGFs
    threshold_score = 65
    MostPromising = globals()[dfname][globals()[dfname]['Score'] > threshold_score]
    outfile = rf"PathToYourFile\{layername}_selected.parquet"
    MostPromising.to_parquet(outfile)
    #MostPromising = globals()[dfname][globals()[dfname]['Score'] > globals()[dfname].Score.quantile(0.75)]
    #MostPromising.to_file(rf"PathToYourFile\{layername}_bestFourth.gpkg", driver="GPKG")

    #check overlap with reference polygons to asses impact of weights
    joined = gpd.sjoin(overlay, MostPromising, how="inner", predicate="intersects")
    unique_polygons = len(joined.index.unique())

    print(f"The number of unique reference polygons that overlap with top polygons is {unique_polygons}. Threshold score was {threshold_score}") 
    print(f"This accounts for {unique_polygons/max_intersects*100} % of all the reference material") 
    print(f"Selection ratio (intersects/selected polygons) is {unique_polygons/len(MostPromising)}")
    print(f"Minimum score in the top was {globals()[dfname].Score.quantile(0.75)}")
    print(f"Saved to {outfile}")

    #COMMENT OUT FROM HERE TO...
    #create stats for each group overlapping to define OGF in context of spesific data
    #expect that stats should be high for pos and low for neg and relatively low std in the context of homogenous OGF group
    joined = gpd.sjoin(globals()[dfname],overlay, how="inner", predicate="intersects")
    globals()[dfname]['overlap'] = globals()[dfname].index.isin(joined.index).astype(int)

    #intersecting with reference
    condition = globals()[dfname]['overlap'] == 1
    intersecting = globals()[dfname][condition]
    print(len(intersecting))
    print(intersecting[All_columns].describe())

    #not intersecting with reference
    condition = globals()[dfname]['overlap'] == 0
    intersecting = globals()[dfname][condition]
    print(len(intersecting))
    print(intersecting[All_columns].describe()) 
    #...HERE IF UNNECESSARY ADDITIONAL INFO COSTING TOO MUCH PROCESSING POWER

    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_minutes = execution_time / 60
    print(layername,"Time used: {:.2f} minutes".format(execution_time_minutes))


end_time = time.time()
execution_time = end_time - start_time
execution_time_minutes = execution_time / 60
print("Total time used: {:.2f} minutes".format(execution_time_minutes))