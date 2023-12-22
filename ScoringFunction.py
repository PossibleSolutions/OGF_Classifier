import numpy as np
import dask.dataframe as dd

# inputs are lists, e.g. variable_maxes=[Agepercentile,Heightpercentile,Volumepercentile,Zonationpercentile]
#its pseudocode
 # iterate through every variable that is considered important in scoring
 # fill nulls with zeroes (or alternatively with other more sophisticated filling method according to data characteristics)
 # normalize the value to 0 to 1 scale and multiply by weigth for a score
 # sum the individual parameter scores to total score   
 # when the last iteration, create a new total score column and write those there

 # there are cases where scaling range e.g. quantile 0.05-0.95 results in rows being considered outliers, in these cases assign value to min or max so that normalized value cannot be negative or over 1
 # scaleme=value cannot be negative, e.g. if scaling range is quantile 0.05-0.95 and value 0.01, then assign 0.05 for value 0
 # nozerodivision=if mostly empty column with filled zeroes, scaling range min and max can be the same 0 resulting in zerodivision error, then use absolute max values 
 # parameter=if the value exceeds the max value, assign it to have max points of 1*weight
def Scorer (dataframe, variables, variable_maxes, variable_maxes_abs, variable_mins, weigths, Pos_columns):
    total=0
    for index, variable in enumerate(variables):
        #print(index)
        if index < len(Pos_columns): #this for the positive impact columns
            #fillnull=np.where(dataframe[variable].isnull(),0.75*variable_maxes[index],dataframe[variable])#for some custom fill
            fillnull=dataframe[variable].fillna(0)
            fillnull=fillnull.replace(999, (0.5*variable_maxes[index]))#if nodatavalue 999, then assign as 50%
            scaleme=np.where((fillnull - variable_mins[index])>0,(fillnull - variable_mins[index]),variable_mins[index])
            nozerodivision=np.where((variable_maxes[index] - variable_mins[index])>0,(variable_maxes[index] - variable_mins[index]),(variable_maxes_abs[index] - variable_mins[index]))
            normalized_column = scaleme / nozerodivision
            equation=normalized_column*weigths[index]
            parameter=np.where(equation<=weigths[index],equation,weigths[index])
            total+=parameter
            #print('Used as a positive impact:', variable)
        else: #this for the negative impact columns
            fillnull=dataframe[variable].fillna(0)
            scaleme=np.where((fillnull - variable_mins[index])>0,(fillnull - variable_mins[index]),variable_mins[index])
            nozerodivision=np.where((variable_maxes[index] - variable_mins[index])>0,(variable_maxes[index] - variable_mins[index]),(variable_maxes_abs[index] - variable_mins[index]))
            normalized_column = scaleme / nozerodivision
            equation=normalized_column*weigths[index]
            parameter=np.where(equation<=weigths[index],equation,weigths[index])
            parameter=np.where(normalized_column>0.85,5*parameter,parameter) #emphasize high human impact rows
            total-=parameter
            #print('Used as a negative impact:', variable)
        if index == len(variables) - 1:
            total = dd.from_array(total) # Convert NumPy array to Dask Series, dask does not support assigning NumPy arrays directly to DataFrame columns
            dataframe['Score']=total
            #print('Assessed factors:',variables)
            #print('their absolute max values:',variable_maxes_abs)
            #print('their max values:',variable_maxes)
            #print('their min values:',variable_mins)
            print('and weight multipliers:',weigths)
    
    return Scorer