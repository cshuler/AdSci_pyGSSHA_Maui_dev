import os 
import pandas as pd
import numpy as np
import subprocess
import sys
import shutil
from distutils.dir_util import copy_tree
import distutils
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import timeit



### Run a command line version of GSSHA model   ###
''' 
PrjName = the project name under which the WMS file is saved (be sure to save the WMS file INSIDE of the RUN directory)
RUN_dir = the RELATIVE or ABSOLUTE PATH to the directory where the project files are saved 
'''

def run_GSSHA(PrjName, RUN_dir):
    
    start_time = timeit.default_timer()     ###### Timer function start
    try: 
        # note GSSHA.exe has to be in the pwd of the kernal NOT the RUN directory
        subprocess.call('gssha.exe {}.prj'.format(PrjName),  cwd=RUN_dir) # shell=True)   
    except: print("FAILED execution!! check to see, GSSHA.exe has to be in the pwd of the notebook kernal NOT in the RUN directory, AND check to see if the absolute paths in the .prj file are correct, have to save from WMS into the pwd")
        
    elapsed = timeit.default_timer() - start_time        ###### Timer function End, now time in seconds lives in elapsed
    return elapsed
        
### Process outlet file from GSSHA into a pandas dataframe  ###
'''
StartDate = the start date and time of the run, in this format with hours and minutes "2018-08-23 00:00"
PrjName   =  the project name under which the WMS file is saved
PrjDir    = can be specified if its not in a RUN subdirectory of the pwd
'''
def process_otl_file(StartDate, PrjName, PrjDir = os.path.join('.', 'RUN')):

    # Create the start time object to enumerate the number of minutes in the outlet file 
    StartDateTime = datetime.strptime(StartDate, '%Y-%m-%d %H:%M')
    
    # read in the outlet file 
    OutletFile = os.path.join(PrjDir, PrjName+".otl") 
    OutHydro = pd.read_csv(OutletFile, names=["Minutes", "CFS"], delim_whitespace=True)
    
    # da magic: Turn stupid minutes into useful datetime objects 
    OutHydro["date"] = OutHydro["Minutes"].apply(lambda x: StartDateTime + timedelta(minutes=x))
    
    # Set the index to the date
    OutHydro.set_index("date", inplace=True)
    
    return OutHydro





    
### The rain gage file maker function for a SINGLE rain gauge ### 
"""
Note that this will modify the PrjName.gag file as well as the PrjName.prj file to update the run length to match the precip.gag length

PrjName   =  the project name under which the WMS file is saved
Input_Precip_df - A dataframe of timerseries precipitation data Note that this has to be in memory INSIDE of the current script
Precip_column_Name - The column in the above Input_Precip_df to use
StartDate - Desired start date of the run (note will work if no hours or minutes but best to add them)
EndDate  - Desired end date of the run 
Lat lon of the gauge site, important for GSSHA calculations  (Needs to be formatted as string) 
RainSeries_timestep_Mins - The timestep of the input rainfall data from Input_Precip_df
PrjDir    = can be specified if its not in a RUN subdirectory of the pwd
ImpPrecip_units - Units of the precip df, has to either be 'Inches' or 'mm
GageName - name of gauge, not so important
"""
    
def make_rain_gag_file(PrjName, Input_Precip_df, Precip_column_Name, StartDate, EndDate, Lat, Lon, 
                        RainSeries_timestep_Mins, ImpPrecip_units, PrjDir = os.path.join('.', 'RUN'), GageName="mooface"):

    # read in data 
    SliceFrame = Input_Precip_df[StartDate:EndDate]

    # Format date string for dumb file 
    SliceFrame_format = SliceFrame.copy()
    SliceFrame_format['da'] = SliceFrame_format.index.strftime('%Y %m %d %H %M')
    SliceFrame_format['datedumb'] = SliceFrame_format['da'].str[:]

    # Pull extranious columns 
    SliceFrame_format = SliceFrame_format[['datedumb', Precip_column_Name]]
    SliceFrame_format.rename(columns={Precip_column_Name: 'Rainfall'}, inplace=True)
    
    
    # Abort mission if there are any missing values in the data, GSSHA will not run with mising rain values 
    if SliceFrame_format.isnull().values.any():
        print("Mission aborted, there are NaN values in the rain data, please fix your data or choose different dates")
        
        return 0,0
    
    # Turn rain in to rain MM if its in inches to start with which is the default
    if ImpPrecip_units == "Inches": 
        SliceFrame_format['Rainfall'] = SliceFrame_format['Rainfall']*25.4 

        
    # Round off the number of significant figs
    sigfigs = 3
    SliceFrame_format['Rainfall'] = SliceFrame_format['Rainfall'].round(sigfigs).apply(lambda x: 
                                                                                                 '{0:g}'.format(float(x)))
    # Put the GAGES card on EVery single row 
    SliceFrame_format["trash"] = "GAGES"

    # reorder columns 
    SliceFrame_format = SliceFrame_format[["trash", "datedumb", 'Rainfall']]

    # Print it off to a txt file with no header, no index and space separator
    FileNamePlace = os.path.join(PrjDir, "{}.gag".format(PrjName))
    SliceFrame_format.to_csv(FileNamePlace, index=False, sep=' ', header = False) 

    # Remove dumb double quotes from the pandas to csv (could probably clean up and use numpy instead if ambitious...)
    with open(FileNamePlace,'r') as file:
        data = file.read()
        data = data.replace('"','')
    with open(FileNamePlace,'w') as file:    
        file.write(data)

    # Create the required header lines for WMS gag  files 
    AddLine = 'EVENT "Rain Gage" \nNRGAG 1 \nNRPDS {}\nCOORD {} {} "{}"'.format(len(SliceFrame_format),Lat, Lon, GageName)  

    with open(FileNamePlace, "r+") as f:
        old = f.read() # read everything in the file
        f.seek(0) # rewind
        f.write("{}\n".format(AddLine) + old) # write the new line before

    # Calculate the number of minutes to run the damn thing. 
    Run_Length_mins = RainSeries_timestep_Mins*len(SliceFrame_format)
    #print("Simulation time is {} minutes".format(Run_Length_mins))
    #print("number of time rows is {}".format(len(SliceFrame_format)))
    
    
    #  Now open and modify the .prj file to run at the new timestep
    card_name = "TOT_TIME"   # THIS is the card we change for the rainfall function here
    df = pd.read_csv(os.path.join(PrjDir, "{}.prj".format(PrjName)), names=["moo"] )  
    # note names=moo is to make a column that will then get chopped off by numpy savetxt
    singleCol = df.columns[0]     # All data was read into a single column with pandas
    idx_tottime = df.loc[df[singleCol].str.contains(card_name, case=False)].index[0]  # this identifies the index of the TOT_TIME card
    df.loc[idx_tottime] = "{}        {}".format(card_name, Run_Length_mins)
    
    # Save the df back to a prj file (the np formulation seems to write better than the pd to csv one)
    np.savetxt(os.path.join(PrjDir, "{}.prj".format(PrjName)), df.values, fmt="%s")
    
    
    return Run_Length_mins, SliceFrame_format         
                     

    
    
    
### General function to modify the .prj file based on its project cards ###
'''
PrjName   =  the project name under which the WMS file is saved
card_name = The .prj file value to modify 
Mod_Value = the value (as a string) to sub in
PrjDir    = can be specified if its not in a RUN subdirectory of the pwd
'''

def modify_prj_file(PrjName, card_name, Mod_Value, PrjDir = os.path.join('.', 'RUN')):
    df = pd.read_csv(os.path.join(PrjDir, "{}.prj".format(PrjName)), names=["moo"] )  # note names=moo is to make a column that will then get chopped off by numpy savetxt
    singleCol = df.columns[0]     # All data was read into a single column with pandas
    idx_tottime = df.loc[df[singleCol].str.contains(card_name, case=False)].index[0]  # this identifies the index of the TOT_TIME card
    df.loc[idx_tottime] = "{}        {}".format(card_name, Mod_Value)
    
    # Save the df back to a prj file (the np formulation seems to write better than the pd to csv one)
    np.savetxt(os.path.join(PrjDir, "{}.prj".format(PrjName)), df.values, fmt="%s")
    
    
    
    
### Calculate NSE of predictions and obs NOTE that predictions and obs are NOT interchangable #### 
def nse(predictions, obs):
    return (1-(np.sum((predictions-obs)**2)/np.sum((obs-np.mean(obs))**2)))


#### Open up the parameters file and replace certain keys with certain values   #### 
def cmt_prama_jama(Param, Val, PrjName, PrjDir = os.path.join(".", "RUN")):
    MapTableFile = os.path.join(PrjDir, "{}.cmt".format(PrjName))
    
    Val = str(Val)   # Convert non-string numbers into strings 
    
    with open(MapTableFile, 'r') as file :    # Read in the file 
        filedata = file.read()

    filedata = filedata.replace(Param, Val)  # Replace the target paramater(s)

    with open(MapTableFile, 'w') as file:   # Write the file out again
        file.write(filedata)
                     
       
    
###########  Equal spacing algorithm to generate parameter values as strings  #####
'''
minval = the start number or exponent (base 10) 
# maxval = the end number or exponent
# numbah = the number of params to generate in the output 
'''
def equal_spacing(minval, maxval, numbah):
    p_list = np.linspace(minval,maxval,numbah)
    p_list = p_list.round(3)
    Param_range = p_list.astype(str)
    return Param_range 

# Logarithmic spacing note that inputs are not the start & end value, its the EXPONENT so a good hydro cond dist = (-1, 2, 5)
def log_spacing(minval, maxval, numbah):
    p_list = np.logspace(minval,maxval,numbah)
    p_list = p_list.round(3)
    Param_range = p_list.astype(str)
    return Param_range                      
                     
                     
                     
            
##### refresh model from the pristine Dir 

# Nuke out the RUN directory to start fresh
def refresh_model(PrjName, PrjDir = os.path.join(".", "RUN"), PristineDir = os.path.join(".", "PRISTINE_MODEL_COPY")):           
    for f in os.listdir(PrjDir):  os.remove(os.path.join(PrjDir, f))   
    copy_tree(PristineDir, PrjDir)  # copy out the project filesfrom pristine to RUN
    
    # CRITICAL to remember!!! This is run every time the model is refreshed in order to scrub the training 0s in the parameters 
    cmt_prama_jama(".000000", " ", PrjName)  # scrub out the ".000000"  BS that lovely WMS sticks onto everything... 

                     
                     
                     
                     
####### Assign base values to the non-test cmt  parameters 
'''
Code_list_in_WMS  = this is a list or array of all the parameter codes that are in the clean saved WMS cmt file. 
                    Note that this list has to be a list of strings, and without the .000000 that WMS adds because 
                    we scrub those out on every iteration, because they are dumb
                    
Code_Key_df  = this is the dataframe that keys out the parameter codes with the base values,
               as the model evolves to have better tuned base values this will be the place to permamntly 
               store that info 
'''

def assign_cmt_base_vals(Code_Key_df, Code_list_in_WMS, PrjName):
    
    for code in Code_list_in_WMS:    
        # super complicated querry function, looks up the base value from the df based on the param code and converts to string for punching into .cmt file 
        value = str(Code_Key_df.query('Param_Code=={}'.format(code))['Base_Val'].item())  
        cmt_prama_jama(code, value, PrjName)
        
        
        
#### Process streamflow datasets from da streamflow dataframe  ### 


def Isolate_Stream_Data(Input_Stream_df, StreamFlow_column_Name, StartDate, EndDate, 
                        StreamFlowObs_resample_Timestep_mins = 60):
    SliceFrame = Input_Stream_df[StartDate:EndDate]
    
    # USGS data is amazingly terrible, it does not have consistent timesteps, fix this issue here
    mins = StreamFlowObs_resample_Timestep_mins
    SliceFrame = SliceFrame.resample('{}T'.format(mins)).mean().interpolate()
    
    # Rename the column name to a standard one so as not to have to input it in the future functions 
    SliceFrame = SliceFrame[[StreamFlow_column_Name]]
    SliceFrame.rename(columns={StreamFlow_column_Name: 'Streamflow'}, inplace=True)
    
    return SliceFrame      
        

        
######## Scan timeseries data and find the X number of largest events   #####
'''
Input_Stream_df  = Timeseries dataframe, Note it has to have datetime as the index already 
StreamFlow_column_Name  = The column name with flow or rainfall values 
num_top_events = 10     = The number of events to clip the results to 
resample_timestep='1D'  = Timestep can be 1 day ('1D') or any hours ('6H') (in ""s btw )
'''

def Biguns_large_event_finder(Input_Stream_df, StreamFlow_column_Name, num_top_events = 10, resample_timestep='1D'):   
    Flow_1d = Input_Stream_df.resample(resample_timestep).mean()
    biguns = Flow_1d.sort_values(StreamFlow_column_Name, ascending=False)
    biguns.reset_index(inplace=True)
    biguns = biguns[0:num_top_events]
    return biguns    
        
        

        
#### Plot stuff  

'''
OutHydro   =    The output data from the model run, processed by the process_otl_file funciton 
SlicedStreamflow_df   = The processed stream flowdataframe from the Isolate_Stream_Data function 
StreamFlow_column_Name = The column name of the streamflow data 
Rain_Data_Frame =    T the rainfall data for the given run as produced by the make_rain_gag_file fucntion 
RunID  =  The ID composed of the parameter name and its value for identifying figures later composed in the iteration cell 
elapsed =  The amount of time the model took to run from the rub model function 
Save   = If you want to save the figure 
FigFolder = the path to where you want the figure saved 

'''

def Plot_and_Compare(OutHydro, SlicedStreamflow_df, Rain_Data_Frame, RunID, elapsed, 
                     Save=True, FigFolder=os.path.join(".", "Figures")): 
    
    # Calculating NSE
    targets = SlicedStreamflow_df.resample('60T').mean().interpolate()
    NSE_Frame = OutHydro.join(targets, how='inner')
    NSE_stat = nse(NSE_Frame['CFS'].values, NSE_Frame["Streamflow"].values)
    print("The NSE is {}".format(NSE_stat))
    
        # Plotting 
    # For some reason need to cast the rainfall as a number not object?
    Rain_Data_Frame['Rainfall'] = Rain_Data_Frame['Rainfall'].apply(lambda x: float(x)) 
    fig, ax = plt.subplots(figsize=(8, 3))
    ax2 = ax.twinx()
    lns0 = ax2.plot(Rain_Data_Frame['Rainfall'], '.', c='b', alpha=0.4, label="Rainfall ")   # Plot rainfall 
    lns1 = ax.plot(SlicedStreamflow_df['Streamflow'], '-', c='g', alpha=0.5, label="Observed Waihehe flow CFS") # Plot Observed
    lns2 = ax.plot(OutHydro['CFS'], '-', c='k', alpha=0.5, label="Modeled_CFS")   # plot modeled 
    ax.set_ylabel("Flow CFS")
    ax2.set_ylabel("Rainfall mm", color='b')
    # Wierd stuff for a twinned  axis legend
    lns = lns0+lns1+lns2; labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    
    plt.title("{} - NSE={}, time={}_sec".format(RunID,
                                             round(NSE_stat, 2), round(elapsed, 0)))
    # save fig
    if Save:
        Startrun = str(Rain_Data_Frame.index[0])
        Startrun = Startrun[:10]
        Endrun   = str(Rain_Data_Frame.index[-1])
        Endrun   = Endrun[:10] 
        plt.savefig(os.path.join(FigFolder, "{}-{}-to-{}.png".format(RunID, Startrun, Endrun)))



    
#### Check and scrub out periods where the rainfall and stream data overlap, remove those areas with issues 
"""
df_of_event_dates = A dataframe of single event dates with dates in a column called datetime
Input_Precip_df = The dataframe of rainfall values, indexed by date 
Rain_column_Name = column name with rain values 
StreamFlow_column_Name  = column name of stream values 
"""

def check_rain_stream_data(df_of_event_dates, Input_Precip_df, Input_Stream_df, Rain_column_Name, StreamFlow_column_Name):

    starts   = []
    ends     = []
    prob_RF  = []
    prob_SF  = []
    
    for row in df_of_event_dates.iterrows():
        # Set new start and end dates for each run 
        bigdate = row[1]['datetime']
        StartDate = (bigdate-timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
        EndDate = (bigdate+timedelta(days=2)).strftime('%Y-%m-%d %H:%M')

        # read in data 
        SliceFrame = Input_Precip_df[StartDate:EndDate]
        
        # Record the start date of where there is a problematic record
        if SliceFrame[Rain_column_Name].isnull().values.any() | SliceFrame.empty:
            prob_RF.append(StartDate)

        # As long as the rainfall data has a full record 
        if not SliceFrame[Rain_column_Name].isnull().values.any() | SliceFrame.empty: 

            # use unique start date to create streamflow obs file 
            SlicedStreamflow_df = Isolate_Stream_Data(Input_Stream_df, StreamFlow_column_Name, StartDate, EndDate)
            
            # Record the start date of where there is a problematic record
            if SlicedStreamflow_df['Streamflow'].isnull().values.any() | SlicedStreamflow_df.empty:
                prob_SF.append(StartDate)

            if not SlicedStreamflow_df['Streamflow'].isnull().values.any() | SlicedStreamflow_df.empty: 
                starts.append(StartDate)
                ends.append(EndDate)

    return starts, ends, prob_RF, prob_SF

        
        
        
        
        
        
        
        
        
        
        
        
        
                     
                     
                     
                     
                     