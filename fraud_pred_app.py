import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import load
import streamlit as st


def preparing_data(data_ben, data_inp, data_out):
    '''this function prepares complete dataset by merging three dataset- 1. Beneficiary data, 2.Inpatient data, 3. Outpatient data
    and also merges with labeled data available in train csv file'''
    
    #Replacing 2 with 0 for chronic conditions ,that means chroniv condition No is 0 and yes is 1
    data_ben = data_ben.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

    data_ben = data_ben.replace({'RenalDiseaseIndicator': 'Y'}, 1)
    data_ben['RenalDiseaseIndicator'] = data_ben['RenalDiseaseIndicator'].astype(int)
    
    # Lets Create Age column to the dataset
    data_ben['DOB'] = pd.to_datetime(data_ben['DOB'] , format = '%Y-%m-%d')
    data_ben['DOD'] = pd.to_datetime(data_ben['DOD'],format = '%Y-%m-%d',errors='ignore')
    data_ben['Age'] = round(((data_ben['DOD'] - data_ben['DOB']).dt.days)/365)

    # As we see that last DOD value is 2009-12-01 ,which means Beneficiary Details data is of year 2009.
    # so we will calculate age of other benficiaries for year 2009.
    data_ben.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - data_ben['DOB']).dt.days)/365), inplace=True)
    
    #Lets create a new variable 'WhetherDead' with flag 1 means Dead and 0 means not Dead
    data_ben.loc[data_ben.DOD.isna(),'WhetherDead']=0
    data_ben.loc[data_ben.DOD.notna(),'WhetherDead']=1
    
    #As patient can be admitted for atleast 1 day, so we will add 1 to the difference of Discharge Date and Admission Date 
    data_inp['AdmissionDt'] = pd.to_datetime(data_inp['AdmissionDt'] , format = '%Y-%m-%d')
    data_inp['DischargeDt'] = pd.to_datetime(data_inp['DischargeDt'],format = '%Y-%m-%d')
    data_inp['AdmitForDays'] = ((data_inp['DischargeDt'] - data_inp['AdmissionDt']).dt.days)+1
    
    #Lets make union of Inpatienta and outpatient data .
    #We will use all keys in outpatient data as we want to make union and dont want duplicate columns from both tables.
    merged_data = pd.concat([data_inp, data_out])
    
    #Lets merge All patient data with beneficiary details data based on 'BeneID' as joining key for inner join
    merged_data = pd.merge(merged_data, data_ben, left_on='BeneID', right_on='BeneID', how='inner')
    
    return merged_data

@st.cache
def get_train_data():
    '''Use this function to load train data and merge with test data so that we can generate more accurate feature '''
    train_data_ben = pd.read_csv('archive/Train_Beneficiarydata-1542865627584.csv')
    train_data_inp = pd.read_csv('archive/Train_Inpatientdata-1542865627584.csv')
    train_data_out = pd.read_csv('archive/Train_Outpatientdata-1542865627584.csv')
    return preparing_data(train_data_ben, train_data_inp, train_data_out)

def feature_engg(test_data):
    '''this function will generate data point after feature engineering on raw data passed'''
    #storing test data columns for merging by these columns
    col_merge=test_data.columns

    ## Lets add both test and train datasets for generting accurate feature engineered data
    #train_data_all = get_train_data()
    train_test_merged = pd.concat([test_data, get_train_data()[col_merge]])
    #remove duplicate entriesu
    train_test_merged = train_test_merged.drop_duplicates(subset='ClaimID')
    
    #average feature grouped by provider
    train_test_merged["PerProviderAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('Provider')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerProviderAvg_DeductibleAmtPaid"]=train_test_merged.groupby('Provider')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerProviderAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('Provider')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerProviderAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('Provider')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerProviderAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('Provider')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerProviderAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('Provider')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerProviderAvg_Age"]=train_test_merged.groupby('Provider')['Age'].transform('mean')
    train_test_merged["PerProviderAvg_NoOfMonths_PartACov"]=train_test_merged.groupby('Provider')['NoOfMonths_PartACov'].transform('mean')
    train_test_merged["PerProviderAvg_NoOfMonths_PartBCov"]=train_test_merged.groupby('Provider')['NoOfMonths_PartBCov'].transform('mean')
    train_test_merged["PerProviderAvg_AdmitForDays"]=train_test_merged.groupby('Provider')['AdmitForDays'].transform('mean')
    #defragmenting dataframe, since after each section of preprocessing dataframe is getting larger this increases time and space complexity
    #so to reduce this we are merging train data and test data after generating some subsection of preprocessing
    #and then we are removing generated columns in train data to reduce space complexity
    # we are doing so because we only require test so after generating features for test data are deleting from train data 
    #this below line generates list of columns that only need to be merged and also in the same order as it was originaly generated in train data
    #maintaining sequence is mandetaory as Std Scaler preprocess in the same sequence as it was trained
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data[['ClaimID']].merge(train_test_merged, on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by Ben ID
    train_test_merged["PerBeneIDAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('BeneID')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerBeneIDAvg_DeductibleAmtPaid"]=train_test_merged.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerBeneIDAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerBeneIDAvg_AdmitForDays"]=train_test_merged.groupby('BeneID')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by attending physician
    train_test_merged["PerAttendingPhysicianAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('AttendingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_DeductibleAmtPaid"]=train_test_merged.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_AdmitForDays"]=train_test_merged.groupby('AttendingPhysician')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    to_be_ret = temp_cols_list
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by operating physician
    train_test_merged["PerOperatingPhysicianAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('OperatingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_DeductibleAmtPaid"]=train_test_merged.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_AdmitForDays"]=train_test_merged.groupby('OperatingPhysician')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by dx code group
    train_test_merged["PerDiagnosisGroupCodeAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('DiagnosisGroupCode')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_DeductibleAmtPaid"]=train_test_merged.groupby('DiagnosisGroupCode')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_AdmitForDays"]=train_test_merged.groupby('DiagnosisGroupCode')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by admit dx code
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_AdmitForDays"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim procedure code 1
    train_test_merged["PerClmProcedureCode_1Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmProcedureCode_1')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_AdmitForDays"]=train_test_merged.groupby('ClmProcedureCode_1')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim procedure code 2
    train_test_merged["PerClmProcedureCode_2Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmProcedureCode_2')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmProcedureCode_2')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_AdmitForDays"]=train_test_merged.groupby('ClmProcedureCode_2')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim dx code 1
    train_test_merged["PerClmDiagnosisCode_1Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_1')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_1')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim dx code 2
    train_test_merged["PerClmDiagnosisCode_2Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_2')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_2')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim dx code 3
    train_test_merged["PerClmDiagnosisCode_3Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_3')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_3')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature grouped by Provider+BeneID, Provider+Attending Physician, Provider+ClmAdmitDiagnosisCode, Provider+ClmProcedureCode_1, Provider+ClmDiagnosisCode_1, Provider+State
    train_test_merged["ClmCount_Provider"]=train_test_merged.groupby(['Provider'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID"]=train_test_merged.groupby(['Provider','BeneID'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_AttendingPhysician"]=train_test_merged.groupby(['Provider','AttendingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_OtherPhysician"]=train_test_merged.groupby(['Provider','OtherPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_OperatingPhysician"]=train_test_merged.groupby(['Provider','OperatingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmAdmitDiagnosisCode"]=train_test_merged.groupby(['Provider','ClmAdmitDiagnosisCode'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','ClmProcedureCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_2"]=train_test_merged.groupby(['Provider','ClmProcedureCode_2'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_3"]=train_test_merged.groupby(['Provider','ClmProcedureCode_3'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_4"]=train_test_merged.groupby(['Provider','ClmProcedureCode_4'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_5"]=train_test_merged.groupby(['Provider','ClmProcedureCode_5'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_1"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_2"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_2'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_3"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_3'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_4"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_4'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_5"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_5'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_6"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_6'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_7"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_7'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_8"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_8'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_9"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_9'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_DiagnosisGroupCode"]=train_test_merged.groupby(['Provider','DiagnosisGroupCode'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_AttendingPhysician"]=train_test_merged.groupby(['Provider','BeneID','AttendingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_OtherPhysician"]=train_test_merged.groupby(['Provider','BeneID','OtherPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','BeneID','AttendingPhysician','ClmProcedureCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_1"]=train_test_merged.groupby(['Provider','BeneID','AttendingPhysician','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_OperatingPhysician"]=train_test_merged.groupby(['Provider','BeneID','OperatingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','BeneID','ClmProcedureCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_ClmDiagnosisCode_1"]=train_test_merged.groupby(['Provider','BeneID','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','BeneID','ClmDiagnosisCode_1','ClmProcedureCode_1'])['ClaimID'].transform('count')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #here creating dx code grp for ClmDiagnosisCode_1
    train_test_merged['ClmDiagnosisCode_1_Grp'] = train_test_merged['ClmDiagnosisCode_1'].astype(str).str[0:2]
    #Average features group by dx code group as per proposed idea in abstract - for ClmDiagnosisCode_1
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #here creating dx code grp for ClmDiagnosisCode_1
    train_test_merged['ClmDiagnosisCode_2_Grp'] = train_test_merged['ClmDiagnosisCode_2'].astype(str).str[0:2]
    #Average features group by dx code group as per proposed idea in abstract - for ClmDiagnosisCode_2
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #here creating dx code grp for ClmDiagnosisCode_3
    train_test_merged['ClmDiagnosisCode_3_Grp'] = train_test_merged['ClmDiagnosisCode_3'].astype(str).str[0:2]
    #Average features group by dx code group as per proposed idea in abstract - for ClmDiagnosisCode_3
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    # for calculating tf_idf on claim dx codes
    dx_col_list = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4']
    temp_data = tf_idf_on_dx_cpt(train_test_merged[['ClaimID', 'Provider']+dx_col_list], dx_col_list)
    #defragmenting df
    test_data_all = test_data_all.merge(temp_data, on=['ClaimID', 'Provider'])
    
    
    # for calculating tf_idf on claim cpt codes
    cpt_col_list = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3']
    temp_data = tf_idf_on_dx_cpt(train_test_merged[['ClaimID', 'Provider']+cpt_col_list], cpt_col_list)
    #defragmenting df
    test_data_all = test_data_all.merge(temp_data, on=['ClaimID', 'Provider'])
    del temp_data
    

    ## Lets Convert types of gender and race to categorical.
    train_test_merged.Gender=train_test_merged.Gender.astype('category')
    train_test_merged.Race=train_test_merged.Race.astype('category')

    # Lets create dummies for categorrical columns.
    train_test_merged=pd.get_dummies(train_test_merged,columns=['Gender','Race'],drop_first=True)
    test_data = test_data.loc[:, ~test_data.columns.isin(['Gender','Race'])]
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    del train_test_merged
    
    ##### Lets impute numeric columns with 0
    cols1 = test_data_all.select_dtypes([np.number]).columns
    test_data_all[cols1]=test_data_all[cols1].fillna(value=0)
    
    # Lets remove unnecessary columns ,as we grouped based on these columns and derived maximum infromation from them.
    remove_these_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
           'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
           'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
           'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
           'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
           'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
           'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
           'ClmAdmitDiagnosisCode', 'AdmissionDt',
           'DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD',
            'State', 'County', 'ClmDiagnosisCode_1_Grp', 'ClmDiagnosisCode_2_Grp', 'ClmDiagnosisCode_3_Grp', 'Gender','Race']

    test_data_all = test_data_all.drop(axis=1, columns=remove_these_columns)

    ## Lets apply StandardScaler and transform values to its z form,where 99.7% values range between -3 to 3.
    sc = load('std_scaler.bin')   # MinMaxScaler
    X_test=sc.transform(test_data_all.iloc[:,1:])   #Apply Standard Scaler to unseen data
    return X_test

def tf_idf_on_dx_cpt(dataframe, dx_or_cpt_col_list):
    '''this function calculates tf, idf and tf_idf features on dx codes or cpt codes as per proposed idea of abstract document'''
    N = dataframe.groupby('Provider')['Provider'].count().shape[0] #no of unique provider = no of document corpus
    
    for each_col in dx_or_cpt_col_list:
        term_freq = dataframe.groupby(['Provider', each_col])[['ClaimID']].count().reset_index()
        term_freq.rename(columns={'ClaimID': each_col+'_term'}, inplace=True)
        dataframe = dataframe.merge(term_freq, on=['Provider', each_col], how='outer')
        no_of_dx_in_each_prov = dataframe.groupby('Provider')[each_col].count().reset_index()
        no_of_dx_in_each_prov.rename(columns={each_col:each_col+'_doc'}, inplace=True)
        dataframe = dataframe.merge(no_of_dx_in_each_prov, on=['Provider'], how='outer')
        dataframe[each_col+'TF'] = dataframe[each_col+'_term']/dataframe[each_col+'_doc']

        no_of_doc_containing_dx = dataframe.groupby(each_col)[['Provider']].count().reset_index()
        no_of_doc_containing_dx.rename(columns={'Provider':each_col+'_IDF'}, inplace=True)
        no_of_doc_containing_dx[each_col+'_IDF'] = np.log2(N/no_of_doc_containing_dx[each_col+'_IDF'])
        dataframe = dataframe.merge(no_of_doc_containing_dx, on=each_col, how='outer')
        dataframe[each_col+'TF-IDF'] = dataframe[each_col+'TF']*dataframe[each_col+'_IDF']
        dataframe.drop([each_col, each_col+'_term', each_col+'_doc'], axis=1, inplace=True)

    return dataframe

def fraud_prov_predict(raw_data):
    '''this function takes raw data as input, preprocess and featurize it and returned the predicted value'''
    start=time.time()
    featured_data = feature_engg(raw_data)
    end=time.time()
    st.write('time taken in feature engg ', end-start)
    start=time.time()
    xgb_clf = XGBClassifier(booster='gbtree')
    xgb_clf.load_model('XGB_Model.json')
    y_pred = xgb_clf.predict(featured_data)
    end=time.time()
    st.write('time taken in prediction ', end-start)
    return y_pred

@st.cache
def get_data():
    test_data_ben = pd.read_csv('archive/Test_Beneficiarydata-1542969243754.csv')
    test_data_inp = pd.read_csv('archive/Test_Inpatientdata-1542969243754.csv')
    test_data_out = pd.read_csv('archive/Test_Outpatientdata-1542969243754.csv')
    test_ddata_merged = preparing_data(test_data_ben, test_data_inp, test_data_out)
    return (test_data_ben, test_data_inp, test_data_out, test_ddata_merged)

st.title('Medicare Fraud Provider Prediction')  
df = get_data()

with st.sidebar:
    side_option = st.selectbox('Menu', ['Data Sample', 'Prediction', 'View Source Code'])

if side_option=='Data Sample':
    category = st.multiselect(label='Select Type of Sample data', options=['Beneficiary', 'Inpatient', 'Outpatient', 'Merged Data'])
    
    for cat in category:
        if cat == 'Beneficiary':
            st.write("Beneficiary data sample", df[0].head(100))
        elif cat == 'Inpatient':
            st.write("Inpatient data sample", df[1].head(100))
        elif cat == 'Outpatient':
            st.write("Outpatient data sample", df[2].head(100))
        elif cat == 'Merged Data':
            st.write("Merged data sample", df[3].head(100))
elif side_option=='Prediction':
    with st.sidebar:
        how_pred = st.selectbox(label='How you want to select sample for prediction', options=['Number', 'Range'])
    check_empty_dataset = 0
    if how_pred == 'Number':
        with st.sidebar:
            nth_val = st.number_input(label='Enter nth value', min_value=-df[3].shape[0], max_value=df[3].shape[0], value=0, key='for number option')
            nth_val_options = st.radio(label='What to treat the value', options=('Top '+str(nth_val)+'s', 'Bottom '+str(nth_val)+'s', str(nth_val)+' Random sample', str(nth_val)+'th Index', str(nth_val)+'th row'))
            if nth_val_options == 'Top '+str(nth_val)+'s':
                sample_test_data = df[3].head(nth_val)
            elif nth_val_options == 'Bottom '+str(nth_val)+'s':
                sample_test_data = df[3].tail(nth_val)
            elif nth_val_options == str(nth_val)+' Random sample':
                sample_test_data = df[3].sample(nth_val)
            elif nth_val_options == str(nth_val)+'th Index':
                sample_test_data = df[3].iloc[[nth_val]]
            else:
                sample_test_data = df[3].loc[[nth_val]]
    else:
        with st.sidebar:
            st.write('Enter range to select sample data for prediction')
            lower_lim = st.number_input(label='Enter lower limit', min_value=-df[3].shape[0], max_value=df[3].shape[0], value=0, key='for range lower limit')
            upper_lim = st.number_input(label='Enter upper limit', min_value=-df[3].shape[0], max_value=df[3].shape[0], value=10, key='for range upper limit')
            if st.checkbox(label='Index'):
                sample_test_data = df[3].iloc[lower_lim:upper_lim]
            else:
                sample_test_data = df[3].loc[lower_lim:upper_lim]
    check_empty_dataset = sample_test_data.shape[0]
    st.write('Selected sample data', sample_test_data)
    if check_empty_dataset==0:
        st.write('Please enter valid range/number of sample')
        button_disable = True
    else:
        button_disable = False
    if st.button('Predict', disabled=button_disable):
        start = time.time()
        with st.spinner("Please wait while processing, it will take less than a minute..."):
            pred_y = fraud_prov_predict(sample_test_data)
        st.success("Done..!")
        end = time.time()
        st.write('Time taken by prediction function to preprocess and predict ', end-start)
        sample_test_data['PridictedFraud'] = pred_y
        st.write('Predicted sample data', sample_test_data)
else:
    code="""
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import load
import streamlit as st


def preparing_data(data_ben, data_inp, data_out):
    '''this function prepares complete dataset by merging three dataset- 1. Beneficiary data, 2.Inpatient data, 3. Outpatient data
    and also merges with labeled data available in train csv file'''
    
    #Replacing 2 with 0 for chronic conditions ,that means chroniv condition No is 0 and yes is 1
    data_ben = data_ben.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

    data_ben = data_ben.replace({'RenalDiseaseIndicator': 'Y'}, 1)
    data_ben['RenalDiseaseIndicator'] = data_ben['RenalDiseaseIndicator'].astype(int)
    
    # Lets Create Age column to the dataset
    data_ben['DOB'] = pd.to_datetime(data_ben['DOB'] , format = '%Y-%m-%d')
    data_ben['DOD'] = pd.to_datetime(data_ben['DOD'],format = '%Y-%m-%d',errors='ignore')
    data_ben['Age'] = round(((data_ben['DOD'] - data_ben['DOB']).dt.days)/365)

    # As we see that last DOD value is 2009-12-01 ,which means Beneficiary Details data is of year 2009.
    # so we will calculate age of other benficiaries for year 2009.
    data_ben.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - data_ben['DOB']).dt.days)/365), inplace=True)
    
    #Lets create a new variable 'WhetherDead' with flag 1 means Dead and 0 means not Dead
    data_ben.loc[data_ben.DOD.isna(),'WhetherDead']=0
    data_ben.loc[data_ben.DOD.notna(),'WhetherDead']=1
    
    #As patient can be admitted for atleast 1 day, so we will add 1 to the difference of Discharge Date and Admission Date 
    data_inp['AdmissionDt'] = pd.to_datetime(data_inp['AdmissionDt'] , format = '%Y-%m-%d')
    data_inp['DischargeDt'] = pd.to_datetime(data_inp['DischargeDt'],format = '%Y-%m-%d')
    data_inp['AdmitForDays'] = ((data_inp['DischargeDt'] - data_inp['AdmissionDt']).dt.days)+1
    
    #Lets make union of Inpatienta and outpatient data .
    #We will use all keys in outpatient data as we want to make union and dont want duplicate columns from both tables.
    merged_data = pd.concat([data_inp, data_out])
    
    #Lets merge All patient data with beneficiary details data based on 'BeneID' as joining key for inner join
    merged_data = pd.merge(merged_data, data_ben, left_on='BeneID', right_on='BeneID', how='inner')
    
    return merged_data

@st.cache
def get_train_data():
    '''Use this function to load train data and merge with test data so that we can generate more accurate feature '''
    train_data_ben = pd.read_csv('archive/Train_Beneficiarydata-1542865627584.csv')
    train_data_inp = pd.read_csv('archive/Train_Inpatientdata-1542865627584.csv')
    train_data_out = pd.read_csv('archive/Train_Outpatientdata-1542865627584.csv')
    return preparing_data(train_data_ben, train_data_inp, train_data_out)

def feature_engg(test_data):
    '''this function will generate data point after feature engineering on raw data passed'''
    #storing test data columns for merging by these columns
    col_merge=test_data.columns

    ## Lets add both test and train datasets for generting accurate feature engineered data
    #train_data_all = get_train_data()
    train_test_merged = pd.concat([test_data, get_train_data()[col_merge]])
    #remove duplicate entriesu
    train_test_merged = train_test_merged.drop_duplicates(subset='ClaimID')
    
    #average feature grouped by provider
    train_test_merged["PerProviderAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('Provider')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerProviderAvg_DeductibleAmtPaid"]=train_test_merged.groupby('Provider')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerProviderAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('Provider')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerProviderAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('Provider')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerProviderAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('Provider')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerProviderAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('Provider')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerProviderAvg_Age"]=train_test_merged.groupby('Provider')['Age'].transform('mean')
    train_test_merged["PerProviderAvg_NoOfMonths_PartACov"]=train_test_merged.groupby('Provider')['NoOfMonths_PartACov'].transform('mean')
    train_test_merged["PerProviderAvg_NoOfMonths_PartBCov"]=train_test_merged.groupby('Provider')['NoOfMonths_PartBCov'].transform('mean')
    train_test_merged["PerProviderAvg_AdmitForDays"]=train_test_merged.groupby('Provider')['AdmitForDays'].transform('mean')
    #defragmenting dataframe, since after each section of preprocessing dataframe is getting larger this increases time and space complexity
    #so to reduce this we are merging train data and test data after generating some subsection of preprocessing
    #and then we are removing generated columns in train data to reduce space complexity
    # we are doing so because we only require test so after generating features for test data are deleting from train data 
    #this below line generates list of columns that only need to be merged and also in the same order as it was originaly generated in train data
    #maintaining sequence is mandetaory as Std Scaler preprocess in the same sequence as it was trained
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data[['ClaimID']].merge(train_test_merged, on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by Ben ID
    train_test_merged["PerBeneIDAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('BeneID')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerBeneIDAvg_DeductibleAmtPaid"]=train_test_merged.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerBeneIDAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerBeneIDAvg_AdmitForDays"]=train_test_merged.groupby('BeneID')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by attending physician
    train_test_merged["PerAttendingPhysicianAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('AttendingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_DeductibleAmtPaid"]=train_test_merged.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerAttendingPhysicianAvg_AdmitForDays"]=train_test_merged.groupby('AttendingPhysician')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    to_be_ret = temp_cols_list
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by operating physician
    train_test_merged["PerOperatingPhysicianAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('OperatingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_DeductibleAmtPaid"]=train_test_merged.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerOperatingPhysicianAvg_AdmitForDays"]=train_test_merged.groupby('OperatingPhysician')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by dx code group
    train_test_merged["PerDiagnosisGroupCodeAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('DiagnosisGroupCode')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_DeductibleAmtPaid"]=train_test_merged.groupby('DiagnosisGroupCode')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('DiagnosisGroupCode')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerDiagnosisGroupCodeAvg_AdmitForDays"]=train_test_merged.groupby('DiagnosisGroupCode')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by admit dx code
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmAdmitDiagnosisCodeAvg_AdmitForDays"]=train_test_merged.groupby('ClmAdmitDiagnosisCode')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim procedure code 1
    train_test_merged["PerClmProcedureCode_1Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmProcedureCode_1')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_1Avg_AdmitForDays"]=train_test_merged.groupby('ClmProcedureCode_1')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim procedure code 2
    train_test_merged["PerClmProcedureCode_2Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmProcedureCode_2')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmProcedureCode_2')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmProcedureCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmProcedureCode_2Avg_AdmitForDays"]=train_test_merged.groupby('ClmProcedureCode_2')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim dx code 1
    train_test_merged["PerClmDiagnosisCode_1Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_1')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1Avg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_1')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim dx code 2
    train_test_merged["PerClmDiagnosisCode_2Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_2')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2Avg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_2')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature group by claim dx code 3
    train_test_merged["PerClmDiagnosisCode_3Avg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_3')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3Avg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_3')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #average feature grouped by Provider+BeneID, Provider+Attending Physician, Provider+ClmAdmitDiagnosisCode, Provider+ClmProcedureCode_1, Provider+ClmDiagnosisCode_1, Provider+State
    train_test_merged["ClmCount_Provider"]=train_test_merged.groupby(['Provider'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID"]=train_test_merged.groupby(['Provider','BeneID'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_AttendingPhysician"]=train_test_merged.groupby(['Provider','AttendingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_OtherPhysician"]=train_test_merged.groupby(['Provider','OtherPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_OperatingPhysician"]=train_test_merged.groupby(['Provider','OperatingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmAdmitDiagnosisCode"]=train_test_merged.groupby(['Provider','ClmAdmitDiagnosisCode'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','ClmProcedureCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_2"]=train_test_merged.groupby(['Provider','ClmProcedureCode_2'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_3"]=train_test_merged.groupby(['Provider','ClmProcedureCode_3'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_4"]=train_test_merged.groupby(['Provider','ClmProcedureCode_4'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmProcedureCode_5"]=train_test_merged.groupby(['Provider','ClmProcedureCode_5'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_1"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_2"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_2'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_3"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_3'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_4"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_4'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_5"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_5'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_6"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_6'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_7"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_7'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_8"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_8'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_ClmDiagnosisCode_9"]=train_test_merged.groupby(['Provider','ClmDiagnosisCode_9'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_DiagnosisGroupCode"]=train_test_merged.groupby(['Provider','DiagnosisGroupCode'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_AttendingPhysician"]=train_test_merged.groupby(['Provider','BeneID','AttendingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_OtherPhysician"]=train_test_merged.groupby(['Provider','BeneID','OtherPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','BeneID','AttendingPhysician','ClmProcedureCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_1"]=train_test_merged.groupby(['Provider','BeneID','AttendingPhysician','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_OperatingPhysician"]=train_test_merged.groupby(['Provider','BeneID','OperatingPhysician'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','BeneID','ClmProcedureCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_ClmDiagnosisCode_1"]=train_test_merged.groupby(['Provider','BeneID','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
    train_test_merged["ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_1"]=train_test_merged.groupby(['Provider','BeneID','ClmDiagnosisCode_1','ClmProcedureCode_1'])['ClaimID'].transform('count')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #here creating dx code grp for ClmDiagnosisCode_1
    train_test_merged['ClmDiagnosisCode_1_Grp'] = train_test_merged['ClmDiagnosisCode_1'].astype(str).str[0:2]
    #Average features group by dx code group as per proposed idea in abstract - for ClmDiagnosisCode_1
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_1_GrpAvg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_1_Grp')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #here creating dx code grp for ClmDiagnosisCode_1
    train_test_merged['ClmDiagnosisCode_2_Grp'] = train_test_merged['ClmDiagnosisCode_2'].astype(str).str[0:2]
    #Average features group by dx code group as per proposed idea in abstract - for ClmDiagnosisCode_2
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_2_GrpAvg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_2_Grp')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    #here creating dx code grp for ClmDiagnosisCode_3
    train_test_merged['ClmDiagnosisCode_3_Grp'] = train_test_merged['ClmDiagnosisCode_3'].astype(str).str[0:2]
    #Average features group by dx code group as per proposed idea in abstract - for ClmDiagnosisCode_3
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_InscClaimAmtReimbursed"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['InscClaimAmtReimbursed'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_DeductibleAmtPaid"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['DeductibleAmtPaid'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_IPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['IPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_IPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['IPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_OPAnnualReimbursementAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_OPAnnualDeductibleAmt"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['OPAnnualDeductibleAmt'].transform('mean')
    train_test_merged["PerClmDiagnosisCode_3_GrpAvg_AdmitForDays"]=train_test_merged.groupby('ClmDiagnosisCode_3_Grp')['AdmitForDays'].transform('mean')
    #defragmenting df
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    train_test_merged.drop(columns=temp_cols_list, axis=1, inplace=True)
    
    
    # for calculating tf_idf on claim dx codes
    dx_col_list = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4']
    temp_data = tf_idf_on_dx_cpt(train_test_merged[['ClaimID', 'Provider']+dx_col_list], dx_col_list)
    #defragmenting df
    test_data_all = test_data_all.merge(temp_data, on=['ClaimID', 'Provider'])
    
    
    # for calculating tf_idf on claim cpt codes
    cpt_col_list = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3']
    temp_data = tf_idf_on_dx_cpt(train_test_merged[['ClaimID', 'Provider']+cpt_col_list], cpt_col_list)
    #defragmenting df
    test_data_all = test_data_all.merge(temp_data, on=['ClaimID', 'Provider'])
    del temp_data
    

    ## Lets Convert types of gender and race to categorical.
    train_test_merged.Gender=train_test_merged.Gender.astype('category')
    train_test_merged.Race=train_test_merged.Race.astype('category')

    # Lets create dummies for categorrical columns.
    train_test_merged=pd.get_dummies(train_test_merged,columns=['Gender','Race'],drop_first=True)
    test_data = test_data.loc[:, ~test_data.columns.isin(['Gender','Race'])]
    temp_cols_list = sorted(set(train_test_merged.columns)-set(test_data.columns), key=list(train_test_merged.columns).index)
    test_data_all = test_data_all.merge(train_test_merged[['ClaimID']+temp_cols_list], on='ClaimID')
    del train_test_merged
    
    ##### Lets impute numeric columns with 0
    cols1 = test_data_all.select_dtypes([np.number]).columns
    test_data_all[cols1]=test_data_all[cols1].fillna(value=0)
    
    # Lets remove unnecessary columns ,as we grouped based on these columns and derived maximum infromation from them.
    remove_these_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
           'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
           'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
           'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
           'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
           'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
           'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
           'ClmAdmitDiagnosisCode', 'AdmissionDt',
           'DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD',
            'State', 'County', 'ClmDiagnosisCode_1_Grp', 'ClmDiagnosisCode_2_Grp', 'ClmDiagnosisCode_3_Grp', 'Gender','Race']

    test_data_all = test_data_all.drop(axis=1, columns=remove_these_columns)

    ## Lets apply StandardScaler and transform values to its z form,where 99.7% values range between -3 to 3.
    sc = load('std_scaler.bin')   # MinMaxScaler
    X_test=sc.transform(test_data_all.iloc[:,1:])   #Apply Standard Scaler to unseen data
    return X_test

def tf_idf_on_dx_cpt(dataframe, dx_or_cpt_col_list):
    '''this function calculates tf, idf and tf_idf features on dx codes or cpt codes as per proposed idea of abstract document'''
    N = dataframe.groupby('Provider')['Provider'].count().shape[0] #no of unique provider = no of document corpus
    
    for each_col in dx_or_cpt_col_list:
        term_freq = dataframe.groupby(['Provider', each_col])[['ClaimID']].count().reset_index()
        term_freq.rename(columns={'ClaimID': each_col+'_term'}, inplace=True)
        dataframe = dataframe.merge(term_freq, on=['Provider', each_col], how='outer')
        no_of_dx_in_each_prov = dataframe.groupby('Provider')[each_col].count().reset_index()
        no_of_dx_in_each_prov.rename(columns={each_col:each_col+'_doc'}, inplace=True)
        dataframe = dataframe.merge(no_of_dx_in_each_prov, on=['Provider'], how='outer')
        dataframe[each_col+'TF'] = dataframe[each_col+'_term']/dataframe[each_col+'_doc']

        no_of_doc_containing_dx = dataframe.groupby(each_col)[['Provider']].count().reset_index()
        no_of_doc_containing_dx.rename(columns={'Provider':each_col+'_IDF'}, inplace=True)
        no_of_doc_containing_dx[each_col+'_IDF'] = np.log2(N/no_of_doc_containing_dx[each_col+'_IDF'])
        dataframe = dataframe.merge(no_of_doc_containing_dx, on=each_col, how='outer')
        dataframe[each_col+'TF-IDF'] = dataframe[each_col+'TF']*dataframe[each_col+'_IDF']
        dataframe.drop([each_col, each_col+'_term', each_col+'_doc'], axis=1, inplace=True)

    return dataframe

def fraud_prov_predict(raw_data):
    '''this function takes raw data as input, preprocess and featurize it and returned the predicted value'''
    start=time.time()
    featured_data = feature_engg(raw_data)
    end=time.time()
    st.write('time taken in feature engg ', end-start)
    start=time.time()
    xgb_clf = XGBClassifier(booster='gbtree')
    xgb_clf.load_model('XGB_Model.json')
    y_pred = xgb_clf.predict(featured_data)
    end=time.time()
    st.write('time taken in prediction ', end-start)
    return y_pred

@st.cache
def get_data():
    test_data_ben = pd.read_csv('archive/Test_Beneficiarydata-1542969243754.csv')
    test_data_inp = pd.read_csv('archive/Test_Inpatientdata-1542969243754.csv')
    test_data_out = pd.read_csv('archive/Test_Outpatientdata-1542969243754.csv')
    test_ddata_merged = preparing_data(test_data_ben, test_data_inp, test_data_out)
    return (test_data_ben, test_data_inp, test_data_out, test_ddata_merged)

st.title('Medicare Fraud Provider Prediction')  
df = get_data()

with st.sidebar:
    side_option = st.selectbox('Menu', ['Data Sample', 'Prediction', 'View Source Code'])

if side_option=='Data Sample':
    category = st.multiselect(label='Select Type of Sample data', options=['Beneficiary', 'Inpatient', 'Outpatient', 'Merged Data'])
    
    for cat in category:
        if cat == 'Beneficiary':
            st.write("Beneficiary data sample", df[0].head(100))
        elif cat == 'Inpatient':
            st.write("Inpatient data sample", df[1].head(100))
        elif cat == 'Outpatient':
            st.write("Outpatient data sample", df[2].head(100))
        elif cat == 'Merged Data':
            st.write("Merged data sample", df[3].head(100))
elif side_option=='Prediction':
    with st.sidebar:
        how_pred = st.selectbox(label='How you want to select sample for prediction', options=['Number', 'Range'])
    check_empty_dataset = 0
    if how_pred == 'Number':
        with st.sidebar:
            nth_val = st.number_input(label='Enter nth value', min_value=-df[3].shape[0], max_value=df[3].shape[0], value=0, key='for number option')
            nth_val_options = st.radio(label='What to treat the value', options=('Top '+str(nth_val)+'s', 'Bottom '+str(nth_val)+'s', str(nth_val)+' Random sample', str(nth_val)+'th Index', str(nth_val)+'th row'))
            if nth_val_options == 'Top '+str(nth_val)+'s':
                sample_test_data = df[3].head(nth_val)
            elif nth_val_options == 'Bottom '+str(nth_val)+'s':
                sample_test_data = df[3].tail(nth_val)
            elif nth_val_options == str(nth_val)+' Random sample':
                sample_test_data = df[3].sample(nth_val)
            elif nth_val_options == str(nth_val)+'th Index':
                sample_test_data = df[3].iloc[[nth_val]]
            else:
                sample_test_data = df[3].loc[[nth_val]]
    else:
        with st.sidebar:
            st.write('Enter range to select sample data for prediction')
            lower_lim = st.number_input(label='Enter lower limit', min_value=-df[3].shape[0], max_value=df[3].shape[0], value=0, key='for range lower limit')
            upper_lim = st.number_input(label='Enter upper limit', min_value=-df[3].shape[0], max_value=df[3].shape[0], value=10, key='for range upper limit')
            if st.checkbox(label='Index'):
                sample_test_data = df[3].iloc[lower_lim:upper_lim]
            else:
                sample_test_data = df[3].loc[lower_lim:upper_lim]
    check_empty_dataset = sample_test_data.shape[0]
    st.write('Selected sample data', sample_test_data)
    if check_empty_dataset==0:
        st.write('Please enter valid range/number of sample')
        button_disable = True
    else:
        button_disable = False
    if st.button('Predict', disabled=button_disable):
        start = time.time()
        with st.spinner("Please wait while processing, it will take less than a minute..."):
            pred_y = fraud_prov_predict(sample_test_data)
        st.success("Done..!")
        end = time.time()
        st.write('Time taken by prediction function to preprocess and predict ', end-start)
        sample_test_data['PridictedFraud'] = pred_y
        st.write('Predicted sample data', sample_test_data)
        """
    st.code(code, language='python')
        
