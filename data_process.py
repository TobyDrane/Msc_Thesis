import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from causalnex.discretiser import Discretiser


# Process the input data (input: dataframe)
# # returns a dataframe with all correct independent variables within (NOTE: Including "time" types)
def process_independent_data(df: pd.DataFrame):
    EffectiveLine = df.EffectiveLine
    df['Full_GGTP'] = df['GGTP'] / EffectiveLine
    df['Full_GGWP'] = df['GGWP'] / EffectiveLine
    df['Full_GrossGrossModelPrice'] = df['GrossGrossModelPrice'] / EffectiveLine
    df['Full_GrossNetModelPrice'] = df['GrossNetModelPrice'] / EffectiveLine
    df['Full_GrossGrossTechnicalPrice'] = df['GrossGrossTechnicalPrice'] / EffectiveLine
    df['Full_GrossNetTechnicalPrice'] = df['GrossNetTechnicalPrice'] / EffectiveLine

    # Drop Brit shares, result of performing the above actions and producing 100% shares
    to_drop = ['GGTP', 'GGWP', 'GrossGrossModelPrice', 'GrossNetModelPrice',
               'GrossGrossTechnicalPrice', 'GrossNetTechnicalPrice']
    # Drop n/a's
    to_drop_na = ['EclipsePolicyID', 'Exposure', 'GroupClass', 'LifetimePolicyReference', 'LinePct',
                  'PolicyReference', 'PriorEclipsePolicyID', 'PriorPolicyRef', 'GNWP']
    # Drop dependents
    to_drop_depnds = ['ClaimCount', 'ClaimFrequency', 'CLR_Cat', 'CLR_ExCat', 'Full_Inc_ExCat',
                      'ILR_ExCat', 'ILR_Cat', 'Inc', 'Inc_Cat', 'Inc_ExCat', 'EffectiveLine']
    df = df.drop(labels=(to_drop + to_drop_depnds + to_drop_na), axis=1)

    # -------------------------
    # Handle Categorical values
    # -------------------------

    # PlacingBasis - 4 different values ('OM', 'Re', 'Binder', 'Other')
    # drop this column and install 4 separate columns
    # ('Indicator_OM', 'Indicator_Binder', 'Indicator_Re', 'Indicator_Other')
    df = df.join(pd.get_dummies(df.PlacingBasis, prefix='Indicator'))
    df = df.drop(labels='PlacingBasis', axis=1)
    # Bucket Other with Binder and drop Other
    df['Indicator_Binder'] = np.bitwise_or(df.Indicator_Binder, df.Indicator_Other)
    df = df.drop(labels='Indicator_Other', axis=1)

    # Do same with Excess
    df = df.join(pd.get_dummies(df.Excess, prefix='Indicator'))
    df = df.drop(labels='Excess', axis=1)
    # Do the same with LeaderStatus
    df = df.join(pd.get_dummies(df.LeaderStatus, prefix='Indicator'))
    df = df.drop(labels='LeaderStatus', axis=1)

    # We drop the description columns when we also have a code, and then only code the categorical code
    df = df.drop(labels=['StatsMinorClassDescription', 'SubClass'], axis=1)

    # NOTE: We also drop BrokerUltimateName, PLR_band, PLR_band_ex_adj, Territory
    df = df.drop(labels=['BrokerUltimateName', 'PLR_band', 'PLR_band_ex_adj', 'Territory'], axis=1)
    # Temporary remove StatsMinorClassCode
    # NOTE: I'm not too sure if we need this value but for now to get things working we shall remove it
    df = df.drop(labels='StatsMinorClassCode', axis=1)

    return df


# Same helper function type but for the dependent data
def process_dependent_data(df: pd.DataFrame):
    EffectiveLine = df['EffectiveLine']

    # These are our choices of dependent values
    to_filter = ['ClaimCount', 'CLR_Cat', 'CLR_ExCat', 'Full_Inc_ExCat', 'ILR_Cat', 'ILR_ExCat']
    df_filtered = df.filter(items=to_filter)

    df_filtered['Full_ClaimFrequency'] = df['ClaimCount'] / (df['Full_GNWP'] / 1000000)
    df_filtered['Full_Inc'] = df['Inc'] / EffectiveLine
    df_filtered['ILR'] = df['ILR_ExCat'] + df['ILR_Cat']

    # Add the claim count classification
    df_filtered['ClaimCountClassification'] = df_filtered['ClaimCount'] > 0

    return df_filtered


# Need to remove time types see Notion for information on why
def remove_time_types(df: pd.DataFrame):
    to_drop = ['ExpiryDate', 'InceptionDate', 'RenewalDate', 'YOA', 'YOA_cat', 'YOA_recent']
    df = df.drop(labels=to_drop, axis=1)
    return df

# Handy function to split a dataframe into train, val and test datasets
# Splits into 70%, 15%, 15% respectively
def df_train_val_test_split(total_data: pd.DataFrame):
    # Initial split into train and test
    data_train, data_test = train_test_split(total_data, train_size=0.7)
    # Second split test into 50/50 val
    data_val, data_test = train_test_split(data_test, train_size=0.5)

    return data_train, data_val, data_test


# Same as above but produces no validation test set, only a train and test
def df_train_test_split(total_data: pd.DataFrame, train_size):
    data_train, data_test = train_test_split(total_data, train_size=train_size)
    return data_train, data_test


# Probably one of the most important helper functions
# Takes the dataset with a desired target and returns two tuples of the dependant & independent
# Each tuple contains train, val and test data splits - 70%, 15% & 15% respectively
def create_train_val_test(target, df_indps: pd.DataFrame, df_dpnds: pd.DataFrame):
    df_indps_filtered = df_indps
    # NOTE: We normalize the dataset independent values only
    df_indps_filtered = pd.DataFrame(scale(df_indps_filtered), columns=df_indps_filtered.columns)

    df_dpnds_filtered = df_dpnds.filter(items=[target])
    total_data = df_indps_filtered.join(df_dpnds_filtered)

    # Now perform splits
    data_train, data_val, data_test = df_train_val_test_split(total_data)

    # Now we remove the target variable to create X&Y datasets
    data_d_train = data_train.pop(target).to_frame()
    data_d_val = data_val.pop(target).to_frame()
    data_d_test = data_test.pop(target).to_frame()

    return (data_train, data_val, data_test), (data_d_train, data_d_val, data_d_test)


def create_train_test(target, df_indps: pd.DataFrame, df_dpnds: pd.DataFrame, train_size):
    df_indps_filtered = df_indps
    # df_indps_filtered = pd.DataFrame(scale(df_indps_filtered), columns=df_indps_filtered.columns)

    df_dpnds_filtered = df_dpnds.filter(items=[target])
    total_data = df_indps_filtered.join(df_dpnds_filtered)

    # Perform splits
    data_train, data_test = df_train_test_split(total_data, train_size)

    # Now we remove the target variable to create X&Y datasets
    data_d_train = data_train.pop(target).to_frame()
    data_d_test = data_test.pop(target).to_frame()

    return (data_train, data_test), (data_d_train, data_d_test)


# Random Over Sampling to handle class imbalance
# class imbalance is core issue of our classification algorithms
def handle_imbalance_oversample(X: pd.DataFrame, Y: pd.DataFrame):
    print('Balance Before', Y.value_counts())
    oversample = RandomOverSampler()
    X_over, Y_over = oversample.fit_resample(X, Y)
    print('Balance After', Y_over.value_counts())
    return X_over, Y_over


def SMOTE_imbalance(X: pd.DataFrame, Y: pd.DataFrame):
    print('Balance Before', Y.value_counts())
    oversample = SMOTE()
    X_over, Y_over = oversample.fit_resample(X, Y)
    print('Balance After', Y_over.value_counts())
    return X_over, Y_over


def standard_scale(X: pd.DataFrame, Y: pd.DataFrame):
    names = X.columns
    X_scaled = StandardScaler().fit_transform(X, Y)
    return pd.DataFrame(X_scaled, columns=names)


def remove_specific_labels(df: pd.DataFrame, labels):
    return df.drop(labels=labels, axis=1)


def get_specific_labels(df: pd.DataFrame, labels):
    return df.filter(items=labels)


def log_labels(df: pd.DataFrame, labels_to_log):
    df = df.filter(items=labels_to_log)
    for column in df:
        df['Log_' + column] = np.log(df[column] + 1)
    return df


def log_all(df: pd.DataFrame):
    original_cols = df.columns
    for column in df:
        df['Log_' + column] = np.log(df[column] + 1)
    # Now remove original columns
    df = remove_specific_labels(df, original_cols)
    return df


def discreatize_data(df: pd.DataFrame, bins):
    discreatizer = Discretiser(num_buckets=bins)
    for column in df:
        df[column] = discreatizer.fit_transform(df[column].to_numpy())
    return df


# SubClassCode is categorical but has the issue of containing alot of different values > 50
# thus cannot just use dummy variables, so we therefore take top frequency values and some expert
# knowledge that transport is useful
def handle_subclass_code_cyber(df: pd.DataFrame, is_using_energy):
    _df = df.copy()
    if is_using_energy:
        """ Energy Dataset """
        # Offshore Operating
        _df['SubClassCode_OFSO'] = (_df['SubClassCode'] == 47301).astype(int)
        # Onshore Control of Well
        _df['SubClassCode_ONSCW'] = (_df['SubClassCode'] == 47201).astype(int)
        # General cost of control
        _df['SubClassCode_GCC'] = (_df['SubClassCode'] == 47200).astype(int)
        # Offshore Control of Well
        _df['SubClassCode_OFCW'] = (_df['SubClassCode'] == 47202).astype(int)
        # General offshore energy
        _df['SubClassCode_GOFSE'] = (_df['SubClassCode'] == 47300).astype(int)

        # Bin everything else into an "other"
        not_other_codes = [47301, 47201, 47200, 47202, 47300]
        _df['SubClassCode_Oth'] = (~_df['SubClassCode'].isin(not_other_codes)).astype(int)
    else:
        """ Cyber Dataset """
        # Durable Goods Manufacturer
        _df['SubClassCode_DGM'] = (_df['SubClassCode'] == 96125).astype(int)
        # Clothing Retail
        _df['SubClassCode_Re'] = (_df['SubClassCode'] == 96112).astype(int)
        # Insurance - Health
        _df['SubClassCode_IH'] = (_df['SubClassCode'] == 96106).astype(int)
        # Technology
        _df['SubClassCode_Te'] = (_df['SubClassCode'] == 90310).astype(int)
        # Professional Services
        _df['SubClassCode_PS'] = (_df['SubClassCode'] == 90314).astype(int)
        # Business Finance / Banking
        _df['SubClassCode_BFB'] = (_df['SubClassCode'] == 96101).astype(int)
        # Retail / Hospitality
        _df['SubClassCode_RH'] = (_df['SubClassCode'] == 90317).astype(int)
        # Healthcare Provider
        _df['SubClassCode_HP'] = (_df['SubClassCode'] == 90313).astype(int)
        # Privacy
        _df['SubClassCode_Pr'] = (_df['SubClassCode'] == 90312).astype(int)
        # Insurance - Commercial
        _df['SubClassCode_IC'] = (_df['SubClassCode'] == 96104).astype(int)

        # Create a transport bucket - useful for Cyber only
        # Airlines, Air Cargo, Automotive Retail, Rail - Freight, Rail - Passengers & Marine (Cargo)
        # 96127, 96128, 96111, 90613, 90677, 90536
        transport_codes = [96127, 96128, 96111, 90613, 90677, 90536]
        _df['SubClassCode_Tr'] = (_df['SubClassCode'].isin(transport_codes)).astype(int)

        # Finally bin everything else into an "other"
        not_other_codes = [96125, 96112, 96106, 90310, 90314, 96101, 90317, 90313, 90312, 96104] + transport_codes
        _df['SubClassCode_Oth'] = (~_df['SubClassCode'].isin(not_other_codes)).astype(int)

    return _df