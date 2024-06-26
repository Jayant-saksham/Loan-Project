from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class Constants:
    """
    A class containing constants used throughout the project.
    """

    CSV_FILE_NAME = 'loan_data.csv'

    # EXTRACT_FILE_LOCATION = "../data/loan_data.csv"

    K_FOLDS = 5

    MODELS = {
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(12, 12))
    }

    COLUMN_MEANING = {
        'SK_ID_CURR': 'ID of loan in our sample',
        'TARGET': 'Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases)',
        'NAME_CONTRACT_TYPE': 'Identification if loan is cash or revolving',
        'CODE_GENDER': 'Gender of the client',
        'FLAG_OWN_CAR': 'Flag if the client owns a car',
        'FLAG_OWN_REALTY': 'Flag if client owns a house or flat',
        'CNT_CHILDREN': 'Number of children the client has',
        'AMT_INCOME_TOTAL': 'Income of the client',
        'AMT_CREDIT': 'Credit amount of the loan',
        'AMT_ANNUITY': 'Loan annuity',
        'AMT_GOODS_PRICE': 'For consumer loans it is the price of the goods for which the loan is given',
        'NAME_TYPE_SUITE': 'Who was accompanying client when he was applying for the loan',
        'NAME_INCOME_TYPE': 'Clients income type (businessman, working, maternity leave,…)',
        'NAME_EDUCATION_TYPE': 'Level of highest education the client achieved',
        'NAME_FAMILY_STATUS': 'Family status of the client',
        'NAME_HOUSING_TYPE': 'What is the housing situation of the client (renting, living with parents, ...)',
        'REGION_POPULATION_RELATIVE': 'Normalized population of region where client lives (higher number means the client lives in more populated region)',
        'DAYS_BIRTH': 'Client\'s age in days at the time of application',
        'DAYS_EMPLOYED': 'How many days before the application the person started current employment',
        'DAYS_REGISTRATION': 'How many days before the application did client change his registration',
        'DAYS_ID_PUBLISH': 'How many days before the application did client change the identity document with which he applied for the loan',
        'OWN_CAR_AGE': 'Age of client\'s car',
        'FLAG_MOBIL': 'Flag if client has mobile phone',
        'FLAG_EMP_PHONE': 'Flag if client has work phone',
        'FLAG_WORK_PHONE': 'Flag if client has home phone',
        'FLAG_CONT_MOBILE': 'Flag if client has email',
        'FLAG_PHONE': 'Type of occupation of the client',
        'FLAG_EMAIL': 'Number of family members',
        'OCCUPATION_TYPE': 'Our rating of the region where client lives (1,2,3)',
        'CNT_FAM_MEMBERS': 'Our rating of the region where client lives with taking city into account (1,2,3)',
        'REGION_RATING_CLIENT': 'On which day of the week did the client apply for the loan',
        'REGION_RATING_CLIENT_W_CITY': 'Approximately at what hour did the client apply for the loan',
        'WEEKDAY_APPR_PROCESS_START': 'Flag if client\'s permanent address does not match contact address (1=different, 0=same, at region level)',
        'HOUR_APPR_PROCESS_START': 'Flag if client\'s permanent address does not match work address (1=different, 0=same)',
        'REG_REGION_NOT_LIVE_REGION': 'Flag if client\'s contact address does not match work address (1=different, 0=same)',
        'REG_REGION_NOT_WORK_REGION': 'Flag if client\'s permanent address does not match contact address (1=different, 0=same)',
        'LIVE_REGION_NOT_WORK_REGION': 'Flag if client lives in region where he applied for the loan (1=yes, 0=no)',
        'REG_CITY_NOT_LIVE_CITY': 'Flag if client\'s contact address does not match his permanent address (1=different, 0=same)',
        'REG_CITY_NOT_WORK_CITY': 'Flag if client\'s contact address does not match his work address (1=different, 0=same)',
        'LIVE_CITY_NOT_WORK_CITY': 'Flag if client lives in city where he applied for the loan (1=yes, 0=no)',
        'ORGANIZATION_TYPE': 'Type of organization where client works',
        'EXT_SOURCE_1': 'Normalized score from external data source',
        'EXT_SOURCE_2': 'Normalized score from external data source',
        'EXT_SOURCE_3': 'Normalized score from external data source',
        'APARTMENTS_AVG': 'Normalized information about building where the client lives, Average',
        'BASEMENTAREA_AVG': 'Normalized information about building where the client lives, Average',
        'YEARS_BEGINEXPLUATATION_AVG': 'Normalized information about building where the client lives, Average',
        'YEARS_BUILD_AVG': 'Normalized information about building where the client lives, Average',
        'COMMONAREA_AVG': 'Normalized information about building where the client lives, Average',
        'ELEVATORS_AVG': 'Normalized information about building where the client lives, Average',
        'ENTRANCES_AVG': 'Normalized information about building where the client lives, Average',
        'FLOORSMAX_AVG': 'Normalized information about building where the client lives, Average',
        'FLOORSMIN_AVG': 'Normalized information about building where the client lives, Average',
        'LANDAREA_AVG': 'Normalized information about building where the client lives, Average',
        'LIVINGAPARTMENTS_AVG': 'Normalized information about building where the client lives, Average',
        'LIVINGAREA_AVG': 'Normalized information about building where the client lives, Average',
        'NONLIVINGAPARTMENTS_AVG': 'Normalized information about building where the client lives, Average',
        'NONLIVINGAREA_AVG': 'Normalized information about building where the client lives, Average',
        'APARTMENTS_MODE': 'Normalized information about building where the client lives, Mode',
        'BASEMENTAREA_MODE': 'Normalized information about building where the client lives, Mode',
        'YEARS_BEGINEXPLUATATION_MODE': 'Normalized information about building where the client lives, Mode',
        'YEARS_BUILD_MODE': 'Normalized information about building where the client lives, Mode',
        'COMMONAREA_MODE': 'Normalized information about building where the client lives, Mode',
        'ELEVATORS_MODE': 'Normalized information about building where the client lives, Mode',
        'ENTRANCES_MODE': 'Normalized information about building where the client lives, Mode',
        'FLOORSMAX_MODE': 'Normalized information about building where the client lives, Mode',
        'FLOORSMIN_MODE': 'Normalized information about building where the client lives, Mode',
        'LANDAREA_MODE': 'Normalized information about building where the client lives, Mode',
        'LIVINGAPARTMENTS_MODE': 'Normalized information about building where the client lives, Mode',
        'LIVINGAREA_MODE': 'Normalized information about building where the client lives, Mode',
        'NONLIVINGAPARTMENTS_MODE': 'Normalized information about building where the client lives, Mode',
        'NONLIVINGAREA_MODE': 'Normalized information about building where the client lives, Mode',
        'APARTMENTS_MEDI': 'Normalized information about building where the client lives, Median',
        'BASEMENTAREA_MEDI': 'Normalized information about building where the client lives, Median',
        'YEARS_BEGINEXPLUATATION_MEDI': 'Normalized information about building where the client lives, Median',
        'YEARS_BUILD_MEDI': 'Normalized information about building where the client lives, Median',
        'COMMONAREA_MEDI': 'Normalized information about building where the client lives, Median',
        'ELEVATORS_MEDI': 'Normalized information about building where the client lives, Median',
        'ENTRANCES_MEDI': 'Normalized information about building where the client lives, Median',
        'FLOORSMAX_MEDI': 'Normalized information about building where the client lives, Median',
        'FLOORSMIN_MEDI': 'Normalized information about building where the client lives, Median',
        'LANDAREA_MEDI': 'Normalized information about building where the client lives, Median',
        'LIVINGAPARTMENTS_MEDI': 'Normalized information about building where the client lives, Median',
        'LIVINGAREA_MEDI': 'Normalized information about building where the client lives, Median',
        'NONLIVINGAPARTMENTS_MEDI': 'Normalized information about building where the client lives, Median',
        'NONLIVINGAREA_MEDI': 'Normalized information about building where the client lives, Median',
        'FONDKAPREMONT_MODE': 'Normalized information about building where the client lives, Modality of the loan',
        'HOUSETYPE_MODE': 'Normalized information about building where the client lives, Modality of the house',
        'TOTALAREA_MODE': 'Normalized information about building where the client lives, Total area',
        'WALLSMATERIAL_MODE': 'Normalized information about building where the client lives, Modality of the wall',
        'EMERGENCYSTATE_MODE': 'Normalized information about building where the client lives, Modality of the emergency',
        'OBS_30_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings with observable 30 DPD (days past due) default',
        'DEF_30_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings defaulted on 30 DPD (days past due)',
        'OBS_60_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings with observable 60 DPD (days past due) default',
        'DEF_60_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings defaulted on 60 (days past due)',
        'DAYS_LAST_PHONE_CHANGE': 'How many days before application did client change phone',
        'FLAG_DOCUMENT_2': 'Did client provide document 2',
        'FLAG_DOCUMENT_3': 'Did client provide document 3',
        'FLAG_DOCUMENT_4': 'Did client provide document 4',
        'FLAG_DOCUMENT_5': 'Did client provide document 5',
        'FLAG_DOCUMENT_6': 'Did client provide document 6',
        'FLAG_DOCUMENT_7': 'Did client provide document 7',
        'FLAG_DOCUMENT_8': 'Did client provide document 8',
        'FLAG_DOCUMENT_9': 'Did client provide document 9',
        'FLAG_DOCUMENT_10': 'Did client provide document 10',
        'FLAG_DOCUMENT_11': 'Did client provide document 11',
        'FLAG_DOCUMENT_12': 'Did client provide document 12',
        'FLAG_DOCUMENT_13': 'Did client provide document 13',
        'FLAG_DOCUMENT_14': 'Did client provide document 14',
        'FLAG_DOCUMENT_15': 'Did client provide document 15',
        'FLAG_DOCUMENT_16': 'Did client provide document 16',
        'FLAG_DOCUMENT_17': 'Did client provide document 17',
        'FLAG_DOCUMENT_18': 'Did client provide document 18',
        'FLAG_DOCUMENT_19': 'Did client provide document 19',
        'FLAG_DOCUMENT_20': 'Did client provide document 20',
        'FLAG_DOCUMENT_21': 'Did client provide document 21',
        'AMT_REQ_CREDIT_BUREAU_HOUR': 'Number of enquiries to Credit Bureau about the client one hour before application',
        'AMT_REQ_CREDIT_BUREAU_DAY': 'Number of enquiries to Credit Bureau about the client one day before application',
        'AMT_REQ_CREDIT_BUREAU_WEEK': 'Number of enquiries to Credit Bureau about the client one week before application',
        'AMT_REQ_CREDIT_BUREAU_MON': 'Number of enquiries to Credit Bureau about the client one month before application',
        'AMT_REQ_CREDIT_BUREAU_QRT': 'Number of enquiries to Credit Bureau about the client 3 month before application',
        'AMT_REQ_CREDIT_BUREAU_YEAR': 'Number of enquiries to Credit Bureau about the client one year before application'
    }
