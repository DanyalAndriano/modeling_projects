import pandas as pd


def unique(data, num1=None, num2=None, categories=None):
    if categories == True:
        data = data.select_dtypes(include=['category','object', 'bool'])
        for col in data.columns[num1:num2]:
            print('Unique Range for {} is: {}'.format(col, data[col].unique()))
    else:
        for col in data.columns[num1:num2]:
            print('Unique Range for {} is: {}'.format(col, data[col].unique()))
            
def save_data(df, filename, path=None):
    # 1) remove commas from strings
    # 2) save as tab delimited
    if not path:
        df.to_csv(filename)

def add_season(data, feature='month'):
    df = data.copy()
    
    summer = ['jun', 'jul', 'aug']
    fall = ['sep', 'oct', 'nov']
    winter = ['dec', 'jan', 'feb']
    spring = ['mar', 'apr', 'may']
    seasons = [summer, fall, winter, spring]

    for month in summer:
        df.loc[df[feature] == month, 'season'] = 'summer'
    for month in winter:
        df.loc[df[feature] == month, 'season'] = 'winter'
    for month in spring:
        df.loc[df[feature] == month, 'season'] = 'spring'
    for month in fall:
        df.loc[df[feature] == month, 'season'] = 'fall'
    
    df.drop('month', axis=1, inplace=True)
    return df

def one_hot_df(df, target='y'):
    data = df.copy()
    if target:
        data.drop([target], axis=1, inplace=True)
        
    coded_df = pd.DataFrame()

    for col in df.select_dtypes(include=['category','object','bool']):
        coded_df[col] = df[col]
        data.drop(col, axis=1, inplace=True)
        
    one_hot_df = pd.get_dummies(coded_df)
    data = pd.concat([data, one_hot_df], axis=1)
    return data
