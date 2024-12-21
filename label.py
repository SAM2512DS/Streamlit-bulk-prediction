from sklearn.preprocessing import LabelEncoder
def encoder(df):
    encoder=LabelEncoder()
    for column in range(len(df.columns)):
        df[df.columns[column]]=encoder.fit_transform(df[df.columns[column]])
        return df