import pandas as pd 
 
df = pd.read_csv('train.csv') 
 
df.drop(['id', 'bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 
    'relation', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start',  
    'career_end'], axis=1, inplace=True) 
 
# df.info() 
 
# print('Люди с телефонами покупали курс чаще') 
# temp = df.groupby(by = 'has_mobile')['result'].value_counts() 
# print(temp) 
 
# print('Мужчины идиоты, ведь реже покупают курс') 
# temp = df.groupby(by = 'sex')['result'].value_counts() 
# print(temp) 
# print('Да лол. Почему я не удивлен?') 
 
# print('Люди с большим количеством подписчиков, реже покупают курс') 
# temp = round(df.groupby(by = 'result')['followers_count'].agg(['max', 'mean']), 2) 
# print(temp) 
 
# print(df['sex']) 
 
def sex_apply(sex): 
    if sex == 2: 
        return 0 
    return 1 
 
df['sex'] = df['sex'].apply(sex_apply) 
 
def edu_form_apply(edu_form): 
    if edu_form == 'Distance Learning': 
        return 'Distance Learning' 
    elif edu_form == 'Part-time': 
        return 'Part-time' 
    else: 
        return 'Full-time' 
 
df['education_form'] = df['education_form'].apply(edu_form_apply) 
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form']) 
df.drop(['education_form'], axis=1, inplace=True) 
 
def edu_status_apply(edu_status): 
    if edu_status == 'Undergraduate applicant': 
        return 0 
    elif edu_status == "Student (Master's)" or edu_status == "Student (Bachelor's)" or edu_status == "Student (Specialist)": 
        return 1 
    elif edu_status == "Alumnus (Master's)" or edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Specialist)": 
        return 2 
    else: 
        return 3 
 
df['education_status'] = df['education_status'].apply(edu_status_apply) 
 
def langs_apply(langs): 
    langs = langs.split(';') 
    if 'Русский' in langs: 
        return 0 
    else: 
        return 1 
 
df['langs'] = df['langs'].apply(langs_apply) 
 
def ocu_type_apply(ocu_type): 
    if ocu_type == 'work': 
        return 0 
    else: 
        return 1 
 
df['occupation_type'] = df['occupation_type'].apply(ocu_type_apply) 
 
df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 

x = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred) * 100, 2))
# print('Confusion matrix:')
# print(confusion_matrix(y_test, y_pred))
