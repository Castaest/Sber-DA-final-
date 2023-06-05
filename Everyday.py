#!/usr/bin/env python
# coding: utf-8

from tqdm.auto import tqdm
import requests      
import json          
import time          
import os            
import re
import pandas as pd 
import numpy as np 
from urllib.parse import urlencode
import warnings
from datetime import datetime, timedelta
import pymorphy2
from sklearn.preprocessing import MultiLabelBinarizer
import pickle as pkl
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco
import torch

warnings.filterwarnings('ignore')

# извлечение размещенных вчера вакансий

d = datetime.today() - timedelta(days = 1)

query = {
    'parent_area': 113,
    'per_page': 50,
    'professional_roles': [10, 156, 157, 165],
    'date_from' : d.strftime('%Y-%m-%d'),
    'date_to' : datetime.today().strftime('%Y-%m-%d')
}

query_copy = query.copy()

roles = '&'.join([f'professional_role={r}' for r in query_copy.pop('professional_roles')])

url_params = roles + (f'&{urlencode(query_copy)}' if len(query_copy) > 0 else '')

target_url = 'https://api.hh.ru/vacancies/' + '?' + url_params
num_pages = requests.get(target_url).json()['pages']

print(f'Найдены {requests.get(target_url).json()["found"]} вакансий')

ids = []
for idx in range(num_pages + 1):
    response = requests.get(target_url, {'page': idx})
    data = response.json()
    if 'items' not in data:
        break
    ids.extend(x['id'] for x in data['items'])

data = []
j = 0
for i in tqdm(range(len(ids))): 
    url = f'https://api.hh.ru/vacancies/{ids[i]}'
    res = requests.get(url)
    data.append(res.json())
    j += 1
    if j == 5:
        j = 0
        time.sleep(2)

def clean_tags(html_text: str) -> str:
        """Функция удаляет HTML теги из строки
        ----------
        html_text: str - входная строка с тегами
        returns: str - выходная строка без тегов
        """
        pattern = re.compile("<.*?>")
        return re.sub(pattern, "", html_text)


def get_vacancy(x: dict) -> dict:
        """Функция извлекает только нужные поля из словаря вакансии.
        ----------
        x: dict -  словарь вакансии, скачанный с hh.ru
        returns: dict - словарь с необходимыми полями

        """	
        out = {}
        out['id'] = x['id']
        out['name'] = x['name'].lower()
        out['area'] = x['area']['name']
        out['salary_from'] = None if x['salary'] is None else x['salary']['from']
        out['salary_to'] = None if x['salary'] is None else x['salary']['to']
        out['salary_currency'] = None if x['salary'] is None else x['salary']['currency']
        out['salary_gross'] = None if x['salary'] is None else x['salary']['gross']
        out['city'] = None if x['address'] is None else x['address']['city']
        out['experience'] = x['experience']['name']
        out['schedule'] = x['schedule']['name']
        out['employment'] = x['employment']['name']
        out['description'] = clean_tags(x['description'])
        out['key_skills'] = x['key_skills']
        out['accept_handicapped']  = x['accept_handicapped']
        out['professional_roles'] = x['professional_roles']
        out['employer'] = x['employer']['name']
        out['published_at'] = x['published_at'] 
        out['has_test'] = None if x['has_test'] is None  else  x['has_test']
        out['test'] = None if x['test'] is None else x['test']['required']
        out['accept_temporary'] = x['accept_temporary']
        out['languages'] = None if x['languages'] == [] else x['languages'][0]['id'] + ' ' + x['languages'][0]['level']['id']
    
    
        return out

# создание и запись файла с обработанными вакансиями

l = []
for x in data:
    l.append(get_vacancy(x))
df = pd.DataFrame(l) 

fname = 'C:\\Users\\aab\\Documents\\Обучение\\vac' + d.strftime('%Y-%m-%d') + '.pkl'
with open(fname, 'wb') as f:
    pkl.dump(df, f)


# подготовка датафрейма для модели

df.drop_duplicates(subset = ['id'], inplace = True)
df.set_index('id', inplace = True)
df = df[(df['salary_currency'] == 'RUR') | (df['salary_currency'].isna())]
df = df[~df['name'].apply(lambda x: any(k in x for k in ['финансовый', 'системный', 'бизнес', '1c', '1с', 'system']))]

df['salary_from'] = np.where(df['salary_gross'], df['salary_from'] * 0.87, df['salary_from'])
df['salary_to'] = np.where(df['salary_gross'], df['salary_to'] * 0.87, df['salary_to'])

df['salary_from'].fillna(df['salary_to'], inplace = True)
df['salary_to'].fillna(df['salary_from'], inplace = True)
df['salary_mid'] = (df['salary_from'] + df['salary_to']) / 2

dx = df.replace(to_replace= '[^\\da-zA-Zа-яёА-ЯЁ\- ]', value = '', regex = True)
dx[['name', 'description']] = dx[['name', 'description']].apply(lambda x: x.astype(str).str.lower())

stop_words = ['по', 'с', 'и', 'в', 'г', 'со']
dx['name_2'] = dx['name'].apply(lambda x: [item for item in x.split() if item not in stop_words])

morph = pymorphy2.MorphAnalyzer()

def lemmatize(words):
    return ' '.join([morph.parse(w)[0].normal_form for w in words])

dx['name_3'] = dx['name_2'].apply(lambda x: [morph.parse(w)[0].normal_form for w in x])

dx['skill_list'] = dx['key_skills'].apply(lambda x: [val for dic in x for val in dic.values()])
dx['role'] = dx['professional_roles'].apply(lambda x: [val for dic in x for val in dic.values()][1])

dx.drop(['name', 'name_2', 'salary_from', 'salary_to', 'salary_currency', 'salary_gross',  'city', 'description', 'key_skills', 'professional_roles', 'employer', 'published_at'], axis = 1, inplace = True)
categorial_cols = ['area', 'experience', 'schedule', 'employment', 'accept_handicapped', 'has_test', 'test', 'accept_temporary', 'languages', 'role'] 
dx = pd.get_dummies(dx, columns = categorial_cols)

mlb = MultiLabelBinarizer()
dx = dx.join(pd.DataFrame(mlb.fit_transform(dx.pop('name_3')), columns = mlb.classes_, index = dx.index))
dx = dx.join(pd.DataFrame(mlb.fit_transform(dx.pop('skill_list')), columns = mlb.classes_, index = dx.index), rsuffix = '_2')

dx = dx[dx.columns[dx.sum() > 2]]
dx = dx * 1

# загрузка модели, приведение списка фич к списку, на котором обучалась модель

xr = pkl.load(open('C:\\Users\\aab\\Documents\\aml_model.sav', 'rb'))
collist = pkl.load(open('C:\\Users\\aab\\Documents\\Model_features.dat', 'rb'))

train_cols = collist
new_cols = dx.columns
common_cols = collist.intersection(new_cols)
train_not_new = collist.difference(new_cols)

dx = dx[common_cols]

for col in train_not_new:
    dx[col] = 0
    
train_cols = collist
new_cols = dx.columns
common_cols = collist.intersection(new_cols)
train_not_new = collist.difference(new_cols)

dx = dx[common_cols]

for col in train_not_new:
    dx[col] = 0   

# предсказание и восстановление значения зарплаты

result = xr.predict(dx)

with open('C:\\Users\\aab\\Documents\\Salary_bounds.dat', 'rb') as fp:
    salary_min, salary_max = pkl.load(fp)
    
df['salary_pred_norm'] = result.tolist()
df['salary'] = df['salary_pred_norm'].apply(lambda x: 0 if x < 0 else round(x * (salary_max - salary_min) + salary_min), 0)

df['salary_diff'] = df['salary_mid'] - df['salary']

# запись отчета в PDF-файл

from fpdf import FPDF

class PdfReport(FPDF):
    def __init__(self, df):
        FPDF.__init__(self) 
        self.df = df

    def generate(self, text, name, area, employer, salary, description, experience, schedule):
        """Функция генерирует pdf-страницы из переданных ключевых признаков вакансий
        """	
        pdf.add_page()
        pdf.add_font('Calibri', '', 'C:\\Windows\\Fonts\\calibri.ttf', uni = True) 
        pdf.set_font('Calibri', '', 24)
        pdf.set_fill_color(r = 255, g = 240, b = 228)
        pdf.multi_cell(w = 0, h = 20, txt = text, border = 1, align = "L", fill = True)
        pdf.set_font('Calibri', '', 18)
        pdf.multi_cell(w = 0, h = 20, txt = name, border = 1)
        pdf.cell(w = 95, h = 20, txt = 'Город: ' + area, border = 1, ln = 0)
        pdf.cell(w = 95, h = 20, txt = 'Работодатель: ' + employer, border = 1, ln = 1)
        pdf.cell(w = 95, h = 20, txt = 'Опыт: ' + experience, border = 1, ln = 0)
        pdf.cell(w = 95, h = 20, txt = 'График: ' + schedule, border = 1, ln = 1) 
        pdf.cell(w = 0, h = 20, txt = 'Оклад: ' + str(round(salary / 1000, 1)) + ' тыс. руб.', border = 1, ln = 1)
        pdf.set_font('Calibri', '', 12)
        pdf.multi_cell(w = 0, h = 5, txt = description, border = 1, align = "L")
        

s = df.sort_values(by = ['salary_mid'], ascending = False).head()

pdf = PdfReport(s)
pdf.alias_nb_pages()

for ind in s.index:
    pdf.generate('Топ-5 самых высокооплачиваемых вакансий с явно указанной зарплатой', s['name'][ind], df['area'][ind], df['employer'][ind], df['salary_mid'][ind], df['description'][ind], df['experience'][ind], df['schedule'][ind])
    
s = df.sort_values(by = ['salary'], ascending = False).head()    

for ind in s.index:
    pdf.generate('Топ-5 самых высокооплачиваемых вакансий с предсказанной зарплатой', s['name'][ind], df['area'][ind], df['employer'][ind], df['salary'][ind], df['description'][ind], df['experience'][ind], df['schedule'][ind])
    
s = df.sort_values(by = ['salary_diff'], ascending = False).head()    

for ind in s.index:
    pdf.generate('Топ-5 вакансий с самой "завышенной" зарплатой', s['name'][ind], df['area'][ind], df['employer'][ind], df['salary_mid'][ind], df['description'][ind], df['experience'][ind], df['schedule'][ind])
      
pdf.output('C:\\Users\\aab\\Documents\\PDF_TEST.pdf','F')

