# -*- coding: utf-8 -*-
import argparse
import pyodbc
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import uuid
import hashlib
import keyring
import urllib
import lightgbm as lgb
import re
import pickle
import json
import joblib


from datetime import datetime, timezone,timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from pylab import rcParams
from sqlalchemy import create_engine

# for debug
import logging

logging.basicConfig()

class Predict:

    def __init__(self, predict_table):
        self.__server = 'totorboat.database.windows.net'
        self.__database = 'boat'
        self.__username = 'takekazu.omi'
        self.__password = keyring.get_password('sqldatabase', 'takekazu.omi@totorboat.database.windows.net')
        self.__predict_table = predict_table
        self.__futures = [
            'bt2ritu',
            'class',
            'k1seiseki1',
            'k1seiseki2',
            'k2seiseki1',
            'k2seiseki2',
            'k3seiseki1',
            'k3seiseki2',
            'k4seiseki1',
            'k4seiseki2',
            'k5seiseki1',
            'k5seiseki2',
        #    'k6seiseki1',
        #    'k6seiseki2',
            'mt2ritu',
            't2ritu',
            'taiju',
            'tsyoritu',
            'z2ritu',
            'zsyoritu',
            'teiban',
        #    'nami',
        #    'tenji',
        #    'tenko',
            'syoritu',
            'fukusyoritu',
            'c1chaku',
            'c2chaku',
            'csyusou',
        #    'cyusyutu',
        #    'cyusyo',
            'stave'
        ]

        self.__target =['chaku',]
        self.__race_key = ['kaiymd', 'kaijcd', 'race']
        self.__columns=','.join(self.__race_key+self.__futures+self.__target)

        self.__seiseki_index =  pd.Index(['', '1', '2', '3', '4', '5', '6', 'S', 'K', 'F', 'L'])
        self.__class_index =  pd.Index(['A1', 'A2', 'B1', 'B2'])
        self.__tenko_index =  pd.Index(['晴', '曇り', '雨', '雪', '霧'])

        self._groupid = uuid.uuid1()

        self.SqlCache = False

    @property
    def groupid(self):
        return self._groupid

    def getConnection(self):
        params = urllib.parse.quote_plus('DRIVER={SQL Server Native Client 11.0};SERVER='+self.__server+';DATABASE='+self.__database+';UID='+self.__username+';PWD='+ self.__password)
        engine =  ("mssql+pyodbc:///?odbc_connect=%s" % params)
        return engine

    def read_sql(self, sql, name):
        """
        SQL Databaseから読み込んで、ローカルにpickle化してキャッシュする。
        同じSQLの場合は、キャッシュから読む。
        キャッシュのキーは、name+md5sum

        Parameters
        --------------
        sql : str
            実行するSQL文
        name : str
            データの名前

        Returns
        -------------
        df : DataFrame
            SQL文の実行結果
        """

        hexdigest = hashlib.md5(sql.encode('utf-8')).hexdigest()
        path = 'data/'+name+'_'+hexdigest+'.pickle'
        if self.SqlCache and os.path.isfile(path):
            df = pd.read_pickle(path)
        else:
            df = pd.read_sql(sql=sql, con=self.getConnection())
            if self.SqlCache:
                df.to_pickle(path)
        return df

    def write_predict(self, df, modelid):
        """
        SQL Databaseに書く

        Parameters
        --------------
        df : dataframe
            書き込むデータ

        name : str
            データのid

        Returns
        -------------
            groupid: uuid
        """

        now = datetime.utcnow()
        d = df.copy()
        d['groupid'] = self._groupid
        d['cdate'] = now
        d['modelid'] = modelid
        d.to_sql(
            self.__predict_table,
            con = self.getConnection(), if_exists='append', index=False, chunksize=1024)
#        print("groupid:     "+ str(id))
#        print("modelid:     "+ str(modelid))
        return id

    def fix_features(self, df):
        df_t = df.copy()

        for column in self.__futures:

            if column == 'tenko' :
                labels = self.__tenko_index.get_indexer(df_t[column])
            elif column == 'class':
                labels = self.__class_index.get_indexer(df_t[column])
            elif re.match(r'k[1-6]seiseki[1-2]', column):
                labels = self.__seiseki_index.get_indexer(df_t[column])
            else:
                continue

            df_t[column] = labels

        # int に調整
        df_t['teiban'] = df_t['teiban'].astype(int)

        return df_t

    def get_data_from_finished(self, training_data_range, kaijcd):
        # 学習データの元になるターゲットと特徴量を読む。

        sql = f'''
        select {self.__columns} from features3
            where kaiymd >= '{training_data_range[0]}'
            and kaiymd < '{training_data_range[1]}'
            and kaijcd in ({','.join(list(map(str, kaijcd)))})
            and haskekka = 1 and chaku is not null'''

        kyotei = self.read_sql(sql, 'kyotei')

        return kyotei

    def get_data_from_unfinished(self):
        # 結果の出てないデータを読む。取り消しレースも含む
        y = (datetime.utcnow() + timedelta(hours=9-24)).strftime('%Y-%m-%d')

        sql = f'''
        select {self.__columns} from features3
            where kaiymd >= '{y}'
            and haskekka = 0'''

        kyotei = self.read_sql(sql, 'kyotei_'+y)

        return kyotei

    def write_predict_result(self, df, pred, modelid):
        # 予想結果をdbに書く
        df_db = df[['kaiymd', 'kaijcd', 'race', 'teiban']].copy()
        df_db['chaku'] = pred
        self.write_predict(df_db, modelid)

        return df_db

    def splitXy(self, df):
        X = df[self.__futures].copy()
        y = df[self.__target]
        return X, y

    def get_futures(self, df):
        X = df[self.__futures].copy()
        return X

    def train(self, df):
        X, y  = self.splitXy(df)

        # 訓練データとテストデータに分割する
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        # 別途テストデータを分けているので、ここで分けるのは止める 12/17
        X_train, X_test, y_train, y_test = X,X,y,y

        # 上記のパラメータでモデルを学習する
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            max_bin=1000,
            silent=True)
        #model = lgb.LGBMRegressor(silent=True)

        model.fit(X_train, y_train, categorical_feature=[
            'teiban',
            'class',
            'k1seiseki1',
            'k1seiseki2',
            'k2seiseki1',
            'k2seiseki2',
            'k3seiseki1',
            'k3seiseki2',
            'k4seiseki1',
            'k4seiseki2',
            'k5seiseki1',
            'k5seiseki2',
        #    'k6seiseki1',
        #    'k6seiseki2'
        ])

        # see: 3.4. Model persistence
        # https://scikit-learn.org/stable/modules/model_persistence.html

        id = uuid.uuid1()
#        filename = 'model/model-'+str(id)+'.pickle'
        filename = 'model/model-'+str(id)+'.joblib'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        joblib.dump(model, filename, compress=3)
        #with open(filename, mode='wb') as f:
        #    pickle.dump(model, f)
        #

        return model, X_test, y_test, filename

    def predict(self, model, df):
        # データを予測する
        y_pred = model.predict(df)

        return y_pred

    def rmse(self, y_test, y_pred):
        # RMSE を計算する
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        return rmse

    def get_model(self, file):
        _, ext = os.path.splitext(file)
        if ext == '.json' :
            with open(file, encoding='utf-8') as f:
                data = json.load(f)

            for d in data:
                k = d['KaijCd']
                m = joblib.load(d['ModelPath'])
                mp = d['ModelPath']
                yield {'kaijcd':k, 'model':m, 'model_path':mp}

        elif ext == '.joblib' :
            yield {'kaijcd':list(range(1,25)), 'model':joblib.load(file), 'model_path':file}
        else :
            with open(file, mode='rb') as f:
                yield {'kaijcd':list(range(1,25)), 'model':pickle.load(f), 'model_path':file}

    def get_modelid(self, model_path):
        m=list(filter(None, re.split(r'model/|model-|\.pickle|\.joblib', model_path)))
        return m[0]

    def plot_feature_importances(self, df, model, modelid):
        n_features = df.shape[1]
        plt.figure(figsize=(10,10))
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), df.columns)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        png = "model/feature-importances-"+modelid+".png"
        plt.savefig(png)
        plt.close('all')
        return png

def get_args():
    parser = argparse.ArgumentParser(description='Process predict.')

    parser.add_argument('-j', '--json', help='output format json.', action='store_true', dest="json")
    parser.add_argument('-k', '--kaijcd', help='kaijo code. camma sep.', action='store', dest="kaijcd", default="all", type=str)
    parser.add_argument('-t', '--table', help='predict table name', action='store', dest="predict_table", default="predict", type=str)
    parser.add_argument('-m', '--model', help='make model', action='store_true')
    parser.add_argument('-v', '--verify', help='verify predict used by after Dec. 1', action='store', dest='predict11', type=str)
    parser.add_argument('-o', '--oracle', help='make oracle', action='store', dest='oracle', type=str)
    parser.add_argument('-w', '--writedb', help='write predict to database', action='store_true')

    args = parser.parse_args()

    return(args)

def validator(predict, kaijcd, models, date_range, writedb):

    # モデルがトレーニングされた会場に合わせて予想対象を切り替えて、モデル毎に予想する。
    for m in models:
        # コマンドラインで指定されていたら会場コードはそれを使う。それ以外は、モデルがトレーニングされたときのデータか、全会場を使う。
        # TODO 要改善
        kc = kaijcd if len(kaijcd) > 0 else m['kaijcd']

        # 期間内の結果があるレースデータを読む
        data = predict.get_data_from_finished(date_range, kc)

        # カテゴリの数値合わせなど特徴量の調整をする
        df = predict.fix_features(data)

        # 目的変数と説明変数に分ける
        X, y = predict.splitXy(df)

        # 予想する
        pred = predict.predict(m['model'], X)

        # 結果との差異を計算
        rmse = predict.rmse(y, pred)

        modelid = predict.get_modelid(m['model_path'])

        # writedbフラグを見て、DBに結果を書く
        if(writedb):
            predict.write_predict_result(data, pred, modelid)

        o = {'KaijCd':kc, 'StartDate':date_range[0],'EndDate':date_range[1],'ModelId':m['model_path'], 'GroupId':str(predict.groupid), 'RMSE':rmse}

        print("kaijcd:", str(kc), "modelid:", modelid, "RMSE:", str(rmse), file=sys.stderr,flush=True)
        yield o

def main():
    args = get_args()

    predict = Predict(predict_table=args.predict_table)

    # 対象の会場を取得
    if args.kaijcd == 'all' : # 全部
        kaijcd = list(range(1,25))
    elif args.kaijcd == 'model' :
        kaijcd = []
    else:
        kaijcd = list(map(int, args.kaijcd.split(',')))

    if(args.model):
        date_range = ['2011-11-01','2018-11-01']

        # 学習データからmodelを作る
        kyotei_data = predict.get_data_from_finished(date_range, kaijcd)

        df = predict.fix_features(kyotei_data)

        model, X, y, model_path = predict.train(df)

        # 自分自身と突き合わせる
        pred = predict.predict(model, X)
        rmse = predict.rmse(y, pred)

        # 特徴量の重みをplot
        png_path = predict.plot_feature_importances(X, model, predict.get_modelid(model_path))

        if args.json:
            o = {'KaijCd':kaijcd, 'StartDate':date_range[0],'EndDate':date_range[1],'ModelPath':model_path, 'RMSE':rmse, 'feature_importances_png':png_path}
            print(json.dumps(o, indent = 2))
        else:
            print("kaijcd:     " + str(kaijcd))
            print("date range: " + str(date_range))
            print("rmse:       " + str(rmse))
            print("model path: " + model_path)
            print("feature importances png: " + png_path)

    if(args.predict11):
        # 除外したデータを予想する
        # model はファイルから戻す

        date_range = ['2018-11-01','2030-01-01']

        model_path = args.predict11

        models = predict.get_model(model_path)

        o = list(validator(predict, kaijcd, models, date_range, args.writedb))
        if args.json:
            print(json.dumps(o, indent = 2))
        else:
            for i in o:
                print("kaijcd:     " + str(i['KaijCd']))
                print("date range: " + str([i['StartDate'],i['EndDate']]))
                print("rmse:       " + str(i['RMSE']))
                print("model path: " + i['ModelPath'])

    if(args.oracle):
        # 結果の出てないレースを予想する
        # model はファイルから戻す
        model_path = args.oracle

        model = predict.get_model(model_path)

        kyotei_data = predict.get_data_from_unfinished()

        df = predict.fix_features(kyotei_data)

        X = predict.get_futures(df)
        pred = predict.predict(model, X)

        print("model path: " + model_path)

        if(args.writedb):
            predict.write_predict_result(kyotei_data, pred, predict.get_modelid(model_path))


if __name__ == '__main__':
    main()
