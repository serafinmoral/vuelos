import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

import os

def createdatasete(i):
    namen = 'N'+vars[i]


def createdataset(i):
    
    name = vars[i]
    namen = 'N'+vars[i]
    fdataset = total.copy()

    col = total2[namen]
    fdataset= fdataset.join(col)

    return fdataset

def estimatemodel2(i,data):
    
    var = data.iloc[: , -1]


    if var.name[1:] in acciones:
        print("accion")
        y = data.iloc[:,i:i+1]
    
    

        print(i,vars[i])
        print(y)
        if typevar[i] == 0:
            print(var.name,"discrete")

            clf = LogisticRegression(penalty='l1',solver='liblinear').fit(y, var)
            models2.append(clf)
            td2.append(1.0)
        else:
            print(var.name,"continuo")
            lr = LinearRegression().fit(y,var)
            models2.append(lr)
            et = lr.predict(y)
            sd = mean_squared_error(var,et)
            td2.append(sd)

    else:
        models2.append(models[i])

    return 


def estimatemodel(i,data):
    var = data.iloc[: , -1]
    y = data.iloc[:,:-1]

    
    print(i,vars[i])
    if var.dtypes == 'int64':
        typevar.append(0)
        print(var.name,"discrete")

        clf = LogisticRegression(penalty='l1',solver='liblinear').fit(y, var)
        print(clf.intercept_, clf.coef_)
        models.append(clf)
        td.append(1.0)
    else:
        print(var.name,"continuo")
        typevar.append(1)

        lr = LinearRegression().fit(y,var)
        print(lr.intercept_, lr.coef_)
        models.append(lr)
        et = lr.predict(y)
        sd = mean_squared_error(var,et)
        td.append(sd)

    return 


def evaluavuelo(vuelo, p1 = 0.99, p2=0.98,t = 0.5):
    df = pd.read_csv(vuelo)
    df['angdifSCl'] = df['angdifSCl'].astype(float)
    res = []
    res2 = []
    ndf = df.drop('Unnamed: 0', axis=1)
    g = 1.0
    b = 0.0
    res.append(g)
   
    ndf1 = ndf[:-1]
    ndf2 = ndf[1:].copy()
    ndf1.reset_index(drop=True, inplace=True)
    ndf2.reset_index(drop=True, inplace=True)
    for i in range(nvar):
        ndf2.rename(columns={vars[i]:'N'+vars[i]}, inplace=True)

    
    for j in range(len( ndf1)):
        g = g*p1 + b *(1-p2)
        b = g*(1-p1) + b*p2
        x = ndf1.iloc[[j]]
        y = ndf2.iloc[[j]]
       
        for i in range(nvar):
            v = vars[i]
            tg = 1
            tb = 1
            if v in acciones:
                x2 = x.iloc[:,i:i+1]

                model1 = models[i]
                model2 = models2[i]
                if typevar[i] == 0:
                    pg = model1.predict_proba(x)
                    pb = model2.predict_proba(x2)
                   
                    val = y['N'+vars[i]][j]
                    i1 = list(model1.classes_).index(val)
                    i2 = list(model2.classes_).index(val)
                    tg *= pg[0][i1]
                    tb *= pb[0][i2]

                    if ((pg[0][i1]/pb[0][i2])<t):
                        gw.write(vuelo+ " , " + str(j) +" , " + v + " , d , " + str(pg[0][i1])+ " , " +  str(pb[0][i2]) + "\n")

                elif typevar[i] == 1:
                    pg = model1.predict(x) 
                    pb = model2.predict(x2) 
                    
                    val = y['N'+vars[i]][j]

                    pgn = (val-pg[0])/td[i]
                    pbn = (val-pb[0])/td2[i]


                    
                    tg *= norm.pdf(pgn)
                    tb *= norm.pdf(pbn)


                    if ((norm.pdf(pgn)/norm.pdf(pbn))<t):
                        gw.write(vuelo + " , " + str(j) +" , c , " + v + " , " + str(norm.pdf(pgn)) + " , " +  str(norm.pdf(pbn)) + "\n" )


        g*= tg
        b*=tb
        s = g+b
        g = g/s
        b = b/s
        print(g,b)
        res.append(g)
        res2.append(math.log(tg/tb))

    return res, res2
            
                
        

        



    

os.chdir("/home/smc/programas/Carlos/nuevosdatos")

f = open("vuelos",'r') 

gw = open("criticos",'w') 

gw.write("flight, time , var, type, p goo, p bad \n")

vuelos = []
acciones = []
vueloseval = []
typevar = []

vn = []

for x in f:
    x = x.strip()
    vn.append(x)
    df = pd.read_csv(x)
    df['angdifSCl'] = df['angdifSCl'].astype(float)
    
    
    vuelos.append(df)

f.close()

f = open("vueloseval",'r') 
ve = []
vueloseval = []

for x in f:
    x = x.strip()
    ve.append(x)
    df = pd.read_csv(x)
    df['angdifSCl'] = df['angdifSCl'].astype(float)
    
    
    vueloseval.append(df)
f.close()


f = open("acciones",'r') 

for x in f:
    x = x.strip()
    acciones.append(x)
    
    

f.close()

nvuel = len(vuelos)

print (acciones)

listan= []
listan2 = []



for i in range(nvuel):
    nuevodf =  vuelos[i].drop('Unnamed: 0', axis=1)
    
    nuevodf1 = nuevodf[:-1] 
    nuevodf2 = nuevodf[1:]
    nuevodf2.reset_index(drop=True, inplace=True)
    listan.append(nuevodf1)
    listan2.append(nuevodf2)

total = pd.concat(listan)
total2 = pd.concat(listan2)

total.reset_index(drop=True, inplace=True)
total2.reset_index(drop=True, inplace=True)

vars = list(total2.columns)
print(vars)
nvar = len(vars)


models = []
models2 = []
td = []
td2 = []


for i in range(nvar):
    total2.rename(columns={vars[i]:'N'+vars[i]}, inplace=True)
    data = createdataset(i)
    estimatemodel(i,data)
    estimatemodel2(i,data)



f.close()
plt.figure()
i = 1
for v in ve:
    plt.figure(i)
    i+= 1

    res, res2 = evaluavuelo(v)
    plt.plot(res)
    plt.title(v)
    plt.ylim(0.0,1.0)
    
    plt.plot([0,len(res)-1],[0.2,0.2],'--r')
    plt.plot([0,len(res)-1],[0.5,0.5],':g')
    plt.figure(i)
    i+= 1

    plt.plot(res2)
    plt.title(v)
    plt.ylim(-2,2)
    
    plt.plot([0,len(res)-1],[0.0,0.0],'--r')

    
plt.show()

gw.close()

