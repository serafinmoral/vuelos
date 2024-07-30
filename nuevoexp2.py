import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

def createHTMLoutput(filename):
    from datetime import datetime
    now = datetime.now()
    
    Func = open(filename,"w")
   
    # Adding input data to the HTML file
    Func.write("<html>\n<head>\n\
            <title>Logistic Regression output</title>\n</head>\
            <body>\n<h1>Logistic Regression output</h1>\n\
            <h2>"+now.strftime("%d/%m/%Y %H:%M:%S")+"</h2>")
    
    # Saving the data into the HTML file
    Func.close()

def appendHTMLoutput(filename,HtmlString):
    Func = open(filename,"a")
    Func.write(HtmlString)
    Func.close()

def closeHTMLoutput(filename):
    Func = open(filename,"a")
    Func.write("</body></html>")
    Func.close()

def altCalcPlot1(dataset,mode):
    #this function contains an alternative calculation of logistic regression and plots in the HTML output.
    appendHTMLoutput("regr.html", "<br><br>Enter in altCalcPlot1 with "+str(list(dataset.columns))+":")
    #1 Getting the inputs and output
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    
    #2 Creating the Training Set and the Test Set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #3 Feature Scaling    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    
    #4 Building the model
    match mode:
        case "discrete":
            clf = LogisticRegression(penalty='l1',solver='liblinear')
        case "continuo":
            clf = LogisticRegression()
        case _:
            clf = LogisticRegression(penalty='l1',solver='liblinear')
     
    #5 Training the model
    clf.fit(X_train, y_train)

    #6 Inference: Making the predictions of the data points in the test set
    y_pred = clf.predict(sc.transform(X_test))



    plt.figure(1, figsize=(4, 3))
    plt.clf()
    #plt.scatter(X_train, y_train, color="black", marker='+')
    fign, ax = plt.subplots()
    plt.plot(X_test, y_pred, color='red', marker= '.', linestyle='None')
    
    
    import base64
    from io import BytesIO
    tmpfile = BytesIO()

    fign.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = '<br>altCalcPlot1: Generating plot '+str(dataset.iloc[:, -1].name)+ ' '+ mode +':<br>' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '\n'
    appendHTMLoutput("regr.html", html)

    #7 Calculate Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    conf_matrix = confusion_matrix(y_test, y_pred)
    appendHTMLoutput("regr.html","<br>Confusion Matrix:"+str(conf_matrix)+"\n")
    
    # ... and plot the Confusion Matrix:
    tmpfile2 = BytesIO()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    fig.savefig(tmpfile2, format='png')
    encoded = base64.b64encode(tmpfile2.getvalue()).decode('utf-8')
    html = '<br>Confusion Matrix:<br>' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '\n'
    appendHTMLoutput("regr.html", html)



def createdataset(i):
    
    name = vars[i]
    namen = 'N'+vars[i]
    fdataset = total.copy()

    col = total2[namen]
    fdataset= fdataset.join(col)

    return fdataset

def estimatemodel2(i,data):
    var = data.iloc[: , -1]
    appendHTMLoutput("regr.html","<h3>Enter in estimatemodel2."+str(i)+" var.name:"+var.name+"</h3>")
    if var.name[1:] in acciones:
        y = data.iloc[:,:-1]
    

        print(i,vars[i])
        if var.dtypes == 'int64':
            print(var.name,"discrete")

            clf = LogisticRegression(penalty='l1',solver='liblinear').fit(y, var)
            print("intercept & coef1:")
            print(clf.intercept_, clf.coef_)
            appendHTMLoutput("regr.html","<br>estimatemodel2."+str(i)+": entered in discrete")
            altCalcPlot1(data,"tbd")
        else:
            print(var.name,"continuo")
            lr = LinearRegression().fit(y,var)
            print("estimatemodel2  continuo: intercept shape:"+str(lr.intercept_.shape)+"coef shape:"+str(lr.coef_.shape))
            print(lr.intercept_, lr.coef_)
            appendHTMLoutput("regr.html","<br>estimatemodel2."+str(i)+": entered in continuo")
    else:
        models2.append(models[i])
        appendHTMLoutput("regr.html","<br>estimatemodel2."+str(i)+": var.name "+str(var.name)+" not in acciones")
    return 


def estimatemodel(i,data):
    var = data.iloc[: , -1]
    y = data.iloc[:,:-1]
    appendHTMLoutput("regr.html","<h3>Enter in estimatemodel."+str(i)+" var.name:"+var.name+"</h3>")
    print(i,vars[i])
    if var.dtypes == 'int64':
        mode = "discrete"
        print(var.name,mode)
        clf = LogisticRegression(penalty='l1',solver='liblinear').fit(y, var)
        print(clf.intercept_, clf.coef_)
        appendHTMLoutput("regr.html","<br>estimatemodel  discrete: intercept shape:"+str(clf.intercept_.shape)+"coef shape:"+str(clf.coef_.shape))
        appendHTMLoutput("regr.html","<br>estimatemodel."+str(i)+": entered in discrete")
        models.append(clf)
        altCalcPlot1(data,mode)
    else:
        mode = "continuo"
        print(var.name,mode)
        lr = LinearRegression().fit(y,var)
        appendHTMLoutput("regr.html","<br>estimatemodel  continuo: intercept shape:"+str(lr.intercept_.shape)+"coef shape:"+str(lr.coef_.shape))
        print(lr.intercept_, lr.coef_)
        appendHTMLoutput("regr.html","<br>estimatemodel."+str(i)+": entered in continuo")
        models.append(lr)
        #altCalcPlot1(data,mode)
    print("var.ravel():",var.ravel(), " y:", y)
    return 


"""
FIN DE LA DECLARACIÓN DE FUNCIONES
"""
os.chdir("/home/smc/programas/Carlos/nuevosdatos")
#os.chdir("C:/Users/Carlos/Documents/2023PyExp")


createHTMLoutput("regr.html")
vuelos = []
f = open("vuelos",'r') #text file with the names of the files
for x in f:
    x = x.strip()
    df = pd.read_csv(x)
    df['PBNdeviation'] = df['PBNdeviation'].astype(float)
    df['bank'] = df['bank'].astype(float)
    df['angdifHCl'] = df['angdifHCl'].astype(float)
    df['angdifSCl'] = df['angdifSCl'].astype(float)
    
    vuelos.append(df)
f.close()


appendHTMLoutput("regr.html", "<h4>vuelos</h4>len(vuelos):"+str(len(vuelos)))
#appendHTMLoutput("regr.html","vuelos:"+str(vuelos))
appendHTMLoutput("regr.html", "<br>vuelos[0]:"+str(vuelos[0].shape))
appendHTMLoutput("regr.html", "<br>vuelos is a list. Each element is a matrix of all the variables of one flight")


acciones = []
f = open("acciones",'r') #text file with the names of the pilot action columns
for x in f:
    x = x.strip() #Remove spaces at the beginning and at the end of the string
    acciones.append(x)
    #vuelos.append(df)
f.close()

appendHTMLoutput("regr.html", "<h4>acciones</h4>len(acciones):"+str(len(acciones)))
appendHTMLoutput("regr.html","<br>acciones:"+str(acciones))
appendHTMLoutput("regr.html", "<br>acciones is a list of the variable names of pilot actions.")


nvuel = len(vuelos)
listan= []
listan2 = []


for i in range(nvuel):
    nuevodf =  vuelos[i].drop('Unnamed: 0', axis=1)
    nuevodf1 = nuevodf[:-1] #slice the df to omit the last row
    nuevodf2 = nuevodf[1:] #slice the df to omit the 1st row
    nuevodf2.reset_index(drop=True, inplace=True)
    listan.append(nuevodf1)
    listan2.append(nuevodf2)

appendHTMLoutput("regr.html", "<h3>listan and listan2</h3>")
appendHTMLoutput("regr.html", "len(listan):"+str(len(listan)))
appendHTMLoutput("regr.html", "<br>len(listan2):"+str(len(listan2)))
appendHTMLoutput("regr.html", "<br>both dataframes contain the 22 flights<br>")

appendHTMLoutput("regr.html", "<br>listan[0].shape:"+str(listan[0].shape))
appendHTMLoutput("regr.html", "<br>listan2[0].shape:"+str(listan2[0].shape))
appendHTMLoutput("regr.html", "<br>both dataframes have the same name of rows and columns, df by df<br>")

appendHTMLoutput("regr.html","<br>listan[0].iloc[1]:"+str(listan[0].iloc[1])) 
appendHTMLoutput("regr.html","<br>listan2[0].iloc[0]:"+str(listan2[0].iloc[0])) 
appendHTMLoutput("regr.html", "<br>listan2 is  one row ahead of listan<br>")
 

total = pd.concat(listan)
total2 = pd.concat(listan2)

total.reset_index(drop=True, inplace=True)
total2.reset_index(drop=True, inplace=True)

appendHTMLoutput("regr.html", "<h3>total and total2</h3>")
appendHTMLoutput("regr.html", "total = pd.concat(listan) // total2 = pd.concat(listan2)")
appendHTMLoutput("regr.html", "<br>len(total):"+str(len(total)))
appendHTMLoutput("regr.html", "<br>len(total2):"+str(len(total2)))
appendHTMLoutput("regr.html", "<br>Unified dataframes contain the 22 flights<br>")

appendHTMLoutput("regr.html", "<br>total.shape:"+str(total.shape))
appendHTMLoutput("regr.html", "<br>total2.shape:"+str(total2.shape))
appendHTMLoutput("regr.html", "<br>Both dataframes have the same number of rows and columns as 'df', but they are shifted 1 time slice<br>")

vars = list(total2.columns)

appendHTMLoutput("regr.html", "<br>vars:"+str(vars))

#print(len(total),len(total2))

nvar = len(vars)
models = []
models2 = []
for i in range(nvar):
    total2.rename(columns={vars[i]:'N'+vars[i]}, inplace=True)
    list(total2.columns)
    data = createdataset(i)
    appendHTMLoutput("regr.html", "<h1>Created dataset data = createdataset("+str(i)+"):</h1>")
    appendHTMLoutput("regr.html", "<br>__total2 columns renamed: list(total2.columns)"+str(list(total2.columns)))
    appendHTMLoutput("regr.html", "<br>____data = createdataset(i) called:<br>_____data.shape:"+str(list(data.shape))+"<br>_____list(data.columns):"+str(list(data.columns)))
    estimatemodel(i,data)
    estimatemodel2(i,data)


appendHTMLoutput("regr.html", "<h1>Final results</h1>")
appendHTMLoutput("regr.html", "<br>len(models):"+str(len(models))+"\n")
appendHTMLoutput("regr.html", "<br>len(models2):"+str(len(models2))+"\n")
print("models:",len(models))
print("models2:",len(models2))

closeHTMLoutput("regr.html")