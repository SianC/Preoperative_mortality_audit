from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import Label
import tkinter.font as font
from tkinter import filedialog
import tkinter.messagebox as msgBox
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from tkinter.filedialog import asksaveasfile, asksaveasfilename

my_ref={} # to store references to checkboxes 
i=1
p=2
column_search=[] # To store the checkbuttons which are checked


def sp_test(df, bias_column_name, results_column_name):
    boot_length = 100
    boot_results = []
    df_boot = pd.DataFrame()
    options = list(set(list(df[bias_column_name]))) 
    for variable in options:
        for i in range(0, boot_length):
            df_small = df[df[bias_column_name]==variable]
            test_df = df_small.sample(frac=1.0, replace=True, random_state = i)
            boot_sp_results = sum(test_df[results_column_name].astype('int'))/len(test_df)
            boot_results.append(boot_sp_results)
        df_boot[variable] = boot_results
        boot_results = []
    
    return df_boot

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(0,len(y_hat)):
        if y_actual[i]==1 and y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==0 and y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

# def open_file():
#     global df
#     filename = filedialog.askopenfilename(initialdir=os.getcwd(), filetypes=[("CSV files", "*.csv")])
#     if(filename!=''):
#         df = pd.read_csv(filename, encoding = "ISO-8859-1", low_memory=False)
#         mR,mC=df.shape
#         cols = df.columns
#         num=5
#         pd.options.display.float_format = '{:.2f}'.format
#         msg=str(df.iloc[:num,:]) + '\n' + '...\n' + \
#         df.iloc[-num:,:].to_string(header=False) + '\n\n' + \
#         str(df.describe())
#         msgBox.showinfo(title="Data", message=msg)
    

def select_file():
    global df
    f_types=[('CSV files','*.csv'),('Excel files','*.xlsx')]
    file_path=filedialog.askopenfilename(filetypes=f_types)
    if 'sp_single' in globals():
        sp_single.config(text="")
    if 'eo1_single' in globals():
        eo1_single.config(text="")
    if 'eo2_single' in globals():
        eo2_single.config(text="")
    if 'te_single' in globals():
        te_single.config(text="")
    if 'sp_names' in globals():
        sp_names.config(text="")
    if 'eo_names' in globals():
        eo_names.config(text="")
    if 'te_names' in globals():
        te_names.config(text="")
    if file_path:
        lb1.config(text=file_path) # show the file path in Label
        global l1,i,my_ref
        if file_path.endswith('.csv'): # type of file extension 
            df=pd.read_csv(file_path) # create dataframe
        else:
            df=pd.read_excel(file_path)
        l1=list(df) # List of column names as header 
        my_columns() # call to show the checkbuttons



def my_columns():
    global l1,i,my_ref,p, clicked1, clicked2, clicked3, clicked4
    i=1 # to increase the column number 
    col = []
    my_ref={} # to store references to checkboxes 
    lb.config(text=' ') # Remove previosly displayed columns
    print(l1)
    
    for k in l1: # Loop through all headers 
        col.append(k)
        my_ref[k]=BooleanVar() # variable 
        #dd = Checkbutton(tab1, text=k,onvalue=True,offvalue=False,
            #command=lambda: my_update(),variable=my_ref[k])
        #dd.grid(row=p,column=i,padx=2,pady=25)
        #if i < 5:
         #   i=i+1 # increase the column value 
        #else:
         #   i=1
          #  p=p+1
    clicked1 = StringVar()
    clicked1.set(col[0])
    drop1 = OptionMenu(tab1, clicked1, *l1)
    drop1.config(font=f_small)
    drop1.grid(row=3, column=2)
    clicked2 = StringVar()
    clicked2.set(col[0])
    drop2 = OptionMenu(tab1, clicked2, *l1)
    drop2.config(font=f_small)
    drop2.grid(row=4, column=2)
    clicked3 = StringVar()
    clicked3.set(col[0])
    drop3 = OptionMenu(tab1, clicked3, *l1)
    drop3.config(font=f_small)
    drop3.grid(row=5, column=2)
    clicked4 = StringVar()
    clicked4.set(col[0])
    drop4 = OptionMenu(tab1, clicked4, *l1)
    drop4.config(font=f_small)
    drop4.grid(row=6, column=2)

def do_reset():
    if 'sp_single' in globals():
        sp_single.config(text="")
    if 'eo1_single' in globals():
        eo1_single.config(text="")
    if 'eo2_single' in globals():
        eo2_single.config(text="")
    if 'te_single' in globals():
        te_single.config(text="")
    if 'sp_names' in globals():
        sp_names.config(text="")
    if 'eo_names' in globals():
        eo_names.config(text="")
    if 'te_names' in globals():
        te_names.config(text="")
    if 'canvas2' in globals():
        canvas2.delete("all")
    if 'canvas31' in globals():
        canvas31.delete("all")
    if 'canvas32' in globals():
        canvas32.delete("all")
    if 'canvas4' in globals():
        canvas4.delete("all")
    if 'canvas5' in globals():
        canvas5.delete("all")
    if 'canvas61' in globals():
        canvas62.delete("all")
    if 'canvas7' in globals():
        canvas7.delete("all")
    


def save_file(df):
    try:
    # with block automatically closes file
        with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
            df.to_csv(file.name)
    except AttributeError:
    # if user cancels save, filedialog returns None rather than a file object, and the 'with' will raise an error
        print("The user cancelled save")

def save_image(figure):
    filename = asksaveasfilename(initialfile = 'Untitled.png',defaultextension=".png",filetypes=[("All Files","*.*"),("Portable Graphics Format","*.png")])
    figure.savefig(filename)

def calculate_sp():
    global df, clicked1, clicked2, clicked3, df_result_sp, sp_single, sp_names, canvas2
    #print(clicked1)
    
    sp_names = ttk.Label(tab2, text="Bias Column = "+str(clicked1.get())+", Results Column = "+str(clicked2.get())+" and Ground Truth column = "+str(clicked3.get()), font=f_small)
    sp_names.grid(column=2, row=1, sticky=N)
    
    bias_c = list(df[clicked1.get()])
    results_c = list(df[clicked2.get()])
    gt_c = list(df[clicked3.get()])
    
    sp_df = pd.DataFrame()
    sp_df["bias"] = bias_c
    sp_df["results"] = results_c
    sp_df["gt"] = gt_c
    
    df_boot = pd.DataFrame()
    options = list(set(list(sp_df["bias"]))) 
    for variable in options:
        #print(variable)
        
        #for i in range(0, boot_length):
        test_df = sp_df[sp_df["bias"]==variable]
        #print(len(test_df))
        #    test_df = df_small.sample(frac=1.0, replace=True, random_state = i)
        boot_sp_results = sum(test_df["results"].astype('int'))/len(test_df)
        #print(boot_sp_results)
        df_boot[variable] = [boot_sp_results]
        boot_results = []
        #print(df_boot) 


    df_result = pd.DataFrame()
    for i, col1 in enumerate(df_boot.columns):
        for j, col2 in enumerate(df_boot.columns):
            if i<j:
                new_column_name = str(col1)+"vs"+str(col2)
                df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
            df_result_sp_s = df_result
    
    if len(df_result_sp_s.columns) == 1:
        sp_single = ttk.Label(tab2, text="The Statistical parity score is "+str(round(df_result_sp_s.iloc[0,0], 3)), font=f)
        sp_single.grid(column=1, row=3, sticky=E)
            
    else:
        G1, G2, result = [], [], []
        for i, col1 in enumerate(df_boot.columns):
            for j, col2 in enumerate(df_boot.columns):
                #if i<j:
                G1.append(col1)
                G2.append(col2)
                result.append(float(abs(df_boot[col1] - df_boot[col2])))
                    #new_column_name = str(col1)+"vs"+str(col2)
                    #df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
        df_result_sp = pd.DataFrame()
        df_result_sp["G1"] = G1
        df_result_sp["G2"] = G2
        df_result_sp["Result"] = result
        
        data_wn = df_result_sp.pivot(index="G1", columns="G2", values="Result")
        columns = data_wn.columns
        data = np.tril(data_wn)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        sns.set_style("white")
        fig = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, cmap='Greens', vmin=0, vmax=1)#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Statistical Parity Difference Results')
        ax.set_xlabel('Protected Characteristic Subgroups')
        ax.set_ylabel('Protected Characteristic Subgroups')
        
        canvas2 = FigureCanvasTkAgg(fig, master=tab2)  # A tk.DrawingArea.
        canvas2.draw()
        canvas2.get_tk_widget().grid(column=2, row=3, sticky=S)
        
        image_save_button_sp = Button( tab2, text='Save Graph as Image', command=lambda: save_image(fig))
        image_save_button_sp.grid(column=1, row=3, sticky=N)
        
    
    save_button_sp = Button(tab2, text='Save Results', command=lambda: save_file(df_result_sp_s))
    save_button_sp.grid(column=1, row=2, sticky=N)
    #     f = plt.bar(x=range(0,len(col1)), list(col1))
    #     canvas = FigureCanvasTkAgg(f)
    #     canvas.show()
    #     #canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    #     toolbar = NavigationToolbar2TkAgg(canvas)
    #     toolbar.update()
    #     #canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#df_result_sp.iloc[0,0]
def calculate_eo():
    global df, clicked1, clicked2, clicked3, df_result_eo, eo2_single, eo1_single, eo_names, canvas31, canvas32
    #print(clicked1)
    eo_names = ttk.Label(tab3, text="Bias Column = "+str(clicked1.get())+", Results Column = "+str(clicked2.get())+" and Ground Truth column = "+str(clicked3.get()), font=f_small)
    eo_names.grid(column=2, row=1, sticky=N)
    
    bias_c = list(df[clicked1.get()])
    results_c = list(df[clicked2.get()])
    gt_c = list(df[clicked3.get()])
    
    eo_df = pd.DataFrame()
    eo_df["bias"] = bias_c
    eo_df["results"] = results_c
    eo_df["gt"] = gt_c
    
    df_boot_eo_1 = pd.DataFrame()
    df_boot_eo_2 = pd.DataFrame()
    options = list(set(list(eo_df["bias"]))) 
    for variable in options:
        #print(variable)
        
        #for i in range(0, boot_length):
        test_df = eo_df[eo_df["bias"]==variable]
        
        y_hat = list(test_df["results"])
        y_actual = list(test_df["gt"])
        #print(len(test_df))
        y_hat = list(test_df["results"])
        y_actual = list(test_df["gt"])
        #TP, FP, TN, FN = perf_measure(test_df["gt"], test_df["results"])
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(0,len(y_hat)):
            if y_actual[i]==1 and y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==0 and y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1
        #    test_df = df_small.sample(frac=1.0, replace=True, random_state = i)
        boot_eo_results_1 = FP/(FP+TN)
        boot_eo_results_2 = TP/(TP+FN)
        #boot_results.append(boot_te_results)
        df_boot_eo_1[variable] = [boot_eo_results_1]
        df_boot_eo_2[variable] = [boot_eo_results_2]
        #boot_results = []
    
    df_result_1 = pd.DataFrame()
    df_result_2 = pd.DataFrame()
    for i, col1 in enumerate(df_boot_eo_1.columns):
        for j, col2 in enumerate(df_boot_eo_1.columns):
            if i<j:
                new_column_name = str(col1)+"vs"+str(col2)
                df_result_1[new_column_name] = abs(df_boot_eo_1[col1] - df_boot_eo_1[col2])
                df_result_2[new_column_name] = abs(df_boot_eo_2[col1] - df_boot_eo_2[col2])
    df_result_eo_1_s = df_result_1
    df_result_eo_2_s = df_result_2
    
    if len(df_result_eo_1_s.columns) == 1:
        eo1_single = ttk.Label(tab3, text="The equalised odds score considering False Positives is "+str(round(df_result_eo_1_s.iloc[0,0],3)), font=f)
        eo1_single.grid(column=1, row=4, sticky=E)
        eo2_single = ttk.Label(tab3, text="The equalised odds score considering True Positives is "+str(round(df_result_eo_2_s.iloc[0,0],3)), font=f)
        eo2_single.grid(column=1, row=5, sticky=E)
    
    else:
        G1, G2, result = [], [], []
        G12, G22, result2 = [], [], []
        for i, col1 in enumerate(df_boot_eo_1.columns):
            for j, col2 in enumerate(df_boot_eo_1.columns):
                #if i<j:
                G1.append(col1)
                G2.append(col2)
                result.append(float(abs(df_boot_eo_1[col1] - df_boot_eo_1[col2])))
                
                G12.append(col1)
                G22.append(col2)
                result2.append(float(abs(df_boot_eo_2[col1] - df_boot_eo_2[col2])))
                    #new_column_name = str(col1)+"vs"+str(col2)
                    #df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
        df_result_eo_1 = pd.DataFrame()
        df_result_eo_1["G1"] = G1
        df_result_eo_1["G2"] = G2
        df_result_eo_1["Result"] = result
        
        df_result_eo_2 = pd.DataFrame()
        df_result_eo_2["G1"] = G12
        df_result_eo_2["G2"] = G22
        df_result_eo_2["Result"] = result2
        
        data = df_result_eo_1.pivot(index="G1", columns="G2", values="Result")
        columns = data.columns
        data = np.tril(data)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        sns.set_style("white")
        fig1 = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig1.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, cmap='Greens', vmin=0, vmax=1)#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Equalised Odds Difference Results, FPR')
        ax.set_xlabel('Protected Characteristic Subgroups')
        ax.set_ylabel('Protected Characteristic Subgroups')
        
        canvas31 = FigureCanvasTkAgg(fig1, master=tab3)  # A tk.DrawingArea.
        canvas31.draw()
        canvas31.get_tk_widget().grid(column=2, row=3, sticky=S)
        
        data = df_result_eo_2.pivot(index="G1", columns="G2", values="Result")
        columns = data.columns
        data = np.tril(data)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        sns.set_style("white")
        fig2 = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig2.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, cmap='Greens', vmin=0, vmax=1)#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Equalised Odds Difference Results, TPR')
        ax.set_xlabel('Protected Characteristic Subgroups')
        ax.set_ylabel('Protected Characteristic Subgroups')
        
        canvas32 = FigureCanvasTkAgg(fig2, master=tab3)  # A tk.DrawingArea.
        canvas32.draw()
        canvas32.get_tk_widget().grid(column=3, row=3, sticky=S)
        
        image_save_button_sp = Button( tab3, text='Save Graph 1 as Image', command=lambda: save_image(fig1))
        image_save_button_sp.grid(column=1, row=4, sticky=N)
        
        image_save_button_sp = Button( tab3, text='Save Graph 2 as Image', command=lambda: save_image(fig2))
        image_save_button_sp.grid(column=1, row=5, sticky=N)
    
    save_button_eo1 = Button(tab3, text='Save Results Part 1', command=lambda: save_file(df_result_eo_1_s))
    save_button_eo1.grid(column=1, row=2, sticky=N)
    
    save_button_eo2 = Button(tab3, text='Save Results Part 2', command=lambda: save_file(df_result_eo_2_s))
    save_button_eo2.grid(column=1, row=3, sticky=N)
    ## Do EO test here
    ## Output to screen results
    ## Save results
    
def calculate_te():
    global df, clicked1, clicked2, clicked3, df_result_te, te_single, te_names, canvas4
    #print(clicked1)
    
    te_names = ttk.Label(tab4, text="Bias Column = "+str(clicked1.get())+", Results Column = "+str(clicked2.get())+" and Ground Truth column = "+str(clicked3.get()), font=f_small)
    te_names.grid(column=2, row=1, sticky=N)
    
    bias_c = list(df[clicked1.get()])
    results_c = list(df[clicked2.get()])
    gt_c = list(df[clicked3.get()])
    
    te_df = pd.DataFrame()
    te_df["bias"] = bias_c
    te_df["results"] = results_c
    te_df["gt"] = gt_c
    
    df_boot_te = pd.DataFrame()
    options = list(set(list(te_df["bias"]))) 
    for variable in options:
        #print(variable)
        
        #for i in range(0, boot_length):
        test_df = te_df[te_df["bias"]==variable]
        
        y_hat = list(test_df["results"])
        y_actual = list(test_df["gt"])
        #TP, FP, TN, FN = perf_measure(test_df["gt"], test_df["results"])
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(0,len(y_hat)):
            if y_actual[i]==1 and y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==0 and y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1
        #print(len(test_df))
        #    test_df = df_small.sample(frac=1.0, replace=True, random_state = i)
        boot_te_results = FN/FP
        #boot_results.append(boot_te_results)
        df_boot_te[variable] = [boot_te_results]
        boot_results = []
    
    df_result = pd.DataFrame()
    for i, col1 in enumerate(df_boot_te.columns):
        for j, col2 in enumerate(df_boot_te.columns):
            if i<j:
                new_column_name = str(col1)+"vs"+str(col2)
                df_result[new_column_name] = abs(df_boot_te[col1] - df_boot_te[col2])
    df_result_te_s = df_result
    
    if len(df_result_te_s.columns) == 1:
        te_single = ttk.Label(tab4, text="The treatment equality score is "+str(round(df_result_te_s.iloc[0,0],3)), font=f)
        te_single.grid(column=1, row=3, sticky=E)
    
    else:
        G1, G2, result = [], [], []
        for i, col1 in enumerate(df_boot_te.columns):
            for j, col2 in enumerate(df_boot_te.columns):
                #if i<j:
                G1.append(col1)
                G2.append(col2)
                result.append(float(abs(df_boot_te[col1] - df_boot_te[col2])))
                    #new_column_name = str(col1)+"vs"+str(col2)
                    #df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
        df_result_te = pd.DataFrame()
        df_result_te["G1"] = G1
        df_result_te["G2"] = G2
        df_result_te["Result"] = result
        
        data = df_result_te.pivot(index="G1", columns="G2", values="Result")
        columns = data.columns
        data = np.tril(data)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        if np.nanmax(data) < 1:
            te_vmax = 1
        else:
            te_vmax = -((-np.nanmax(data)//5))
        
        sns.set_style("white")
        fig = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, vmax = te_vmax, cmap='Greens')#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Treatment Equality Difference Results')
        ax.set_xlabel('Protected Characteristic Subgroups')
        ax.set_ylabel('Protected Characteristic Subgroups')
        
        canvas4 = FigureCanvasTkAgg(fig, master=tab4)  # A tk.DrawingArea.
        canvas4.draw()
        canvas4.get_tk_widget().grid(column=2, row=3, sticky=S)
        
        image_save_button_sp = Button( tab4, text='Save Graph as Image', command=lambda: save_image(fig))
        image_save_button_sp.grid(column=1, row=3, sticky=N)
        
    save_button_sp = Button(tab4, text='Save Results', command=lambda: save_file(df_result_te_s))
    save_button_sp.grid(column=1, row=2, sticky=N)
        
column_search = [v for v in my_ref if my_ref[v].get()]

def calculate_int_sp():
    global df, clicked1, clicked2, clicked3, df_result_sp_int, sp_single, sp_names, canvas5
    #print(clicked1)
    
    #sp_names = ttk.Label(tab2, text="Bias Column = "+str(clicked1.get())+", Results Column = "+str(clicked2.get())+" and Ground Truth column = "+str(clicked3.get()), font=f_small)
    #sp_names.grid(column=2, row=1, sticky=W)
    
    bias_c = list(df[clicked1.get()])
    bias_2_c = list(df[clicked4.get()])
    results_c = list(df[clicked2.get()])
    gt_c = list(df[clicked3.get()])
    
    sp_df = pd.DataFrame()
    sp_df["bias"] = bias_c
    sp_df["bias_2"] = bias_2_c
    sp_df["results"] = results_c
    sp_df["gt"] = gt_c
    
    
    df_boot = pd.DataFrame()
    options_1 = list(set(list(sp_df["bias"]))) 
    options_2 = list(set(list(sp_df["bias_2"]))) 
    for variable_1 in options_1:
        for variable_2 in options_2:
            #print(variable)
            
            #for i in range(0, boot_length):
            test_half_df = sp_df[sp_df["bias"]==variable_1]
            test_df = test_half_df[test_half_df["bias_2"]==variable_2]
            #print(len(test_df))
            #    test_df = df_small.sample(frac=1.0, replace=True, random_state = i)
            boot_sp_results = sum(test_df["results"].astype('int'))/len(test_df)
            #print(boot_sp_results)
            df_boot[str(variable_1)+str(variable_2)] = [boot_sp_results]
            boot_results = []
            #print(df_boot) 


    df_result = pd.DataFrame()
    for i, col1 in enumerate(df_boot.columns):
        for j, col2 in enumerate(df_boot.columns):
            if i<j:
                new_column_name = str(col1)+"vs"+str(col2)
                df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
            df_result_sp_s = df_result
    
    if len(df_result_sp_s.columns) == 1:
        sp_single = ttk.Label(tab2, text="The Statistical parity score is "+str(round(df_result_sp_s.iloc[0,0], 3)), font=f)
        sp_single.grid(column=1, row=3, sticky=E)
            
    else:
        print("Reached here")
        G1, G2, result = [], [], []
        for i, col1 in enumerate(df_boot.columns):
            for j, col2 in enumerate(df_boot.columns):
                #if i<j:
                G1.append(col1)
                G2.append(col2)
                result.append(float(abs(df_boot[col1] - df_boot[col2])))
                    #new_column_name = str(col1)+"vs"+str(col2)
                    #df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
        df_result_sp_int = pd.DataFrame()
        df_result_sp_int["G1"] = G1
        df_result_sp_int["G2"] = G2
        df_result_sp_int["Result"] = result
        
        data_wn = df_result_sp_int.pivot(index="G1", columns="G2", values="Result")
        columns = data_wn.columns
        data = np.tril(data_wn)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        sns.set_style("white")
        fig = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, cmap='Greens', vmin=0, vmax=1)#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        #ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
        #ax.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        ax.set_title('Statistical Parity Difference Results')
        ax.set_xlabel('Protected Characteristics Subgroups')
        ax.set_ylabel('Protected Characteristics Subgroups')
        
        canvas5 = FigureCanvasTkAgg(fig, master=tab5)  # A tk.DrawingArea.
        canvas5.draw()
        canvas5.get_tk_widget().grid(column=2, row=3, sticky=S)
        
        image_save_button_sp = Button( tab5, text='Save Graph as Image', command=lambda: save_image(fig))
        image_save_button_sp.grid(column=1, row=3, sticky=N)
    
    save_button_sp = Button(tab5, text='Save Results', command=lambda: save_file(df_result_sp_s))
    save_button_sp.grid(column=1, row=2, sticky=N)
    #     f = plt.bar(x=range(0,len(col1)), list(col1))
    #     canvas = FigureCanvasTkAgg(f)
    #     canvas.show()
    #     #canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    #     toolbar = NavigationToolbar2TkAgg(canvas)
    #     toolbar.update()
    #     #canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def calculate_int_eo():
    global df, clicked1, clicked2, clicked3, df_result_int_eo, eo2_single, eo1_single, eo_names, canvas61, canvas62
    #print(clicked1)
    eo_names = ttk.Label(tab3, text="Bias Column = "+str(clicked1.get())+", Results Column = "+str(clicked2.get())+" and Ground Truth column = "+str(clicked3.get()), font=f_small)
    eo_names.grid(column=2, row=1, sticky=W)
    
    bias_c = list(df[clicked1.get()])
    results_c = list(df[clicked2.get()])
    gt_c = list(df[clicked3.get()])
    bias_2_c = list(df[clicked4.get()])
    
    eo_df = pd.DataFrame()
    eo_df["bias"] = bias_c
    eo_df["bias_2"] = bias_2_c
    eo_df["results"] = results_c
    eo_df["gt"] = gt_c
    
    df_boot_eo_1 = pd.DataFrame()
    df_boot_eo_2 = pd.DataFrame()
    options_1 = list(set(list(eo_df["bias"]))) 
    options_2 = list(set(list(eo_df["bias_2"])))
    for variable_1 in options_1:
        for variable_2 in options_2:
        #print(variable)
            
            #for i in range(0, boot_length):
            test_half_df = eo_df[eo_df["bias"]==variable_1]
            test_df = test_half_df[test_half_df["bias_2"]==variable_2]
            
            y_hat = list(test_df["results"])
            y_actual = list(test_df["gt"])
            #print(len(test_df))
            y_hat = list(test_df["results"])
            y_actual = list(test_df["gt"])
            #TP, FP, TN, FN = perf_measure(test_df["gt"], test_df["results"])
            TP = 0
            FP = 0
            TN = 0
            FN = 0
    
            for i in range(0,len(y_hat)):
                if y_actual[i]==1 and y_hat[i]==1:
                   TP += 1
                if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                   FP += 1
                if y_actual[i]==0 and y_hat[i]==0:
                   TN += 1
                if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                   FN += 1
            #    test_df = df_small.sample(frac=1.0, replace=True, random_state = i)
            boot_eo_results_1 = FP/(FP+TN)
            boot_eo_results_2 = TP/(TP+FN)
            #boot_results.append(boot_te_results)
            df_boot_eo_1[str(variable_1)+str(variable_2)] = [boot_eo_results_1]
            df_boot_eo_2[str(variable_1)+str(variable_2)] = [boot_eo_results_2]
            #boot_results = []
        
    df_result_1 = pd.DataFrame()
    df_result_2 = pd.DataFrame()
    for i, col1 in enumerate(df_boot_eo_1.columns):
        for j, col2 in enumerate(df_boot_eo_1.columns):
            if i<j:
                new_column_name = str(col1)+"vs"+str(col2)
                df_result_1[new_column_name] = abs(df_boot_eo_1[col1] - df_boot_eo_1[col2])
                df_result_2[new_column_name] = abs(df_boot_eo_2[col1] - df_boot_eo_2[col2])
    df_result_eo_1_s = df_result_1
    df_result_eo_2_s = df_result_2
    
    if len(df_result_eo_1_s.columns) == 1:
        eo1_single = ttk.Label(tab3, text="The equalised odds score considering False Positives is "+str(round(df_result_eo_1_s.iloc[0,0],3)), font=f)
        eo1_single.grid(column=1, row=2, sticky=E)
        eo2_single = ttk.Label(tab3, text="The equalised odds score considering True Positives is "+str(round(df_result_eo_2_s.iloc[0,0],3)), font=f)
        eo2_single.grid(column=1, row=3, sticky=E)
    
    else:
        G1, G2, result = [], [], []
        G12, G22, result2 = [], [], []
        for i, col1 in enumerate(df_boot_eo_1.columns):
            for j, col2 in enumerate(df_boot_eo_1.columns):
                #if i<j:
                G1.append(col1)
                G2.append(col2)
                result.append(float(abs(df_boot_eo_1[col1] - df_boot_eo_1[col2])))
                
                G12.append(col1)
                G22.append(col2)
                result2.append(float(abs(df_boot_eo_2[col1] - df_boot_eo_2[col2])))
                    #new_column_name = str(col1)+"vs"+str(col2)
                    #df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
        df_result_int_eo_1 = pd.DataFrame()
        df_result_int_eo_1["G1"] = G1
        df_result_int_eo_1["G2"] = G2
        df_result_int_eo_1["Result"] = result
        
        df_result_int_eo_2 = pd.DataFrame()
        df_result_int_eo_2["G1"] = G12
        df_result_int_eo_2["G2"] = G22
        df_result_int_eo_2["Result"] = result2
        
        data = df_result_int_eo_1.pivot(index="G1", columns="G2", values="Result")
        columns = data.columns
        data = np.tril(data)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        sns.set_style("white")
        fig1 = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig1.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, cmap='Greens', vmin=0, vmax=1)#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Equalised Odds Difference Results, FPR')
        ax.set_xlabel('Protected Characteristic Subgroups')
        ax.set_ylabel('Protected Characteristic Subgroups')
        
        canvas61 = FigureCanvasTkAgg(fig1, master=tab6)  # A tk.DrawingArea.
        canvas61.draw()
        canvas61.get_tk_widget().grid(column=2, row=3, sticky=S)
        
        data = df_result_int_eo_2.pivot(index="G1", columns="G2", values="Result")
        columns = data.columns
        data = np.tril(data)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        sns.set_style("white")
        fig2 = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig2.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, cmap='Greens', vmin=0, vmax=1)#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Equalised Odds Difference Results, TPR')
        ax.set_xlabel('Protected Characteristic Subgroups')
        ax.set_ylabel('Protected Characteristic Subgroups')
        
        canvas62 = FigureCanvasTkAgg(fig2, master=tab6)  # A tk.DrawingArea.
        canvas62.draw()
        canvas62.get_tk_widget().grid(column=3, row=3, sticky=S)
        
        image_save_button_sp = Button( tab6, text='Save Graph 1 as Image', command=lambda: save_image(fig1))
        image_save_button_sp.grid(column=1, row=4, sticky=N)
        
        image_save_button_sp = Button( tab6, text='Save Graph 2 as Image', command=lambda: save_image(fig2))
        image_save_button_sp.grid(column=1, row=5, sticky=N)
    
    save_button_eo1 = Button(tab6, text='Save Results Part 1', command=lambda: save_file(df_result_eo_1_s))
    save_button_eo1.grid(column=1, row=2, sticky=N)
    
    save_button_eo2 = Button(tab6, text='Save Results Part 2', command=lambda: save_file(df_result_eo_2_s))
    save_button_eo2.grid(column=1, row=3, sticky=N)


def calculate_int_te():
    global df, clicked1, clicked2, clicked3, df_result_te_int, te_single, te_names, canvas7
    #print(clicked1)
    
    te_names = ttk.Label(tab4, text="Bias Column = "+str(clicked1.get())+", Results Column = "+str(clicked2.get())+" and Ground Truth column = "+str(clicked3.get()), font=f_small)
    te_names.grid(column=2, row=1, sticky=W)
    
    bias_c = list(df[clicked1.get()])
    results_c = list(df[clicked2.get()])
    gt_c = list(df[clicked3.get()])
    bias_2_c = list(df[clicked4.get()])
    
    te_df = pd.DataFrame()
    te_df["bias"] = bias_c
    te_df["bias_2"] = bias_2_c
    te_df["results"] = results_c
    te_df["gt"] = gt_c
    
    df_boot_te = pd.DataFrame()
    options_1 = list(set(list(te_df["bias"])))
    options_2 = list(set(list(te_df["bias_2"]))) 
    for variable_1 in options_1:
        for variable_2 in options_2:
            #print(variable)
            
            #for i in range(0, boot_length):
            test_half_df = te_df[te_df["bias"]==variable_1]
            test_df = test_half_df[test_half_df["bias_2"]==variable_2]
            
            y_hat = list(test_df["results"])
            y_actual = list(test_df["gt"])
            #TP, FP, TN, FN = perf_measure(test_df["gt"], test_df["results"])
            TP = 0
            FP = 0
            TN = 0
            FN = 0
    
            for i in range(0,len(y_hat)):
                if y_actual[i]==1 and y_hat[i]==1:
                   TP += 1
                if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                   FP += 1
                if y_actual[i]==0 and y_hat[i]==0:
                   TN += 1
                if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                   FN += 1
            #print(len(test_df))
            #    test_df = df_small.sample(frac=1.0, replace=True, random_state = i)
            if FP == 0:
                boot_te_results = 100
            else:
                boot_te_results = FN/FP
            #boot_results.append(boot_te_results)
            df_boot_te[str(variable_1)+str(variable_2)] = [boot_te_results]
            boot_results = []
    
    df_result = pd.DataFrame()
    for i, col1 in enumerate(df_boot_te.columns):
        for j, col2 in enumerate(df_boot_te.columns):
            if i<j:
                new_column_name = str(col1)+"vs"+str(col2)
                #df_result[new_column_name] = abs(df_boot_te[col1] - df_boot_te[col2])
                df_result[new_column_name] = df_boot_te[col1] - df_boot_te[col2]
    df_result_te_s = df_result
    
    if len(df_result_te_s.columns) == 1:
        te_single = ttk.Label(tab4, text="The treatment equality score is "+str(round(df_result_te_s.iloc[0,0],3)), font=f)
        te_single.grid(column=1, row=3, sticky=E)
    
    else:
        G1, G2, result = [], [], []
        for i, col1 in enumerate(df_boot_te.columns):
            for j, col2 in enumerate(df_boot_te.columns):
                #if i<j:
                G1.append(col1)
                G2.append(col2)
                result.append(float(abs(df_boot_te[col1] - df_boot_te[col2])))
                    #new_column_name = str(col1)+"vs"+str(col2)
                    #df_result[new_column_name] = abs(df_boot[col1] - df_boot[col2])
        df_result_te_int = pd.DataFrame()
        df_result_te_int["G1"] = G1
        df_result_te_int["G2"] = G2
        df_result_te_int["Result"] = result
        
        data = df_result_te_int.pivot(index="G1", columns="G2", values="Result")
        columns = data.columns
        data = np.tril(data)
        #data = np.array([[i if i else np.nan for i in j] for j in data])
        mask = np.arange(data.shape[0])[:,None] <= np.arange(data.shape[1])
        data[mask]=np.nan
        
        if np.nanmax(data) < 1:
            te_vmax = 1
        else:
            te_vmax = np.ceil(np.nanmax(data))
        
        sns.set_style("white")
        fig = Figure(figsize=(8, 8), dpi=100)
        #t = np.arange(0, 3, .01)
        ax = fig.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, xticklabels=columns, yticklabels=columns, ax=ax, annot_kws={"size": 10}, vmax = te_vmax ,cmap='Greens')#plot(t, 2 * np.sin(2 * np.pi * t))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        #ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
        ax.set_title('Treatment Equality Difference Results')
        ax.set_xlabel('Protected Characteristics Subgroups')
        ax.set_ylabel('Protected Characteristics Subgroups')
        
        canvas7 = FigureCanvasTkAgg(fig, master=tab7)  # A tk.DrawingArea.
        canvas7.draw()
        canvas7.get_tk_widget().grid(column=2, row=3, sticky=S)
        
        image_save_button_sp = Button( tab7, text='Save Graph as Image', command=lambda: save_image(fig))
        image_save_button_sp.grid(column=1, row=3, sticky=N)
        
    save_button_sp = Button(tab7, text='Save Results', command=lambda: save_file(df_result_te_s))
    save_button_sp.grid(column=1, row=2, sticky=N)



def my_update(): # called when checkbutton is clicked 
    global my_ref
    column_search = [v for v in my_ref if my_ref[v].get()]

    ### One more way to Create the same List of checked columns 
    #column_search=[] # create a blank list 
    for j in my_ref:
        pass # 
        #print(j,my_ref[j].get()) # Key and value 
        #if my_ref[j].get():
        #    column_search.append(j)
              
        #print(column_search) # print the selected column names 
    lb.config(text=' '.join(column_search)) # Show in the Label

sns.set()
root = Tk()
root.title("Fairness Auditing Tool")
tabControl = ttk.Notebook(root)
root.state("zoomed")
f = font.Font(family='Comic Sans MS', size = '24')
f_small = font.Font(family='Comic Sans MS', size = '16')
style = ttk.Style()
style.configure('big.TButton', font=('Comic Sans MS', 16))
style.configure("Treeview.Heading", font=('Comic Sans MS', 16))
#style.configure('My.TFrame', background='white')

tab1 = ttk.Frame(tabControl, style='My.TFrame')
tab2 = ttk.Frame(tabControl, style='My.TFrame')
tab3 = ttk.Frame(tabControl, style='My.TFrame')
tab4 = ttk.Frame(tabControl, style='My.TFrame')
tab5 = ttk.Frame(tabControl, style='My.TFrame')
tab6 = ttk.Frame(tabControl, style='My.TFrame')
tab7 = ttk.Frame(tabControl, style='My.TFrame')
tabControl.add(tab1, text ='Input Data')
tabControl.add(tab2, text ='Statistical Parity - Single Test')
tabControl.add(tab3, text ='Equalised Odds - Single Test')
tabControl.add(tab4, text ='Treatment Equality - Single Test')
tabControl.add(tab5, text ='Statstical Parity - Intersectional Test')
tabControl.add(tab6, text ='Equalised Odds - Intersectional Test')
tabControl.add(tab7, text ='Treatment Equality - Intersectional Test')
tabControl.pack(expand = 1, fill ="both")

#mainframe = ttk.Frame(root, padding="6 6 12 12")
#mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
#root.columnconfigure(0, weight=1)
#root.rowconfigure(0, weight=1)


# bias_column = StringVar()
# bias_entry = ttk.Entry(mainframe, width=7, textvariable=bias_column)
# bias_entry.grid(column=2, row=2, sticky=(W, E))

# result_column = StringVar()
# result_entry = ttk.Entry(mainframe, width=7, textvariable=result_column)
# result_entry.grid(column=2, row=3, sticky=(W, E))

# result = StringVar()
# ttk.Label(mainframe, textvariable=result).grid(column=2, row=4, sticky=(W, E))

# ttk.Button(mainframe, text="Calculate", command= lambda: sp_test(df, bias_column_name, result_column_name)).grid(column=3, row=5, sticky=W)

#ttk.Label(tab1, text="Fairness Auditing Tool").grid(column=2, row=1, sticky=N)
ttk.Label(tab1, text="Input Data:", font=f).grid(column=1, row=2, sticky=E)
ttk.Label(tab1, text="Bias Column:", font=f).grid(column=1, row=3, sticky=E)
ttk.Label(tab1, text="Results Column: ", font=f).grid(column=1, row=4, sticky=E)
ttk.Label(tab1, text="Ground Truth Column: ", font=f).grid(column=1, row=5, sticky=E)
ttk.Label(tab1, text="Second Bias Column: ", font=f).grid(column=1, row=6, sticky=E)
ttk.Label(tab1, text="(Only required for intersectional testing) ", font=f_small).grid(column=3, row=6, sticky=W)
ttk.Label(tab1, text='Please upload your dataset and pick the appropriate column headers.', font=f_small).grid(column=3, row=8, sticky=W)
ttk.Label(tab1, text='If you only want to test a single characteristic for fairness, you do not need to change the second bias drop down.', font=f_small).grid(column=3, row=9, sticky=W)
ttk.Label(tab1, text='Please press the clear button between changing the columns.', font=f_small).grid(column=3, row=10, sticky=W)
# for child in mainframe.winfo_children(): 
#     child.grid_configure(padx=10, pady=10)

# data_entry.focus()
# root.bind("<Return>", calculate)




# open button
open_button = Button(tab1, text='Open a File', command=lambda: select_file(), font=f_small).grid(column=2, row=2, sticky=W)
calculate_button_sp = Button(tab2, text='Calculate', command=lambda: calculate_sp(), font=f_small).grid(column=1, row=1, sticky=N)
calculate_button_eo = Button(tab3, text='Calculate', command=lambda: calculate_eo(), font=f_small).grid(column=1, row=1, sticky=N)
calculate_button_te = Button(tab4, text='Calculate', command=lambda: calculate_te(), font=f_small).grid(column=1, row=1, sticky=N)
calculate_button_sp_int = Button(tab5, text='Calculate', command=lambda: calculate_int_sp(), font=f_small).grid(column=1, row=1, sticky=N)
calculate_button_te_int = Button(tab7, text='Calculate', command=lambda: calculate_int_te(), font=f_small).grid(column=1, row=1, sticky=N)
calculate_button_eo_int = Button(tab6, text='Calculate', command=lambda: calculate_int_eo(), font=f_small).grid(column=1, row=1, sticky=N)
clear_button = Button(tab1, text="Clear Results", command=lambda: do_reset(), font=f_small).grid(column=3, row=6, sticky=E)
#tree = ttk.Treeview(root, show="headings")
#status_label = Label(root, text="", padx=20, pady=10)
#selected_file_label = Label(root, text="Selected File:")
#status_label.pack()

lb1=Label(tab1,bg='lightgreen',text='') # Path 
lb1.grid(row=5,column=5)
lb=Label(tab1,text='') # List display
lb.grid(row=4,column=5)

# datatype of menu text 
#clicked = StringVar()   
# initial menu text 
#clicked.set(create_options(df)[0]) 
# Create Dropdown menu 
#drop = OptionMenu( root , clicked , *create_options(df) )  
# Create button, it will change label text 
#button = Button( root , text = "click Me" , command = show ).pack() 
# Create Label 
#label = Label( root , text = " " ) 
#label.pack() 

root.mainloop()