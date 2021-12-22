from pulp import *
from ortools.linear_solver import pywraplp
from time import *
import streamlit as st
import pandas as pd

model_list = ["Bin Packing", "Variable Sized Bin Packing"]
library_list = ["PuLP", "OR-Tools"]

df_base = pd.read_csv("output.csv", decimal=",", dtype={"Order ID": str, "Item Weight": float, "Total Items Sold": int})

show_base = st.button("Ver base de dados")
if show_base:
    st.write(df_base)

selected_model = st.selectbox('Selecione o modelo desejado', model_list)
selected_library = st.selectbox('Selecione a biblioteca desejado', library_list)

selected_order = st.sidebar.selectbox('Selecione o número do pedido desejado', df_base["Order ID"].unique())
df = df_base[df_base["Order ID"] == selected_order]
sku_list = df["Item SKU"].values.tolist()

choose_order = st.sidebar.button("Escolher pedido")
if choose_order:
    st.sidebar.write(f"Os itens do pedido {selected_order} são:") 
    st.sidebar.write(f"{sku_list}")

optimize = st.button("Otimizar")

st.text(" ")
st.text(" ")
st.text(" ")

def bpp_pulp(selected_order: str):
    def get_items(order_id: int):
        df = pd.read_csv("output.csv", decimal=",", dtype={"Order ID": str, "Item Weight": float, "Item Final Volume": float, "Total Items Sold": int})
        
        df = df[df["Order ID"] == order_id]
        sku_list = df["Item SKU"].values.tolist()
        
        li = []
        order_details = dict()
        for row in df.to_dict("records"):
            order_details[row["Item SKU"]] =  {
                    "weight": row["Item Weight"],
                    "volume": row["Item Final Volume"],
                }

        return sku_list, order_details
    
    sku_list, order_details = get_items(selected_order)

    #set of items
    I = sku_list

    #number of items
    n = len(I)

    #max number of bins
    m = n

    #volume of items
    wi = dict()
    for i in I:
        wi[i] = order_details[i]['volume']

    #create binary variable assigned 1 when bin is used, 0 otherwise
    y = LpVariable.dicts("BinUsed", range(m), lowBound = 0, upBound = 1, cat = LpInteger)

    #create binary variable assigned 1 when item i is placed into bin j
    tupleItemBin = [(I[i], binNum) for i in range(n) 
                    for binNum in range(m)]
    x = LpVariable.dicts("itemBin", tupleItemBin, lowBound = 0, upBound = 1, cat = LpInteger)

    #set the problem
    bpp = LpProblem("BinPackingProblem", LpMinimize)

    #set ojective funcion
    bpp += lpSum([y[j] for j in range(m)])

    #set constraint 1: every item must be at 1 bin
    for i in I:
        bpp += lpSum([x[(i, j)] for j in range(m)]) == 1

    #set constraint 2: number of itens in bin must not exceed its capacity, for every bin
    binCapacity = 5.7
    for j in range(m):
        bpp += lpSum([x[(I[i], j)]*wi[I[i]] for i in range(n)]) <= binCapacity*y[j]

    #solve optimization
    startTime = time()
    bpp.solve()
    solvedTime = time() - startTime

    sleep(1)
    st.sidebar.write("Resolvido em %s segundos" % solvedTime)

    st.sidebar.write("Status:", LpStatus[bpp.status])

    #bins used 
    st.sidebar.write("Caixas usadas: " + str(sum(([y[j].value() for j in range(m)]))))
    st.sidebar.write("Volume ocupado pelas caixas: %f"  %((sum(([y[j].value() for j in range(m)])))*binCapacity))

    bins = {}
    vol = {}
    for itemBinPair in x.keys():
        if(x[itemBinPair].value() == 1):
            itemNum = itemBinPair[0]
            binNum = itemBinPair[1]
            
            if binNum in bins:
                bins[binNum].append(itemNum)
                vol[binNum] = vol[binNum] + wi[itemNum]

            else:
                bins[binNum] = [itemNum]
                vol[binNum] = wi[itemNum]

    st.markdown('## Instrução de embalagem')
    for b in bins.keys():
        st.write("Caixa " + str(b) + ": " + str(bins[b]))
        st.write("Volume de itens: %f" %vol[b])


def vsbpp_pulp(selected_order: str):
    def get_items(order_id: int):
        df = pd.read_csv("output.csv", decimal=",", dtype={"Order ID": str, "Item SKU": str, "Item Final Volume": float, "Item Weight": float})

        df_get = df[df["Order ID"] == order_id]

        sku_list = df_get["Item SKU"].values.tolist()

        li = []
        order_details = dict()
        for row in df_get.to_dict("records"):
            order_details[row["Item SKU"]] =  {
                    "weight": row["Item Weight"],
                    "volume": row["Item Final Volume"]
                }
        return sku_list, order_details
    
    sku_list, order_details = get_items(selected_order)

    #set of items
    I = sku_list

    #number of items
    n = len(I)

    #set of bin's types
    T = ["Sacola", "Caixa_Pequena", "Caixa_Grande"]

    #max number of bins 
    Uj = n*len(T)

    #volume of item i 
    wi = dict()
    for i in I:
        wi[i] = order_details[i]['volume']

    #cost of bin type t: total volume of box
    Ct = {"Sacola": 7.500, "Caixa_Pequena": 9.000, "Caixa_Grande": 15.750}

    #capacity of bin type t 
    Wt = {"Sacola": 2.1, "Caixa_Pequena": 5.7, "Caixa_Grande": 10.6}   

    #create binary variable assigned 1 when bin is used, 0 otherwise
    y = LpVariable.dicts("BinUsed", range(Uj), lowBound = 0, upBound = 1, cat = LpInteger)

    #create binary variable assigned 1 when item i is placed into bin j
    tupleItemBin = [(I[i], binNum) for i in range(n) 
                    for binNum in range(Uj)]
    x = LpVariable.dicts("itemBin", tupleItemBin, lowBound = 0, upBound = 1, cat = LpInteger)

    #set the problem
    vbpp = LpProblem("VariableBinPackingProblem", LpMinimize)

    #set ojective funcion
    vbpp += lpSum([Ct[T[int(j/n)]]*y[j] for j in range(Uj)])

    #set constraint 1: every item must be at 1 bin
    for i in I:
        vbpp += lpSum([x[(i, j)] for j in range(Uj)]) == 1
        
    #set constraint 2: number of itens in bin must not exceed its capacity, for every bin
    for j in range(Uj):
        vbpp += lpSum([x[(I[i], j)]*wi[I[i]] for i in range(n)]) <= (Wt[T[int(j/n)]])*(y[j])
        
    #solve optimization
    startTime = time()
    vbpp.solve()
    solvedTime = time() - startTime
    sleep(1)
    st.sidebar.write("Resolvido em %s segundos" % solvedTime)

    st.sidebar.write("Status:", LpStatus[vbpp.status])

    #bins used 
    st.sidebar.write("Caixas usadas: " + str(sum(([y[j].value() for j in range(Uj)]))))

    bins = {}
    vol = {}
    for itemBinPair in x.keys():
        if(x[itemBinPair].value() == 1):
            itemNum = itemBinPair[0]
            binNum = itemBinPair[1]
            if binNum in bins:
                bins[binNum].append(itemNum)
                vol[binNum] = vol[binNum] + wi[itemNum]
            else:
                bins[binNum] = [itemNum]
                vol[binNum] = wi[itemNum]
    volTotal = 0
    st.markdown('## Instrução de embalagem')
    for b in bins.keys():
        st.write("Caixa " + str(b) + " - " + str(T[int(b/n)]) + ": " + str(bins[b]))
        st.write("Volume de itens: %f" %vol[b])
        volTotal = volTotal + Ct[T[int(b/n)]]
    st.sidebar.write("Volume ocupado pelas caixas: %f" %volTotal)


def bpp_ortools(selected_order: str):
    def get_items(order_id: str):
        df = pd.read_csv("output.csv", decimal=",", dtype={"Order ID": str, "Item SKU": str, "Item Final Volume": float, "Item Weight": float})

        df_get = df[df["Order ID"] == order_id]
        
        sku_list = df_get["Item SKU"].values.tolist()

        data = {}
        for row in df_get:
            data['items'] = sku_list
            data['volume'] = df_get["Item Final Volume"].values.tolist()
            data['bins'] = list(range(len(sku_list)))
            data['bin_capacity'] = 5.7
        return data

    data = get_items(selected_order)
    n = len(data["items"])

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        for j in data['bins']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%s_%s'  %(i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data['bins']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)
        
    # Constraints
    # Each item must be in exactly one bin.
    for i in data['items']:
        solver.Add(sum(x[i, j] for j in data['bins']) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        solver.Add(
            sum(x[(data['items'][i], j)] * data['volume'][i] for i in range(n)) <= y[j] * data['bin_capacity'])

    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum([y[j] for j in data['bins']]))

    startTime = time()
    status = solver.Solve()
    solvedTime = time() - startTime
    print("Solved in %s seconds" % (time() - startTime))

    if(status == 0):
        status_str = "Optimal"

    sleep(1)

    if status == pywraplp.Solver.OPTIMAL:
        st.sidebar.write("Solved in %s seconds" % solvedTime)
        st.sidebar.write("Status:", status_str)
        num_bins = 0.
        st.markdown('## Instrução de embalagem')
        for j in data['bins']:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_volume = 0
                for i in range(n):
                    if x[(data['items'][i], j)].solution_value() > 0:
                        bin_items.append(data['items'][i])
                        bin_volume += data['volume'][i]
                if bin_volume > 0:
                    num_bins += 1
                    st.write('Número da caixa: ', j)
                    st.write('Itens na caixa: ', bin_items)
                    st.write('Volume dos itens: ', bin_volume)
                    st.text(" ")
                    st.text(" ")
        st.text(" ")
        st.sidebar.write('Caixas usadas: ', num_bins)
    else:
        st.sidebar.write('O problema não tem solução ótima.')
        
def vsbpp_ortools(selected_order: str):
    def get_items(order_id: str):
        df = pd.read_csv("output.csv", decimal=",", dtype={"Order ID": str, "Item SKU": str, "Item Final Volume": float, "Item Weight": float})

        df_get = df[df["Order ID"] == order_id]
        
        sku_list = df_get["Item SKU"].values.tolist()
        bin_types = ['Sacola', 'Caixa_Pequena', 'Caixa_Grande']

        data = {}
        for row in df_get:
            data['items'] = sku_list
            data['volume'] = df_get["Item Final Volume"].values.tolist()
            data['bins'] = list(range(len(bin_types)*len(sku_list)))
            data['bin_types'] = bin_types
        return data

    data = get_items(selected_order)
    n = len(data["items"])

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    #cost of bin type t: total volume of box
    Ct = {"Sacola": 7.500, "Caixa_Pequena": 9.000, "Caixa_Grande": 15.750}

    #capacity of bin type t 
    Wt = {"Sacola": 2.1, "Caixa_Pequena": 5.7, "Caixa_Grande": 10.6}

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        for j in data['bins']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%s_%s'  %(i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data['bins']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)

    # Constraints
    # Each item must be in exactly one bin.
    for i in data['items']:
        solver.Add(sum(x[i, j] for j in data['bins']) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        solver.Add(
            sum(x[(data['items'][i], j)] * data['volume'][i] for i in range(n)) <= y[j] * Wt[data['bin_types'][int(j/n)]])

    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum([y[j]*Ct[data['bin_types'][int(j/n)]] for j in data['bins']]))

    startTime = time()
    status = solver.Solve()
    solvedTime = time() - startTime
    print("Solved in %s seconds" %solvedTime)

    if(status == 0):
        status_str = "Optimal"

    sleep(1)

    if status == pywraplp.Solver.OPTIMAL:
        st.sidebar.write("Solved in %s seconds" % solvedTime)
        st.sidebar.write("Status:", status_str)
        num_bins = 0.
        bin_total_volume = 0 
        st.markdown('## Instrução de embalagem')
        for j in data['bins']:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_volume = 0
                for i in range(n):
                    if x[(data['items'][i], j)].solution_value() > 0:
                        bin_items.append(data['items'][i])
                        bin_volume += data['volume'][i]
                if bin_volume > 0:
                    num_bins += 1
                    bin_total_volume = bin_total_volume + Ct[data['bin_types'][int(j/n)]]
                    st.write('Número da caixa: ', j, " - ", str(data['bin_types'][int(j/n)]))
                    st.write('Itens na caixa: ', bin_items)
                    st.write('Items volume: ', bin_volume)
                    st.text(" ")
                    st.text(" ")
        st.text(" ")
        st.sidebar.write('Caixas usadas: ', num_bins)
        st.sidebar.write('Total volume: ', bin_total_volume)
    else:
        st.sidebar.write('O problema não tem solução ótima.') 

if optimize:
    if(selected_model == "Bin Packing" and selected_library == "PuLP"):
        bpp_pulp(selected_order)
    if(selected_model == "Bin Packing" and selected_library == "OR-Tools"):
        bpp_ortools(selected_order)
    if(selected_model == "Variable Sized Bin Packing" and selected_library == "PuLP"):
        vsbpp_pulp(selected_order)
    if(selected_model == "Variable Sized Bin Packing" and selected_library == "OR-Tools"):
        vsbpp_ortools(selected_order)