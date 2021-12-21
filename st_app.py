from pulp import *
from time import *
import streamlit as st
import pandas as pd

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

df_base = pd.read_csv("output.csv", decimal=",", dtype={"Order ID": str, "Item Weight": float, "Total Items Sold": int})

show_base = st.button("Ver base de dados")
if show_base:
    st.write(df_base)

selected_order = st.sidebar.selectbox('Selecione o número do pedido desejado', df_base["Order ID"].unique())
choose_order = st.sidebar.button("Otimizar empacotamento")
if choose_order:

    sku_list, order_details = get_items(selected_order)

    st.write(f"Os itens do pedido {selected_order} são:") 
    st.write(f"{sku_list}")

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
    st.sidebar.write("Solved in %s seconds" % solvedTime)

    st.sidebar.write("Status:", LpStatus[bpp.status])

    #bins used 
    st.sidebar.write("\nBins used: " + str(sum(([y[j].value() for j in range(m)]))))
    st.sidebar.write("Volume ocupado: %f"  %((sum(([y[j].value() for j in range(m)])))*binCapacity))

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
        st.write("Volume total de itens: %f" %vol[b])

