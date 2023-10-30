import streamlit as st
import math
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sqlite3
from sqlite3 import Connection
from sqlalchemy import MetaData, text, Column, Integer, String, ForeignKey, Table, create_engine, Float, Boolean, DateTime
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

#meta = MetaData()
#engine = create_engine('sqlite:///fantasy_data.db', echo=False)
#Session = sessionmaker(bind=engine)
#session = Session()
#Base = declarative_base()
#conn = sqlite3.connect('fantasy_data.db')
#conn.create_function('sqrt', 1, math.sqrt)

#conn = st.connection('sql')
#df = conn.query("select * From drafted")
#st.dataframe(df)

def SetColor(x):
    if(x == 'Lima Time!'):
        return "green"
    elif(x == 'Harveys Wallbangers'):
        return "yellow"
    elif(x == 'Ugly Spuds'):
        return "red"
    elif(x == 'Charmer'):
        return "purple"
    elif x=='Trouble with the Curve':
        return 'lightgreen'
    else:
        return "blue"

def setOutline(x):
    if (x==owner_select):
        return "white"
    else:
        return 'black'

def setWidth(x):
    if x==owner_select:
        return 1
    else:
        return 0


st.title('XDL Fantasy Baseball')
st.subheader('Manager Scorecard')


@st.cache_resource()#allow_output_mutation=True
def get_connection(path):
    """Put the connection in cache to reuse if path does not change."""
    conn = sqlite3.connect('fantasy_data.db')
    conn.create_function('sqrt', 1, math.sqrt)
    return conn

@st.cache_data(hash_funcs={Connection: id})
def load_data(conn):
    e = pd.read_sql("WITH cte As (\
            SELECT e.cbsid, MAX(p.CBSNAME) name, MAX(e.week) maxWeek, max(e.all_pos) all_pos, max(e.pos1B) pos1B, max(pos2B) pos2B, max(pos3B) pos3B, max(posSS) posSS, \
                max(posMI) posMI, max(posCI) posCI, max(posOF) posOF, max(posDH) posDH, max(posSP) posSP, max(posRP) posRP, max(posP) posP \
            FROM eligibility e \
            INNER JOIN players p on (e.cbsid=p.cbsid) \
            GROUP BY e.cbsid) \
            SELECT cte.* FROM cte INNER JOIN eligibility e On (cte.cbsid=e.cbsid AND cte.maxWeek=e.week)", conn)

    z = pd.read_sql("SELECT z.cbsid, p.CBSNAME player, o.owner, COALESCE(d.paid,0) paid, ROUND(z*4.18,1) value, ROUND(z*4.18-d.paid,1) surplus, ROUND(z,2) z, \
                R, RBI, HR, SB, AB, H, Ha, BBa, ER, BA_cnt, BA, IP, W, SO, t.SvHld, ERA, ERA_cnt, WHIP, WHIP_cnt, zR, zRBI, zHR, zSB, zBA \
            FROM vw_players_season_z z \
            LEFT JOIN players p On (z.cbsid=p.cbsid) \
            INNER JOIN vw_player_totals t On (z.cbsid=t.cbsid) \
            LEFT JOIN drafted d On (z.cbsid=d.cbsid) \
            LEFT JOIN owners o On (d.owner_id=o.owner_id) \
            WHERE z.year=2023 \
            ORDER BY z desc", conn)

    df = z.merge(e[['cbsid', 'all_pos', 'pos1B', 'pos2B', 'pos3B', 'posSS', 'posMI', 'posCI', 'posOF', 'posDH', 'posSP', 'posRP', 'posP']], on='cbsid', how='left')
    df.all_pos.fillna('',inplace=True)
    df['type'] = df.all_pos.apply(lambda x: 'h' if 'DH' in x else 'p')

    df['surplus_adj'] = df['surplus'] + abs(df['surplus'].min())
    return df

engine = get_connection("fantasy_data.db")
df = load_data(engine)
a = df[df['owner'].notna()]

# --- Sidebar options
year_select = st.sidebar.radio("Draft Season", (2023, 2024))

owner_select = st.sidebar.radio("Fantasy Team Owner",
    tuple(['All']+list(df.owner.sort_values().unique()))
)
# --- 

#df.loc[df['paid']==0, 'hist'] = '0'
df.loc[df['paid'].between(1,4), 'hist'] = '01 - 04'
df.loc[df['paid'].between(5,9), 'hist'] = '05 - 09'
df.loc[df['paid'].between(10,14), 'hist'] = '10 - 14'
df.loc[df['paid'].between(15,19), 'hist'] = '15 - 19'
df.loc[df['paid'].between(20,24), 'hist'] = '20 - 24'
df.loc[df['paid'].between(25,29), 'hist'] = '25 - 29'
df.loc[df['paid'].between(30,34), 'hist'] = '30 - 34'
df.loc[df['paid'].between(35,39), 'hist'] = '35 - 39'
df.loc[df['paid'].between(40,260), 'hist'] = '40+'

#hist_mean = pd.pivot_table(df, values='player', aggfunc='count', index='owner', columns='hist', fill_value=0).mean().reset_index()
#hist_mean.columns = ['price', 'count']

#owner_hist_mean = pd.pivot_table(df[df['owner']==owner_select], values='player', aggfunc='count', index='owner', columns='hist', fill_value=0).mean().reset_index()
#owner_hist_mean.columns = ['price', 'count']

hist = pd.pivot_table(df, values='player', aggfunc='count', index='hist', columns='owner', fill_value=0, margins_name='avg', margins=True).reset_index()
hist['avg'] = hist['avg'] / df.owner.nunique()


fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=a.paid,
    y=a.value,
    mode='markers',
    marker=dict(color=a['surplus'], cmid=0, size=7),
    marker_line=dict(width=list(map(setWidth, a.owner)), color=list(map(setOutline, a.owner))),
    #marker = dict(color=list(map(SetColor, a.owner))),
    text=a['player'],
    hovertemplate=a['player']+"<br>"+a['owner']+"<br>Paid: "+a['paid'].astype(str)+"<br>Value: "+a['value'].astype(str)
))
fig1.update_layout(
    title='Drafted Players by Cost and Returned Value',
    xaxis_title='Paid',
    yaxis_title='Value'
)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    y=df[df['owner']==owner_select].value,
    x=df[df['owner']==owner_select].paid,
    mode='markers',
    marker=dict(color=df[df['owner']==owner_select]['surplus'], cmid=0, size=7),
    #marker = dict(color=list(map(SetColor, a.owner))),
    text=df[df['owner']==owner_select]['player'],
    hovertemplate=df[df['owner']==owner_select]['player']+"<br>"+df[df['owner']==owner_select]['owner']+"<br>Paid: "+df[df['owner']==owner_select]['paid'].astype(str)+"<br>Value: "+df[df['owner']==owner_select]['value'].astype(str)
))
fig3.update_layout(
    title=f'Drafted Players by {owner_select}',
    yaxis_title='Value',
    yaxis_range = [df.value.min()-1, df.value.max()+1],
    xaxis_title='Paid',
    xaxis_range = [df.paid.min()-5, df.paid.max()+5]
)

if owner_select=='All':
    tab1, tab2, tab3, tab4 = st.tabs(['Top 25 by Value', 'Top 25 by Surplus', 'Owner Summary', 'Chart'])

    with tab1:
        st.subheader("Top 25 by Value")
        st.dataframe(df[['owner', 'player', 'value']].sort_values('value', ascending=False).head(25))

    with tab2:
        st.subheader("Top 25 by Surplus")
        st.dataframe(df[['owner', 'player', 'surplus']].sort_values('surplus', ascending=False).head(25))

    with tab3:
        st.subheader('Draft Summary by Owner')
        sub = df.groupby('owner').agg({'value':'sum', 'surplus':'sum', 'R':'sum', 'RBI':'sum', 'ER':'sum', 
            'HR':'sum', 'SB':'sum', 'H':'sum', 'AB':'sum', 'BBa':'sum', 'Ha':'sum', 'IP':'sum', 'W':'sum', 
            'SO':'sum', 'SvHld':'sum'}).sort_values('value', ascending=False).reset_index()
        sub['BA'] = sub['H']/sub['AB']
        sub['ERA'] = sub['ER']/sub['IP']*9
        sub['WHIP'] = (sub['Ha']+sub['BBa'])/sub['IP']
        st.dataframe(sub)
    
    with tab4:
        st.plotly_chart(fig1)

else:
    t1, t2, t3 = st.tabs(['Drafted Team', 'Draft Histogram', 'Chart'])
    with t1:
        st.write('Draft by',owner_select)
        st.dataframe(df[df['owner']==owner_select][['player', 'surplus_adj', 'paid', 'value', 'surplus', 'R', 'RBI', 'HR', 'SB', 'BA', 'W', 'SO', 'SvHld', 'ERA', 'WHIP']])
    
    with t2:
        fig2 = go.Figure(
            data=[
                go.Bar(name='League', x=hist['hist'].iloc[:-1], y=hist['avg'].iloc[:-1]),
                go.Bar(name=owner_select, x=hist['hist'].iloc[:-1], y=hist[owner_select].iloc[:-1])
        ])
        fig2.update_layout(barmode='group')
        st.plotly_chart(fig2)
        
        #st.dataframe(pd.pivot_table(df, values='player', aggfunc='count', index='hist', columns='owner', fill_value=0, margins=True).reset_index())

    with t3:
        st.plotly_chart(fig3)        
