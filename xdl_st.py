import streamlit as st
import math
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sqlite3
from sqlite3 import Connection
import optimize_lineup as ol
#from sqlalchemy import MetaData, text, Column, Integer, String, ForeignKey, Table, create_engine, Float, Boolean, DateTime
#from sqlalchemy.orm import relationship, backref, sessionmaker
#from sqlalchemy.ext.declarative import declarative_base

#conn = sqlite3.connect('fantasy_data.db')
#conn.create_function('sqrt', 1, math.sqrt)
#conn = st.connection('sql')
#df = conn.query("select * From drafted")
#st.dataframe(df)

def optimize_team(tm, data):
    w = ol.Optimized_Lineups(tm, data)
    #print(tm)
    w._make_pitcher_combos()
    w._make_hitter_combos()
    #print(w.pitcher_optimized_z, w.pitcher_optimized_lineup)
    #print(w.hitter_optimized_z, w.hitter_optimized_lineup)
    return w

optimized = {   
    '9 Grand Kids':{'h':3.4, 'p':10.1},
    'Brewbirds':{'h':14.4, 'p':17.0},
    'Charmer':{'h':27.3,'p':25.4},
    'Dirty Birds':{'h':-16.4,'p':24.1},
    'Harveys Wallbangers':{'h':14.3,'p':-1.8},
    'Lil Trump & the Ivanabees':{'h':-17.4,'p':28.2},
    'Lima Time!':{'h':3.1,'p':23.2},
    'Roid Ragers':{'h':-17.9,'p':16.7},
    'Trouble with the Curve':{'h':15.0,'p':28.7},
    'Ugly Spuds':{'h':-10.5,'p':9.0},
    'Wiscompton Wu-Tang':{'h':8.2,'p':27.3},
    'Young Guns':{'h':-23.3,'p':19.0},
}

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
st.subheader('Draft Summary')


@st.cache_resource()#allow_output_mutation=True
def get_connection(path):
    """Put the connection in cache to reuse if path does not change."""
    conn = sqlite3.connect('fantasy_data.db')
    conn.create_function('sqrt', 1, math.sqrt)
    return conn

@st.cache_data(hash_funcs={Connection: id})
def load_data(conn):
    z = pd.read_sql("WITH cte As (\
            SELECT e.cbsid cbsid_e, MAX(p.CBSNAME) name, MAX(e.week) maxWeek, max(e.all_pos) all_pos, max(e.posC) posC, max(e.pos1B) pos1B, max(pos2B) pos2B, max(pos3B) pos3B, max(posSS) posSS, \
                max(posMI) posMI, max(posCI) posCI, max(posOF) posOF, max(posDH) posDH, max(posSP) posSP, max(posRP) posRP, max(posP) posP \
            FROM eligibility e \
            INNER JOIN players p on (e.cbsid=p.cbsid) \
            GROUP BY e.cbsid) \
            SELECT d.cbsid, COALESCE(p.CBSNAME, 'DNP') player, o.owner, COALESCE(d.paid,0) paid, ROUND(z*4.18,1) value, ROUND(z*4.18-d.paid,1) surplus, ROUND(z,2) z, \
                R, RBI, HR, SB, AB, H, Ha, BBa, ER, BA_cnt, BA, IP, W, SO, t.SvHld, ERA, ERA_cnt, WHIP, WHIP_cnt, zR, zRBI, zHR, zSB, zBA, d.year, d.keeper, cte.*, 0 As optimized \
            FROM drafted d \
            LEFT JOIN players p On (d.cbsid=p.cbsid) \
            LEFT JOIN owners o On (d.owner_id=o.owner_id) \
            LEFT JOIN vw_players_season_z z On (p.cbsid=z.cbsid) \
            LEFT JOIN vw_player_totals t On (z.cbsid=t.cbsid) \
            LEFT JOIN cte On (d.cbsid=cte.cbsid_e) \
            LEFT JOIN eligibility e On (cte.cbsid_e=e.cbsid AND cte.maxWeek=e.week) \
            ORDER BY z desc", conn)

    #df = z.merge(e[['cbsid', 'all_pos', 'pos1B', 'pos2B', 'pos3B', 'posSS', 'posMI', 'posCI', 'posOF', 'posDH', 'posSP', 'posRP', 'posP']], on='cbsid', how='left')
    z.all_pos.fillna('',inplace=True)
    z['type'] = z.all_pos.apply(lambda x: 'h' if 'DH' in x else 'p')
    z['surplus_adj'] = z['surplus'] + abs(z['surplus'].min())
    return z

engine = get_connection("fantasy_data.db")
orig_df = load_data(engine)

# --- Sidebar options
year_select = st.sidebar.radio("Draft Season", (2022, 2023), index=1)

owner_select = st.sidebar.radio("Fantasy Team Owner",
    tuple(['All']+list(orig_df.owner.sort_values().unique()))
)
# --- 

df = orig_df[orig_df['year']==year_select].reset_index().copy()
a = df[df['owner'].notna()]


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


opt_df = pd.DataFrame(optimized).T
opt_df['Total Optimized Z'] = opt_df['h']+opt_df['p']

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
    hovertemplate=df[df['owner']==owner_select]['player']+"<br>"+df[df['owner']==owner_select]['owner']+"<br>Paid: "+df[df['owner']==owner_select]['paid'].astype(str)+"<br>Value: "+df[df['owner']==owner_select]['value'].astype(str)+"<br>Surplus: "+df[df['owner']==owner_select]['surplus'].astype(str)
))
fig3.update_layout(
    title=f'Drafted Players by {owner_select}',
    yaxis_title='Value',
    yaxis_range = [df.value.min()-1, df.value.max()+1],
    xaxis_title='Paid',
    xaxis_range = [df.paid.min()-5, df.paid.max()+5]
)

fig4 = go.Figure(
    go.Scatter(
        x=pd.DataFrame(optimized).T.h,
        y=pd.DataFrame(optimized).T.p,
        mode='markers',
        text=pd.DataFrame(optimized).T.index,
        hovertemplate="%{text}<br><br>Hitting: "+pd.DataFrame(optimized).T.h.astype(str)+"<br>Pitching: "+pd.DataFrame(optimized).T.p.astype(str),
    )
)
fig4.add_hline(y=0, line_width=3, line_dash="dash", line_color="green")
fig4.add_vline(x=0, line_width=3, line_dash="dash", line_color="green")
fig4.update_layout(
    xaxis_title='Sum Z of Drafted Hitters',
    yaxis_title='Sum Z of Drafted Pitchers',
)

if owner_select=='All':
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Awards', 'Chart', 'Optimized Plot', 'Summary', 'Top 25 by Value', 'Top 25 by Surplus', 'Top Players by Position'])

    with tab0:
        st.markdown('Best Draft :trophy: Charmer')
        st.markdown('Best Hitting Draft :star: Charmer')
        st.markdown('Best Pitching Draft :star: Trouble with the Curve')
        st.markdown('Worst Draft :cry: Young Guns')
        st.markdown('Best Player :crown: Ronald Acuna Jr')
        #df.sort_values('surplus',ascending=False).iloc[0]['player']
        st.markdown('Best Return Value :moneybag: Julio Rodriguez')
        st.write()
        best_keepers = df[df['keeper']==1].groupby('owner').agg({'surplus':'sum'}).sort_values('surplus', ascending=False).reset_index().iloc[0]
        st.write('Best Keepers :trophy: ', best_keepers['owner'], " (", ", ".join(df[(df['owner']==best_keepers['owner']) & (df['keeper']==1)].player.tolist()),")")

    with tab1:
        st.plotly_chart(fig1)
    
    with tab2:
        st.plotly_chart(fig4)
    
    with tab3:
        sub = df.groupby('owner').agg({'value':'sum', 'surplus':'sum'}).sort_values('value', ascending=False).reset_index()
        #sub['BA'] = sub['H']/sub['AB']
        #sub['ERA'] = sub['ER']/sub['IP']*9
        #sub['WHIP'] = (sub['Ha']+sub['BBa'])/sub['IP']
        #sub.rename(columns={'h':''})
        sub = sub.merge(opt_df, left_on='owner', right_index=True, how='left')
        
        st.dataframe(sub.sort_values('Total Optimized Z', ascending=False),
            use_container_width=True,
            height=((sub.shape[0] + 1) * 35 + 3), 
            hide_index=True, 
            column_order=['owner', 'Total Optimized Z', 'h', 'p', 'value', 'surplus'],
            column_config={"h":"Optimized Hitting Z", "p":"Optimized Pitching Z"})

    with tab4:
        numRows = 25
        st.subheader("Top 25 Players by Value")
        st.dataframe(df[['player', 'value', 'owner']].sort_values('value', ascending=False).head(numRows), 
            height=((numRows + 1) * 35 + 3),
            column_config={
                "value":st.column_config.NumberColumn("value",format="$%d")
            }
        )

    with tab5:
        numRows = 25
        st.subheader("Top 25 Players by Surplus")
        st.dataframe(df[['player', 'surplus', 'owner']].sort_values('surplus', ascending=False).head(numRows), 
        height=((numRows + 1) * 35 + 3),
        column_config={
                "surplus":st.column_config.NumberColumn("surplus",format="$%d")
            })

    with tab6:
        st.write('Catchers')
        st.dataframe(df[df['posC']==1][['player', 'owner', 'value', 'paid', 'surplus', 'z', 'R', 'RBI', 'HR', 'SB', 'BA', 'BA_cnt']].sort_values('value', ascending=False).head(12),
            height=((12 + 1) * 35 + 3),
            hide_index=True)

        st.write('First Basemen')
        st.dataframe(df[df['pos1B']==1][['player', 'owner', 'value', 'paid', 'surplus', 'z', 'R', 'RBI', 'HR', 'SB', 'BA', 'BA_cnt']].sort_values('value', ascending=False).head(12),
            height=((12 + 1) * 35 + 3),
            hide_index=True)
        
        st.write('Second Basemen')
        st.dataframe(df[df['pos2B']==1][['player', 'owner', 'value', 'paid', 'surplus', 'z', 'R', 'RBI', 'HR', 'SB', 'BA', 'BA_cnt']].sort_values('value', ascending=False).head(12),
            height=((12 + 1) * 35 + 3),
            hide_index=True)
        
        st.write('Third Basemen')
        st.dataframe(df[df['pos3B']==1][['player', 'owner', 'value', 'paid', 'surplus', 'z', 'R', 'RBI', 'HR', 'SB', 'BA', 'BA_cnt']].sort_values('value', ascending=False).head(12),
            height=((12 + 1) * 35 + 3),
            hide_index=True)
    
        st.write('Shortstops')
        st.dataframe(df[df['posSS']==1][['player', 'owner', 'value', 'paid', 'surplus', 'z', 'R', 'RBI', 'HR', 'SB', 'BA', 'BA_cnt']].sort_values('value', ascending=False).head(12),
            height=((12 + 1) * 35 + 3),
            hide_index=True)

        st.write('Outfielders')
        st.dataframe(df[df['posOF']==1][['player', 'owner', 'value', 'paid', 'surplus', 'z', 'R', 'RBI', 'HR', 'SB', 'BA', 'BA_cnt']].sort_values('value', ascending=False).head(36),
            height=((36 + 1) * 35 + 3),
            hide_index=True)
        
        st.write('Pitchers')
        st.dataframe(df[df['posP']==1][['player', 'owner', 'value', 'paid', 'surplus', 'z', 'W', 'SO', 'SvHld', 'ERA', 'WHIP', 'ERA_cnt']].sort_values('value', ascending=False).head(20),
            height=((20 + 1) * 35 + 3),
            hide_index=True)


else:
    t1, t2, t3, t4 = st.tabs(['Optimized Lineup', 'Chart', 'Draft Histogram', 'Drafted Team'])
    
    with t1:
        try:
            st.subheader("Total Z for Drafted Players")
            with st.expander("What is Z?"):
                st.write("""
                            Z refers to a z-score which is a statistic that compares a single value to a group of values. 
                            Each of the hitter and pitcher metrics gets a z-score. Then those are summed to get the total z. 
                            That number is multiplied by a conversion factor to get Value.
                        """)

            opt = ol.Optimized_Lineups(owner_select, df.rename(columns={'owner':'Owner', 'player':'Player'}))
            opt._make_hitter_combos()
            opt._make_pitcher_combos()
            h = pd.DataFrame({'Pos':['C', '1B', '2B', 'SS', '3B', 'MI', 'CI', 'OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'DH1', 'DH2'], 'Player':opt.hitter_optimized_lineup})
            p = pd.DataFrame({'Pos':['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'], 'Player':opt.pitcher_optimized_lineup})
            
            z_col1, z_col2, z_col3 = st.columns(3)
            z_col1.metric('Total Z', value=round(opt.hitter_optimized_z+opt.pitcher_optimized_z,1))
            z_col2.metric("Z from Hitters: ",round(opt.hitter_optimized_z,1))
            z_col3.metric("Z from Pitchers: ",round(opt.pitcher_optimized_z,1))
            
            lineup_col1, lineup_col2 = st.columns(2)
            lineup_col1.dataframe(h, hide_index=True, width=250, height=((h.shape[0] + 1) * 35 + 3))
            lineup_col2.dataframe(p, hide_index=True, width=250, height=((p.shape[0] + 1) * 35 + 3))

        except:
            st.write('Lineup failed to optimize')

    with t2:
        st.plotly_chart(fig3)
    
    with t3:
        fig2 = go.Figure(
            data=[
                go.Bar(name='League', x=hist['hist'].iloc[:-1], y=hist['avg'].iloc[:-1]),
                go.Bar(name=owner_select, x=hist['hist'].iloc[:-1], y=hist[owner_select].iloc[:-1])
        ])
        fig2.update_layout(barmode='group')
        st.plotly_chart(fig2)
        
    with t4:
            st.write('Draft by',owner_select)
            st.dataframe(df[df['owner']==owner_select][['player', 'surplus_adj', 'paid', 'value', 'surplus', 'R', 'RBI', 'HR', 'SB', 'BA', 'W', 'SO', 'SvHld', 'ERA', 'WHIP']], 
                use_container_width=True, hide_index=True, height=((df[df['owner']==owner_select].shape[0] + 1) * 35 + 3))

