import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import MetaData, text, Column, Integer, String, ForeignKey, Table, create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.sqltypes import DATETIME, TIMESTAMP
from starlette.responses import RedirectResponse
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
from typing import Optional
import requests
import fantasy_utils as fu
import optimize_lineup as ol

meta = MetaData()
engine = create_engine('sqlite:///fantasy_data.db', echo=False)
Session = sessionmaker(bind=engine)
session = Session()

n_teams = 12
tm_players = 23
tm_dollars = 260
player_split = .67
pitcher_split = 1 - player_split
tot_dollars = n_teams * tm_dollars
tot_players = n_teams * tm_players
tot_hitters = n_teams * 14
tot_pitchers = n_teams * 9
total_z_over_0 = pd.read_sql(f'SELECT sum(z) z FROM players{str(datetime.now().year)} WHERE z>0', engine).iloc[0]['z']
#total_z_over_0 = 626.134#701.39442#591.1999720030974
orig_conv =  (tm_dollars/tm_players)*(tot_players/total_z_over_0)
print(f"starting conv_factor: {orig_conv}")
owner_list = ["9 Grand Kids", 'Brewbirds', 'Charmer', 'Dirty Birds', 'Harvey', 'Lima Time', 'Mother', 'Roid Ragers', 'Trouble', 'Ugly Spuds', 'Wu-Tang', 'Young Guns']
# owner_sort = [i[1] for i in enumerate(owners.keys())]

drafted_by_pos = {
    'C':n_teams,
    '1B':round(n_teams*1.5),
    '2B':round(n_teams*1.5),
    '3B':math.floor(n_teams*1.5),
    'SS':math.floor(n_teams*1.5),
    'OF':n_teams*5,
    'MI':n_teams,
    'CI':n_teams,
    'DH':n_teams*2, 
    'P':n_teams*9,
    'SP':round(n_teams*6.5),
    'RP':math.floor(n_teams*2.5),
}



class Bid(BaseModel):
    cbsid: int
    owner: str
    price: Optional[list] = 0
    supp: Optional[int] = 0


players = Table('players'+str(datetime.now().year), meta,
                Column('cbsid', Integer, primary_key=True),
                Column('Paid', Integer),
                Column('Supp', Integer),
                Column('Owner', String(25)),
                Column('Timestamp', DATETIME)
    )

def scale_data(df, cols):
    """
    INPUT: 
        df: original dataframe
        list: subset of columns to scale
    OUTPUT:
        df: scaled data
    """
    scaler = MinMaxScaler()
    scaler.fit(df[cols])
    scaled_df = scaler.transform(df[cols])
    scaled_df = pd.DataFrame(scaled_df, index=df.index)
    scaled_df.columns=[df[cols].columns.tolist()]
    return scaled_df



def add_distance_metrics(h, player_id, col_list):
    scaled_df = scale_data(h[h['Owner'].isna()].set_index('cbsid'), col_list)
    df2 = h[h['Owner'].isna()].loc[:,['cbsid', 'Name', 'Pos']+col_list].set_index('cbsid')
    for j, row in scaled_df.iterrows():
        #df2.at[j,'corr'] = pearsonr(scaled_df.loc[player_id,col_list],row[col_list])[0]
        df2.at[j,'eucl_dist'] = np.linalg.norm(scaled_df.loc[player_id,col_list] - row[col_list])
        #df2.at[j,'manh_dist']= sum(abs(e - s) for s, e in zip(scaled_df.loc[player_id,col_list], row[col_list]))
    return df2.sort_values('eucl_dist').iloc[1:11]



def next_closest_in_tier(df, pos, cbsid):
    try:
        i = df[(df['Primary_Pos']==pos) & (df['cbsid']==cbsid) & (df['Owner'].isna())].index[0]
        val = df[(df['Primary_Pos']==pos) & (df['Owner'].isna()) & (df['cbsid']==cbsid)]['Value'].iloc[0]
        return round(df[df['cbsid']==cbsid]['Value'].iloc[0] - df[(df['Primary_Pos']==pos) & (df['Owner'].isna()) & (df['Value']<=val)].iloc[1]['Value'],1)
    except:
        return 0



def optimize_team(tm, df):
    x = ol.Optimized_Lineups(tm, df[df['Owner']==tm])
    x.catchers = [k for k,v in x.h_dict.items() if 'C,' in v['all_pos']]
    x._make_pitcher_combos()
    x._make_hitter_combos()
    return {'pitcher_z':x.pitcher_optimized_z, 'pitcher_lineup':x.pitcher_optimized_lineup, 'hitter_z':x.hitter_optimized_z, 'hitter_lineup':x.hitter_optimized_lineup}


def build_roster(n_teams, owner_list, df, pos_order):
    """
    Build a roster DataFrame for all teams. Uses cbsid internally for reliable
    lookups, then converts to player names before returning for display.
    """
    roster_positions = [
        'C', '1B', '2B', '3B', 'SS', 'MI', 'CI',
        'OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'DH1', 'DH2',
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9',
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
    ]
    bench_slots = [p for p in roster_positions if p.startswith('B')]
    roster = pd.DataFrame(
        index=roster_positions,
        data=np.zeros((len(roster_positions), n_teams), dtype=int),
        columns=owner_list,
    )

    for tm in owner_list:
        team_players = df[df['Owner'] == tm][
            ['cbsid', 'Name', 'Owner', 'Pos', 'Paid', 'Supp', 'Team', 'Timestamp', 'Keeper', 'Value']
        ].copy()

        # --- Bench players (Paid == 0) ---
        bench_players = team_players[team_players['Paid'] == 0].sort_values('Supp')
        for _, row in bench_players.iterrows():
            cbsid = int(row['cbsid'])
            target = 'B' + str(int(row['Supp']))
            # Handle collision: if target bench slot is taken, find the next open one
            if roster.loc[target, tm] != 0:
                for slot in bench_slots:
                    if roster.loc[slot, tm] == 0:
                        target = slot
                        break
            roster.loc[target, tm] = cbsid

        # --- Active players (Paid > 0), least versatile first ---
        active_players = team_players[team_players['Paid'] > 0].copy()
        active_players['versatility'] = active_players['Pos'].apply(
            lambda pos: len(fu.get_eligible_positions(pos, pos_order))
        )
        active_players = active_players.sort_values(['versatility', 'Timestamp'])

        for _, row in active_players.iterrows():
            cbsid = int(row['cbsid'])
            if cbsid in roster[tm].values:
                continue

            player_dict = row[
                ['cbsid', 'Name', 'Owner', 'Pos', 'Paid', 'Supp', 'Team', 'Timestamp', 'Keeper', 'Value']
            ].to_dict()
            results = fu.check_roster_pos(player_dict, roster, df, pos_order)

            for result in results:
                for pid, position in result.items():
                    if position is not None:
                        roster.loc[position, tm] = pid

    # Convert cbsids to player names for display
    return fu.roster_cbsid_to_names(roster, df)


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
async def slash_route():
    return RedirectResponse('/draft')

@app.get("/draft")
async def draft_view(request: Request, status: Optional[str] = 'ok'):
    h = pd.read_sql('players'+str(datetime.now().year), engine)
    h['cbsid'] = h['cbsid'].astype(int)
    h.loc[h['Primary_Pos'].isin(['C', '1B', '2B', '3B', 'OF', 'DH', 'SS']), 'PosClass'] = 'h'
    h.fillna({'PosClass':'p'}, inplace=True)
    h[(h['z']>0) & (h['PosClass']=='h')][['HR', 'SB', 'R', 'RBI', 'BA', 'H', 'AB']].sum()/n_teams
    
    avg_stats = {}
    for a,b in (h[(h['z']>0) & (h['PosClass']=='h')][['HR', 'SB', 'R', 'RBI', 'BA', 'H', 'AB']].sum()/12).reset_index(name='mean').iterrows():
        avg_stats[b['index']] = round(b['mean'],1)
    avg_stats['BA'] = round(avg_stats['H']/avg_stats['AB'],3)

    h.fillna({'Paid':0},inplace=True)
    h['Paid'] = h['Paid'].apply(lambda x: int(x) if x>0 else x)
    for i in ['z', 'Dollars', 'Value', 'Value_ly', 'IP', 'K/9']:
        if i in h.columns:
            h[i] = round(h[i],1)
    for i in ['BA', 'BA_ly', 'Contact%', 'Z-Contact%', 'BB%', 'K%']:
        if i in h.columns:
            h[i] = round(h[i],3)
    for i in ['ERA', 'WHIP', 'Barrel%', 'O-Swing%', 'HardHit%']:
        if i in h.columns:
            h[i] = round(h[i],2)
    for i in ['SO', 'QS', 'SvHld', 'R', 'RBI', 'SB', 'HR']:
        if i in h.columns:
            h.fillna({i:0},inplace=True)
            h[i] = h[i].astype(int)
    owners_df = h.query('Paid>0').groupby('Owner').agg({'Name':'count', 'Paid':'sum', 'z':'sum', 'H':'sum', 'AB':'sum', 'HR':'sum', 'R':'sum', 'RBI':'sum', 'SB':'sum', 'Outs':'sum', 'QS':'sum', 'SO':'sum', 'SvHld':'sum', 'ER':'sum', 'IP':'sum', 'BB':'sum', 'HA':'sum'}).reset_index()
    owners_df.rename(columns={'Name':'Drafted'},inplace=True)
    owners_df['Paid'] = owners_df['Paid'].apply(lambda x: int(x) if x>0 else x)
    owners_df['$/unit'] = round(owners_df['Paid']/owners_df['z'],1)
    owners_df['z'] = round(owners_df['z'],1)
    owners_df['$ Left'] = tm_dollars - owners_df['Paid']
    owners_df['$ Left / Plyr'] = round(owners_df['$ Left'] / (tm_players -owners_df['Drafted']),1)
    owners_df['max_bid'] = owners_df['$ Left'] - (tm_players - owners_df['Drafted'])
    owners_df['Cash'] = round(owners_df['$ Left / Plyr'] / (((tot_dollars - owners_df.Paid.sum()) + owners_df['Paid']) / ((tot_players - owners_df.Drafted.sum()) + owners_df['Drafted'])),2)
    owners_df['Value'] = round((owners_df['z']*orig_conv) - owners_df['Paid'],1)
    owners_df['BA'] = round(owners_df['H']/owners_df['AB'],3)
    owners_df['ERA'] = round(owners_df['ER']/(owners_df['Outs']/3)*9,2)
    owners_df['WHIP'] = round((owners_df['BB']+owners_df['HA'])/(owners_df['Outs']/3),2)
    owners_df['Pts'] = 0
    for i in ['BA', 'HR', 'R', 'RBI', 'SB', 'ERA', 'WHIP', 'QS', 'SO', 'SvHld']:
        owners_df['Pts'] += owners_df[i].rank()
    owners_df['Rank'] = owners_df['Pts'].rank()
    
    # Place drafted players into roster layout
    roster = build_roster(n_teams, owner_list, h, fu.pos_order)
        
    h.loc[h['Paid'].between(1,4), 'hist'] = '1-4'
    h.loc[h['Paid'].between(5,9), 'hist'] = '5-9'
    h.loc[h['Paid'].between(10,14), 'hist'] = '10-14'
    h.loc[h['Paid'].between(15,19), 'hist'] = '15-19'
    h.loc[h['Paid'].between(20,24), 'hist'] = '20-24'
    h.loc[h['Paid'].between(25,29), 'hist'] = '25-29'
    h.loc[h['Paid'].between(30,34), 'hist'] = '30-34'
    h.loc[h['Paid'].between(35,39), 'hist'] = '35-39'
    h.loc[h['Paid']>=40, 'hist'] = '40+'
    dollars_rem = (tot_dollars - owners_df['Paid'].sum())
    #z_rem = (h[h['z']>0]['z'].sum() - owners_df['z'].sum())
    z_rem = h[h['z']>0]['z'].sum() - h[(h['Owner'].notna()) & (h['z']>0)]['z'].sum()
    conv_factor = dollars_rem / z_rem
    print(f"updated conv_factor: {conv_factor}")
    print(f"original conv_factor: {orig_conv}")
    #h['curValue'] = round(h['z']*conv_factor,1)
    h = h.copy()
    h['curValue'] = round(h['Value']*(conv_factor/orig_conv),1)
    h = h.copy()
    h['surplus'] = round(h['Value'] - h['CBS'],2)
    
    return templates.TemplateResponse('draft.html', {'request':request, 'players':h.sort_values('z', ascending=False), 
                                    'status':status,
                                    'owned':h[h['Owner'].notna()], 'owners_df':owners_df.sort_values('Rank', ascending=False), 'roster':roster, 'roster_json':roster.to_dict(orient='records'),
                                    'owner_list': owner_list,
                                    'owners_json':owners_df.to_json(orient='index'), 
                                    'json':h.sort_values('z', ascending=False).to_json(orient='records'),
                                    'avg_stats':avg_stats,
                                    'players_left':(tot_players - owners_df.Drafted.sum()),
                                    'dollars_left':(tot_dollars - owners_df.Paid.sum()),
                                    'inflation_factor':round((conv_factor/orig_conv),2),
                                    'init_dollars_per_z':round((tot_dollars/h[h['z']>=0]['z'].sum()*player_split),2),
                                    'current_dollars_per_z':round(owners_df.Paid.sum() / owners_df.z.sum(),2),
                                    'paid_histogram_data':h[h['Owner']=='Lima Time'].groupby('hist')['Paid'].count().reindex(pd.Series(['1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40+'])).fillna(0).to_json(orient='index'),
                                    'team_z':h.groupby('Owner')[['zHR', 'zSB', 'zR', 'zRBI', 'zBA', 'zQS', 'zSO', 'zSvHld', 'zERA', 'zWHIP']].sum().reset_index().round(2),
                                    })

@app.get("/draft/update_bid")
async def update_db(cbsid=int, owner=str, price=int, supp=Optional[int] == 0):
    print(f"Endpoint triggered for cbsid: {cbsid}, owner: {owner}")
    df = pd.read_sql(f"SELECT cbsid, Name, '{owner}' As Owner, Pos, COALESCE(Paid,{price},0) Paid, Supp, Team, Timestamp, Keeper, Value FROM players{datetime.now().year} WHERE Owner='{owner}' or cbsid={cbsid}", engine)
    df.loc[df['cbsid']==int(cbsid), ['Paid', 'Supp']] = [int(price), int(supp)]
   
    player = df[df['cbsid']==int(cbsid)].iloc[0][['cbsid', 'Name', 'Owner', 'Pos', 'Paid', 'Supp', 'Team', 'Timestamp', 'Keeper', 'Value']].to_dict()
    roster = build_roster(n_teams, owner_list, df, fu.pos_order)
    
    # REMOVE/COMMENT OUT THE ENTIRE ROSTER CHECK TO JUST ADD THE PLAYER
    # results = fu.check_roster_pos(player, roster, df, fu.pos_order)
    # print("Results from first check_roster_pos:", results)
    # for result in results:
    #     print('Processing result:', result)
    #     for val in result.values():
    #         if val is None:
    #             print('no roster spot available')
    #             return RedirectResponse('/draft?status=unrosterable')
   
    # Just add the player directly without position checking
    conn = engine.connect()
    meta.create_all(engine)
    conn.execute(players.update().values(Paid=price, Supp=supp, Owner=owner, Timestamp=datetime.now()).where(players.c.cbsid==cbsid))
    conn.commit()
    conn.close()
    return RedirectResponse('/draft', status_code=303)


@app.get("/draft/sims/{cbsid}")
def sim_players(cbsid: int):
    h = pd.read_sql(f"SELECT * FROM players{datetime.now().year} WHERE cbsid IS NOT NULL", engine)

    if h[h['cbsid']==int(cbsid)].iloc[0]['Primary_Pos'] in ['C', '1B', '2B', '3B', 'SS', 'OF', 'DH']:
        sims = add_distance_metrics(h, cbsid, ['BA', 'R', 'RBI', 'HR', 'SB']).sort_values('eucl_dist')
    else:
        sims = add_distance_metrics(h, cbsid, ['ERA', 'WHIP', 'QS', 'SO', 'SvHld']).sort_values('eucl_dist')
    print(sims)
    sims_data = h[h['cbsid'].isin(sims['Name'].index)][['Name', 'Value']]
    #print('<br>'.join(sims_data))
    return sims_data.to_json(orient='records')#'<br>'.join(sims_data['Name'])



@app.get('/draft/reset_all')
async def reset_all():
    t = text(f"UPDATE players{str(datetime.now().year)} SET Paid=NULL, Supp=NULL, Owner=NULL WHERE Keeper=0")
    conn = engine.connect()
    conn.execute(t)
    conn.commit()
    return RedirectResponse('/draft')



@app.post("/draft/get_bids")
async def get_bids(request: Request):
    data = await request.json()
    opt_pos = ['C', '1B', '2B', '3B', 'SS', 'MI', 'CI', 'OF1', 'OF2', 'OF3', 'OF4','OF5', 'DH1', 'DH2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
    roster = pd.DataFrame(data['roster'])
    roster['Pos'] = opt_pos
    o = pd.DataFrame(data['owners']).T.set_index('Owner').reindex(owner_list)
    o.fillna({'$ Left':260, 'max_bid':237, 'Drafted':0, '$ Left/Plyr':11.3}, inplace=True)
    o['$ Left'] = o['$ Left'].astype(int)

    bids = fu.simulate_auction(data['player_data'], o.to_dict(orient='index'), roster.set_index('Pos'), .3)

    # Resolve auction: ascending $1-increment → winner pays second-highest + 1
    indexed_bids = sorted(enumerate(bids), key=lambda x: x[1], reverse=True)
    winner_idx = indexed_bids[0][0]
    winner_max = indexed_bids[0][1]
    second_max = indexed_bids[1][1] if len(indexed_bids) > 1 else 0
    price = min(second_max + 1, winner_max) if second_max > 0 else 1

    return {
        'winner': owner_list[winner_idx],
        'price': int(price),
        'max_willingness': int(winner_max),
        'bids': dict(zip(owner_list, bids)),
    }



@app.get('/optimize')
async def optimize(tm: str):
    yr = datetime.now().year
    wk = pd.read_sql(f"SELECT max(week) week FROM projections WHERE year={yr}", engine).iloc[0]['week']-1
    df  = pd.read_sql(f"SELECT distinct p.CBSNAME Player, o.owner Owner, r.pos Decision, j.*, e.*, \
                    CASE WHEN e.DH>=5 THEN 'h' ELSE 'p' END As type \
                    FROM roster r \
                    INNER JOIN projections j On (j.cbsid=r.cbsid) \
                    INNER JOIN players p On (r.cbsid=p.cbsid) \
                    INNER JOIN owners o On (r.owner_id=o.owner_id) \
                    INNER JOIN (SELECT cbsid, all_pos, posC C, pos1B '1B', pos2B '2B', pos3B '3B', posSS SS, posOF OF, posDH DH, posSP SP, posRP RP, posP P FROM eligibility WHERE year=2024 and week={wk}) e On (r.cbsid=e.cbsid) \
            WHERE j.year={yr} AND j.week={wk} AND j.proj_type='ros' AND r.year={yr} AND r.week={wk} \
            ORDER BY Owner, year, week", engine)
    
    opt = optimize_team(tm, df)
    df['z'] = round(df['z'],1)
    df.fillna(0,inplace=True)
    opt_pos = ['C', '1B', '2B', 'SS', '3B', 'MI', 'CI', 'OF1', 'OF2', 'OF3', 'OF4','OF5', 'DH1', 'DH2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
    bench = df[(df['Owner']==tm) & (~df['Player'].isin(opt['hitter_lineup']+opt['pitcher_lineup']))].sort_values('type')['Player'].to_list()
    df = df[df['Owner']==tm].set_index('Player').reindex(opt['hitter_lineup']+opt['pitcher_lineup']+bench).reset_index()
    df['optimized_position'] = opt_pos[:df.shape[0]]
    df.loc[df['Player'].isin(opt['hitter_lineup']), 'opt_designation'] = 'starting_hitter'
    df.loc[df['Player'].isin(opt['pitcher_lineup']), 'opt_designation'] = 'starting_pitcher'
    df.loc[(~df['Player'].isin(opt['hitter_lineup'])) & (df['type']=='h'), 'opt_designation'] = 'bench_hitter'
    df.loc[(~df['Player'].isin(opt['pitcher_lineup'])) & (df['type']=='p'), 'opt_designation'] = 'bench_pitcher'
    opt_totals = df[df['Player'].isin(opt['hitter_lineup']+opt['pitcher_lineup'])][['z', 'HR', 'SB', 'R', 'RBI', 'H', 'AB', 'QS', 'SO', 'SvHld', 'IP', 'Ha', 'BBa', 'ER']].sum()
    opt_totals['z'] = round(opt_totals['z'],1)
    opt_totals['hitter_z'] = round(df[df['Player'].isin(opt['hitter_lineup'])]['z'].sum(),1)
    opt_totals['BA'] = round(opt_totals['H']/opt_totals['AB'],3)
    opt_totals['ERA'] = round(opt_totals['ER']/opt_totals['IP']*9,2)
    opt_totals['WHIP'] = round((opt_totals['BBa']+opt_totals['Ha'])/opt_totals['IP'],2)
    opt_totals.to_dict()
    bench_totals = df[~df['Player'].isin(opt['hitter_lineup']+opt['pitcher_lineup'])][['z', 'HR', 'SB', 'R', 'RBI', 'H', 'AB', 'QS', 'SO', 'SvHld', 'IP', 'Ha', 'BBa', 'ER']].sum().to_dict()
    bench_totals['z'] = round(bench_totals['z'],1)
    opt_totals['pitcher_z'] = round(df[df['Player'].isin(opt['pitcher_lineup'])]['z'].sum(),1)
    bench_totals['BA'] = round(bench_totals['H']/bench_totals['AB'],3)
    bench_totals['ERA'] = round(bench_totals['ER']/bench_totals['IP']*9,2)
    bench_totals['WHIP'] = round((bench_totals['BBa']+bench_totals['Ha'])/bench_totals['IP'],2)
    df = df.to_dict(orient='records')

        
    return {tm:{'roster':df, 'opt_totals':opt_totals, 'bench_totals':bench_totals}}
    #return {tm:{'pitcher_z':opt['pitcher_optimized_z'], 'pitcher_lineup':opt['pitcher_optimized_lineup'], 'hitter_z':opt['hitter_optimized_z'], 'hitter_lineup':opt['hitter_optimized_lineup']}}



@app.get('/trade', response_class=HTMLResponse)
async def trade_analyzer(request: Request):
    yr = datetime.now().year
    wk = pd.read_sql(f"SELECT max(week) week FROM projections WHERE year={yr}", engine).iloc[0]['week']-1
    df  = pd.read_sql(f"SELECT distinct p.CBSNAME Player, o.owner Owner, r.pos Decision, j.*, e.*, \
                    CASE WHEN e.DH>=5 THEN 'h' ELSE 'p' END As type \
                    FROM roster r \
                    INNER JOIN projections j On (j.cbsid=r.cbsid) \
                    INNER JOIN players p On (r.cbsid=p.cbsid) \
                    INNER JOIN owners o On (r.owner_id=o.owner_id) \
                    INNER JOIN (SELECT cbsid, all_pos, posC C, pos1B '1B', pos2B '2B', pos3B '3B', posSS SS, posOF OF, posDH DH, posSP SP, posRP RP, posP P FROM eligibility WHERE year=2024 and week={wk}) e On (r.cbsid=e.cbsid) \
            WHERE j.year={yr} AND j.week={wk} AND j.proj_type='ros' AND r.year={yr} AND r.week={wk} \
            ORDER BY Owner, year, week", engine)
    df['z'] = round(df['z'],1)
    df.fillna(0,inplace=True)
    
    teams = df.Owner.sort_values().unique().tolist()
    lg = df[df['Owner']!='Lima Time!'].sort_values(['Owner', 'type']).fillna(0)
    opt = optimize_team('Lima Time!', df)
    opt_pos = ['C', '1B', '2B', 'SS', '3B', 'MI', 'CI', 'OF1', 'OF2', 'OF3', 'OF4','OF5', 'DH1', 'DH2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
    bench = df[(df['Owner']=='Lima Time!') & (~df['Player'].isin(opt['hitter_lineup']+opt['pitcher_lineup']))].sort_values('type')['Player'].to_list()
    df = df[df['Owner']=='Lima Time!'].set_index('Player').reindex(opt['hitter_lineup']+opt['pitcher_lineup']+bench).reset_index()
    df['optimized_position'] = opt_pos[:df.shape[0]]
    df.loc[df['Player'].isin(opt['hitter_lineup']), 'opt_designation'] = 'starting_hitter'
    df.loc[df['Player'].isin(opt['pitcher_lineup']), 'opt_designation'] = 'starting_pitcher'
    df.loc[(~df['Player'].isin(opt['hitter_lineup'])) & (df['type']=='h'), 'opt_designation'] = 'bench_hitter'
    df.loc[(~df['Player'].isin(opt['pitcher_lineup'])) & (df['type']=='p'), 'opt_designation'] = 'bench_pitcher'
    ros_totals = df[df['Player'].isin(opt['hitter_lineup']+opt['pitcher_lineup'])][['HR', 'SB', 'R', 'RBI', 'H', 'AB', 'QS', 'SO', 'SvHld', 'IP', 'Ha', 'BBa', 'ER']].sum().to_dict()
    return templates.TemplateResponse('trade.html', {'request':request, 'data':df.to_dict(orient='records'),
                    'teams':teams, 'lima':opt, 'opt_pos':opt_pos, 'ros_totals': ros_totals,
                    'lg':lg.to_dict(orient='records'),
                    })