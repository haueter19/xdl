import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
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
    return {
        'pitcher_z': x.pitcher_optimized_z,
        'pitcher_lineup': list(x.pitcher_optimized_lineup.values()),
        'pitcher_positions': x.pitcher_optimized_lineup,
        'hitter_z': x.hitter_optimized_z,
        'hitter_lineup': list(x.hitter_optimized_lineup.values()),
        'hitter_positions': x.hitter_optimized_lineup,
    }


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

@app.get('/', response_class=HTMLResponse)
async def slash_route(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

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
    # Ensure all teams appear, even those with no drafted players yet
    all_owners = pd.DataFrame({'Owner': owner_list})
    owners_df = all_owners.merge(owners_df, on='Owner', how='left')
    owners_df.fillna({'Drafted':0, 'Paid':0, 'z':0, 'H':0, 'AB':0, 'HR':0, 'R':0, 'RBI':0, 'SB':0, 'Outs':0, 'QS':0, 'SO':0, 'SvHld':0, 'ER':0, 'IP':0, 'BB':0, 'HA':0}, inplace=True)
    owners_df['Paid'] = owners_df['Paid'].apply(lambda x: int(x) if x>0 else x)
    owners_df['$/unit'] = round(owners_df['Paid']/owners_df['z'].replace(0, np.nan),1).fillna(0)
    owners_df['z'] = round(owners_df['z'],1)
    owners_df['$ Left'] = tm_dollars - owners_df['Paid']
    owners_df['$ Left / Plyr'] = round(owners_df['$ Left'] / (tm_players -owners_df['Drafted']).replace(0, np.nan),1).fillna(0)
    owners_df['max_bid'] = owners_df['$ Left'] - (tm_players - owners_df['Drafted'])
    total_paid = owners_df['Paid'].sum()
    total_drafted = owners_df['Drafted'].sum()
    avg_dollars_per_player = (tot_dollars - total_paid + owners_df['Paid']) / (tot_players - total_drafted + owners_df['Drafted']).replace(0, np.nan)
    owners_df['Cash'] = round(owners_df['$ Left / Plyr'] / avg_dollars_per_player, 2).fillna(1.0)
    owners_df['Value'] = round((owners_df['z']*orig_conv) - owners_df['Paid'],1)
    owners_df['BA'] = round(owners_df['H']/owners_df['AB'].replace(0, np.nan),3).fillna(0)
    owners_df['ERA'] = round(owners_df['ER']/(owners_df['Outs'].replace(0, np.nan)/3)*9,2).fillna(0)
    owners_df['WHIP'] = round((owners_df['BB']+owners_df['HA'])/(owners_df['Outs'].replace(0, np.nan)/3),2).fillna(0)
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
                                    'owned':h[h['Owner'].notna()], 'owners_df':owners_df.sort_values('Rank', ascending=False), 'roster':roster, 'roster_json':roster.to_json(orient='records'),
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
    o.fillna({'$ Left':260, 'max_bid':237, 'Drafted':0, '$ Left / Plyr':11.3, 'Cash':1.0, 'z':0, 'Paid':0}, inplace=True)
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



OPT_POS = ['C', '1B', '2B', 'SS', '3B', 'MI', 'CI', 'OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'DH1', 'DH2',
           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9',
           'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']

STAT_COLS = ['z', 'HR', 'SB', 'R', 'RBI', 'H', 'AB', 'W', 'QS', 'SO', 'Sv+Hld', 'IP', 'Ha', 'BBa', 'ER']


def load_roster_data():
    yr = datetime.now().year-1
    wk = pd.read_sql(f"SELECT max(week) week FROM projections WHERE year={yr}", engine).iloc[0]['week'] - 1
    df = pd.read_sql(f"SELECT distinct p.CBSNAME Player, o.owner Owner, r.pos Decision, j.*, e.*, \
                    CASE WHEN e.DH>=5 THEN 'h' ELSE 'p' END As type \
                    FROM roster r \
                    INNER JOIN projections j On (j.cbsid=r.cbsid) \
                    INNER JOIN players p On (r.cbsid=p.cbsid) \
                    INNER JOIN owners o On (r.owner_id=o.owner_id) \
                    INNER JOIN (SELECT cbsid, all_pos, posC C, pos1B '1B', pos2B '2B', pos3B '3B', posSS SS, posOF OF, posDH DH, posSP SP, posRP RP, posP P FROM eligibility WHERE year=2024 and week={wk}) e On (r.cbsid=e.cbsid) \
            WHERE j.year={yr} AND j.week={wk} AND j.proj_type='ros' AND r.year={yr} AND r.week={wk} \
            ORDER BY Owner, year, week", engine)
    df['z'] = round(df['z'], 1)
    df.fillna(0, inplace=True)
    return df


def build_optimized_roster(tm, df, opt=None):
    if opt is None:
        opt = optimize_team(tm, df)
    starters = opt['hitter_lineup'] + opt['pitcher_lineup']
    bench = df[(df['Owner'] == tm) & (~df['Player'].isin(starters))].sort_values('type')['Player'].to_list()
    tdf = df[df['Owner'] == tm].set_index('Player').reindex(starters + bench).reset_index()
    tdf['optimized_position'] = OPT_POS[:tdf.shape[0]]
    tdf.loc[tdf['Player'].isin(opt['hitter_lineup']), 'opt_designation'] = 'starting_hitter'
    tdf.loc[tdf['Player'].isin(opt['pitcher_lineup']), 'opt_designation'] = 'starting_pitcher'
    tdf.loc[(~tdf['Player'].isin(opt['hitter_lineup'])) & (tdf['type'] == 'h'), 'opt_designation'] = 'bench_hitter'
    tdf.loc[(~tdf['Player'].isin(opt['pitcher_lineup'])) & (tdf['type'] == 'p'), 'opt_designation'] = 'bench_pitcher'
    return tdf, opt


def calc_totals(tdf, opt):
    starters = opt['hitter_lineup'] + opt['pitcher_lineup']
    opt_totals = tdf[tdf['Player'].isin(starters)][STAT_COLS].sum()
    opt_totals['z'] = round(opt_totals['z'], 1)
    opt_totals['hitter_z'] = round(tdf[tdf['Player'].isin(opt['hitter_lineup'])]['z'].sum(), 1)
    opt_totals['pitcher_z'] = round(tdf[tdf['Player'].isin(opt['pitcher_lineup'])]['z'].sum(), 1)
    opt_totals['BA'] = round(opt_totals['H'] / opt_totals['AB'], 3) if opt_totals['AB'] > 0 else 0
    opt_totals['ERA'] = round(opt_totals['ER'] / opt_totals['IP'] * 9, 2) if opt_totals['IP'] > 0 else 0
    opt_totals['WHIP'] = round((opt_totals['BBa'] + opt_totals['Ha']) / opt_totals['IP'], 2) if opt_totals['IP'] > 0 else 0
    bench_totals = tdf[~tdf['Player'].isin(starters)][STAT_COLS].sum()
    bench_totals['z'] = round(bench_totals['z'], 1)
    bench_totals['BA'] = round(bench_totals['H'] / bench_totals['AB'], 3) if bench_totals['AB'] > 0 else 0
    bench_totals['ERA'] = round(bench_totals['ER'] / bench_totals['IP'] * 9, 2) if bench_totals['IP'] > 0 else 0
    bench_totals['WHIP'] = round((bench_totals['BBa'] + bench_totals['Ha']) / bench_totals['IP'], 2) if bench_totals['IP'] > 0 else 0
    return opt_totals.to_dict(), bench_totals.to_dict()


@app.get('/optimize')
async def optimize(tm: str):
    df = load_roster_data()
    tdf, opt = build_optimized_roster(tm, df)
    opt_totals, bench_totals = calc_totals(tdf, opt)
    return {tm: {'roster': tdf.to_dict(orient='records'), 'opt_totals': opt_totals, 'bench_totals': bench_totals}}


@app.get('/trade', response_class=HTMLResponse)
async def trade_analyzer(request: Request):
    df = load_roster_data()
    teams = df.Owner.sort_values().unique().tolist()
    return templates.TemplateResponse('trade.html', {'request': request, 'teams': teams})


class TradeRequest(BaseModel):
    team1: str
    team2: str
    team1_players: list
    team2_players: list


@app.post('/simulate_trade')
async def simulate_trade(trade: TradeRequest):
    df = load_roster_data()

    # Pre-trade optimization for both teams
    tdf1_before, opt1_before = build_optimized_roster(trade.team1, df)
    tdf2_before, opt2_before = build_optimized_roster(trade.team2, df)
    totals1_before, _ = calc_totals(tdf1_before, opt1_before)
    totals2_before, _ = calc_totals(tdf2_before, opt2_before)

    # Swap players
    df_after = df.copy()
    df_after.loc[df_after['cbsid'].isin(trade.team1_players), 'Owner'] = trade.team2
    df_after.loc[df_after['cbsid'].isin(trade.team2_players), 'Owner'] = trade.team1

    # Post-trade optimization for both teams
    tdf1_after, opt1_after = build_optimized_roster(trade.team1, df_after)
    tdf2_after, opt2_after = build_optimized_roster(trade.team2, df_after)
    totals1_after, _ = calc_totals(tdf1_after, opt1_after)
    totals2_after, _ = calc_totals(tdf2_after, opt2_after)

    return {
        'team1': {
            'name': trade.team1,
            'before': totals1_before,
            'after': totals1_after,
            'roster': tdf1_after.to_dict(orient='records'),
        },
        'team2': {
            'name': trade.team2,
            'before': totals2_before,
            'after': totals2_after,
            'roster': tdf2_after.to_dict(orient='records'),
        },
    }


# ── Admin ─────────────────────────────────────────────────────────────────────

_PLAYERS_COLS = None

def _get_players_cols():
    """Return the set of valid column names for the players table (cached)."""
    global _PLAYERS_COLS
    if _PLAYERS_COLS is None:
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(players)"))
            _PLAYERS_COLS = {row[1] for row in result.fetchall()}
    return _PLAYERS_COLS


class NewPlayer(BaseModel):
    cbsid: int
    CBSNAME: str
    PLAYERNAME: Optional[str] = None
    IDFANGRAPHS: Optional[str] = None
    IDFANGRAPHS_minors: Optional[str] = None


class PlayerUpdate(BaseModel):
    data: dict


@app.get('/admin', response_class=HTMLResponse)
async def admin_view(request: Request):
    return templates.TemplateResponse('admin.html', {'request': request})


@app.post('/admin/players', status_code=201)
async def add_player(player: NewPlayer):
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT cbsid FROM players WHERE cbsid = :cbsid"),
            {'cbsid': player.cbsid}
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail=f"Player with cbsid {player.cbsid} already exists")
        conn.execute(
            text("INSERT INTO players (cbsid, CBSNAME, PLAYERNAME, IDFANGRAPHS, IDFANGRAPHS_minors) "
                 "VALUES (:cbsid, :CBSNAME, :PLAYERNAME, :IDFANGRAPHS, :IDFANGRAPHS_minors)"),
            player.model_dump()
        )
        conn.commit()
    return {'status': 'ok', 'cbsid': player.cbsid}


@app.get('/admin/players')
async def search_players(cbsid: Optional[str] = None, name: Optional[str] = None):
    conditions, params = [], {}
    if cbsid:
        conditions.append("cbsid = :cbsid")
        params['cbsid'] = cbsid
    if name:
        conditions.append("UPPER(CBSNAME) LIKE :name")
        params['name'] = f"%{name.upper()}%"
    if not conditions:
        return []
    where = " AND ".join(conditions)
    df = pd.read_sql(text(f"SELECT * FROM players WHERE {where} LIMIT 50"), engine, params=params)
    return json.loads(df.to_json(orient='records'))


@app.put('/admin/players/{cbsid}')
async def update_player(cbsid: int, update: PlayerUpdate):
    if not update.data:
        raise HTTPException(status_code=400, detail="No data provided")
    allowed = _get_players_cols()
    set_parts, params = [], {'cbsid': cbsid}
    for col, val in update.data.items():
        if col == 'cbsid' or col not in allowed:
            continue
        set_parts.append(f'"{col}" = :{col}')
        params[col] = val if val != '' else None
    if not set_parts:
        raise HTTPException(status_code=400, detail="No valid columns to update")
    with engine.connect() as conn:
        conn.execute(text(f"UPDATE players SET {', '.join(set_parts)} WHERE cbsid = :cbsid"), params)
        conn.commit()
    return {'status': 'ok', 'updated': list(update.data.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# SEASON ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
import ast as _ast

_analysis_cache: dict = {}
ANALYSIS_YEARS = [2025, 2024, 2023, 2022]
_ANALYSIS_DEFAULT_YEAR = datetime.now().year - 1

_ANALYSIS_HITTER_POS = {'C', '1B', '2B', '3B', 'SS', 'OF', 'DH', 'MI', 'CI'}
_ANALYSIS_N_TEAMS    = 12
_ANALYSIS_TM_DOLLARS = 260
_ANALYSIS_REPL_H     = _ANALYSIS_N_TEAMS * 14   # 168
_ANALYSIS_REPL_P     = _ANALYSIS_N_TEAMS * 9    # 108

# Short name (as stored in players20XX.Owner) → owner_id
_PNAME_TO_OID: dict[int, dict] = {
    2025: {
        '9 Grand Kids': 30, 'Brewbirds': 3,  'Charmer': 36, 'Dirty Birds': 41,
        'Harvey': 4,        'Lima Time': 38, 'Mother': 47,  'Roid Ragers': 44,
        'Trouble': 1,       'Ugly Spuds': 29,'Wu-Tang': 42, 'Young Guns': 45,
    },
    2024: {
        'Brewbirds': 3,  'Charmer': 36,  'Dirty Birds': 41, 'Harvey': 4,
        'Lima Time': 38, 'Madness': 30,  'Mother': 47,      'Roid Ragers': 44,
        'Trouble': 1,    'Ugly Spuds': 29,'Wu-Tang': 42,    'Young Guns': 45,
    },
    2023: {
        'Brewbirds': 3,  'Charmer': 36,  'Dirty Birds': 41, 'Harvey': 4,
        'Lil Trump': 27, 'Lima Time': 38, 'Midnight': 30,   'Roid Ragers': 44,
        'Trouble': 1,    'Ugly Spuds': 29,'Wu-Tang': 42,    'Young Guns': 45,
    },
}

# CSV owner names → owner_id (for 2025 adds/trades CSVs)
_CSV_TO_OID_2025 = {
    '9 Grand Kids': 30,           'Brewbirds': 3,            'Charmer': 36,
    'Dirty Birds': 41,            'Harveys Wallbangers': 4,
    'Lil Trumps Wiscompton Wu-Tang': 42,  'Wiscompton Wu-Tang': 42,
    'Lima Time!': 38,             "Mom's Cookin": 47,
    'Roid Ragers': 44,            'Trouble With The Curve': 1,
    'Trouble with the Curve': 1,  'Ugly Spuds': 29, 'Young Guns': 45,
}

# Full canonical name → owner_id (for trades_2023.json partner names)
_FULLNAME_TO_OID = {
    '9 Grand Kids': 30,            'Brewbirds': 3,
    'Charmer': 36,                 'Dirty Birds': 41,
    'Harveys Wallbangers': 4,      'Lil Trump & the Ivanabees': 27,
    'Lima Time!': 38,              "Mom's Cookin": 47,
    'Roid Ragers': 44,             'Trouble with the Curve': 1,
    'Wiscompton Wu-Tang': 42,      'Young Guns': 45,
}


def _load_players_year(year: int) -> pd.DataFrame:
    """Return players20XX with normalized columns and resolved cbsid."""
    HPOS = _ANALYSIS_HITTER_POS
    if year == 2025:
        df = pd.read_sql("""
            SELECT CAST(cbsid AS INT) cbsid, Name, COALESCE(Pos,'') Pos,
                   Primary_Pos, z, Value, Owner,
                   COALESCE(Paid,0) Paid, COALESCE(Supp,0) Supp,
                   COALESCE(AB,0) AB, COALESCE(R,0) R, COALESCE(HR,0) HR,
                   COALESCE(RBI,0) RBI, COALESCE(SB,0) SB, COALESCE(H,0) H,
                   COALESCE(IP,0) IP, COALESCE(ERA,0) ERA, COALESCE(WHIP,0) WHIP,
                   COALESCE(SO,0) SO, COALESCE(QS,0) QS, COALESCE(SvHld,0) SvHld,
                   COALESCE(ER,0) ER, COALESCE(HA,0) HA, COALESCE(BB,0) BB
            FROM players2025""", engine)
    else:
        raw = pd.read_sql(f"SELECT * FROM players{year}", engine)
        name_cbsid = pd.read_sql(
            "SELECT CAST(cbsid AS BIGINT) cbsid, CBSNAME FROM players", engine)
        raw = raw.merge(name_cbsid, left_on='Name', right_on='CBSNAME', how='left')
        if 'Sv+Hld' in raw.columns:
            raw.rename(columns={'Sv+Hld': 'SvHld'}, inplace=True)
        if 'Supp' not in raw.columns:
            raw['Supp'] = 0
        # 2023 uses Wins (W) not Quality Starts; map W→QS for uniform handling
        if 'QS' not in raw.columns:
            raw['QS'] = raw.get('W', pd.Series(0, index=raw.index)).fillna(0)
        for col in ['AB','R','HR','RBI','SB','H','IP','ERA','WHIP',
                    'SO','QS','SvHld','ER','HA','BB','Paid','Supp']:
            if col not in raw.columns:
                raw[col] = 0
            raw[col] = raw[col].fillna(0)
        if 'Pos' not in raw.columns:
            raw['Pos'] = raw.get('Primary_Pos', '')
        raw['Pos'] = raw['Pos'].fillna('')
        df = raw[['cbsid','Name','Pos','Primary_Pos','z','Value','Owner','Paid','Supp',
                  'AB','R','HR','RBI','SB','H','IP','ERA','WHIP',
                  'SO','QS','SvHld','ER','HA','BB']].copy()
    df['cbsid'] = pd.to_numeric(df['cbsid'], errors='coerce')
    df = df.dropna(subset=['cbsid'])
    df['cbsid'] = df['cbsid'].astype(int)
    df['ptype'] = df['Primary_Pos'].apply(
        lambda x: 'h' if pd.notna(x) and x in HPOS else 'p')
    return df


def compute_analysis(year: int) -> dict:
    """Full transaction analysis for one season. Results are in-memory cached."""
    if year in _analysis_cache:
        return _analysis_cache[year]

    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'player_projections'))
    from fantasy_projections import FantasyProjections as _FP

    owners_tbl = pd.read_sql('SELECT owner_id, owner FROM owners', engine)
    oid_to_name = dict(zip(owners_tbl['owner_id'], owners_tbl['owner']))
    name_to_oid = _PNAME_TO_OID[year]

    # ── Load players & ptype ────────────────────────────────────────────────
    all_proj = _load_players_year(year)

    # cbsid→Primary_Pos lookup for disambiguation
    _pos_proj = all_proj[['cbsid', 'Primary_Pos']].drop_duplicates('cbsid')
    # Name→cbsid map (no dedup on CBSNAME so we can see duplicates)
    pname_map_full = pd.read_sql("SELECT CAST(cbsid AS INT) cbsid, CBSNAME FROM players", engine)
    pname_map = pname_map_full.drop_duplicates('CBSNAME')  # deduped (one cbsid per name)

    # ── Position-aware cbsid lookup (fixes Jose Ramirez / same-name players) ──
    def _lookup_cbsid(name: str, pos: str) -> Optional[int]:
        """Look up cbsid by name; use Position string to break ties."""
        matches = pname_map_full[pname_map_full['CBSNAME'] == name]
        if len(matches) == 0:
            return None
        if len(matches) == 1:
            return int(matches['cbsid'].iloc[0])
        # Multiple cbsids for same name — disambiguate by hitter vs pitcher
        with_pos = matches.merge(_pos_proj, on='cbsid', how='left')
        parsed_hitter = bool({p.strip() for p in pos.split(',')} & _ANALYSIS_HITTER_POS)
        if parsed_hitter:
            filt = with_pos[with_pos['Primary_Pos'].isin(_ANALYSIS_HITTER_POS)]
        else:
            filt = with_pos[~with_pos['Primary_Pos'].isin(_ANALYSIS_HITTER_POS).fillna(False)]
        return int(filt['cbsid'].iloc[0]) if len(filt) else int(matches['cbsid'].iloc[0])

    # Initialize FantasyProjections (qualifiers will be derived from year-1 CSV stats)
    fp = _FP(
        db_path='fantasy_data.db',
        data_dir='player_projections/data',
        year=year,
        n_teams=_ANALYSIS_N_TEAMS,
        roster_size=23,
        budget_per_team=_ANALYSIS_TM_DOLLARS,
    )

    # ── Season stats (W→QS for pre-2025) ───────────────────────────────────
    raw_stats = pd.read_sql(
        f"SELECT cbsid, week, R, RBI, HR, SB, H, AB, SO, SvHld, ER, Ha, BBa, "
        f"IP, outs, COALESCE(QS,0) QS, COALESCE(W,0) W FROM stats WHERE year={year}", engine)
    if year <= 2024:
        raw_stats['QS'] = raw_stats['W']
    SCOLS = ['R','RBI','HR','SB','H','AB','SO','SvHld','ER','Ha','BBa','IP','outs','QS']

    p_meta = all_proj[['cbsid','Name','ptype','Primary_Pos']].drop_duplicates('cbsid')
    st = raw_stats.groupby('cbsid')[SCOLS].sum(min_count=1).reset_index()
    st = st.merge(p_meta, on='cbsid', how='left').fillna({'ptype': 'p', 'Primary_Pos': 'SP', 'Name': ''})

    # Persistent two-way player fix (e.g. Ohtani)
    reclass = (
        (st['ptype'] == 'p') &
        (st['IP'].fillna(0) < 5) &
        (st['AB'].fillna(0) >= 50)
    )
    st.loc[reclass, 'ptype'] = 'h'
    st.loc[reclass, 'Primary_Pos'] = 'DH'

    # Prepare for FantasyProjections (rename Ha→HA, BBa→BB, add PA)
    st_fp = st.rename(columns={'Ha': 'HA', 'BBa': 'BB'}).copy()
    st_fp['PA'] = st_fp['AB'] + st_fp['BB']
    st_hit_fp = st_fp[st_fp['ptype'] == 'h'].copy()
    st_pit_fp = st_fp[st_fp['ptype'] == 'p'].copy()

    # Evaluate full-season performance; qualifiers derived from year-1 CSV stats
    season_vals = fp.evaluate_season_performance(st_hit_fp, st_pit_fp)
    conv = fp._last_conversion_factor

    # Replacement levels (for Q2 post-trade value computation)
    hz = season_vals[season_vals['ptype'] == 'h']['total_z'].sort_values(ascending=False)
    pz = season_vals[season_vals['ptype'] == 'p']['total_z'].sort_values(ascending=False)
    rl_h = float(hz.iloc[_ANALYSIS_REPL_H - 1]) if len(hz) >= _ANALYSIS_REPL_H else float(hz.min())
    rl_p = float(pz.iloc[_ANALYSIS_REPL_P - 1]) if len(pz) >= _ANALYSIS_REPL_P else float(pz.min())

    # Map values back to st by cbsid
    val_map = season_vals.set_index('cbsid')['Value'].to_dict()
    st['actual_value'] = st['cbsid'].map(val_map).fillna(0)

    # ── Roster + weeks-per-team ─────────────────────────────────────────────
    roster = pd.read_sql(
        f"SELECT cbsid, owner_id, week FROM roster WHERE year={year}", engine)
    first_wk = (roster.groupby(['cbsid','owner_id'])['week'].min()
                .reset_index().rename(columns={'week': 'first_week'}))
    wk = (roster.groupby(['cbsid','owner_id']).size().reset_index(name='weeks')
          .merge(roster.groupby('cbsid')['week'].count().reset_index(name='total_weeks'),
                 on='cbsid'))
    wk['fraction'] = wk['weeks'] / wk['total_weeks']
    wk = wk.merge(st[['cbsid','actual_value']], on='cbsid', how='left')
    wk['team_value'] = (wk['actual_value'].fillna(0) * wk['fraction']).round(2)
    wk = wk.merge(first_wk, on=['cbsid','owner_id'], how='left')

    # ── Acquisition source classification ───────────────────────────────────
    if year == 2025:
        tr_csv = pd.read_csv('2025-trades.csv')
        tr_csv['owner_id'] = tr_csv['owner'].map(_CSV_TO_OID_2025)
        # Fix 2: Use _lookup_cbsid with position disambiguation for all trade players
        traded_pairs: set = set()
        for _, row in tr_csv.iterrows():
            if pd.isna(row.get('owner_id')):
                continue
            recv_oid = int(row['owner_id'])
            try:
                for p in _ast.literal_eval(row['parsed_transactions']):
                    cid = _lookup_cbsid(p['Name'], p.get('Position', ''))
                    if cid is not None:
                        traded_pairs.add((cid, recv_oid))
            except Exception:
                pass

        def _classify(cbsid, owner_id, fw):
            if fw == 1: return 'Draft'
            if (int(cbsid), int(owner_id)) in traded_pairs: return 'Trade'
            return 'FA'
        sources = ['Draft', 'FA', 'Trade']
    else:
        def _classify(cbsid, owner_id, fw):
            return 'Draft' if fw == 1 else 'In-Season'
        sources = ['Draft', 'In-Season']

    wk['source'] = wk.apply(
        lambda r: _classify(r['cbsid'], r['owner_id'], r['first_week']), axis=1)
    wk['owner'] = wk['owner_id'].map(oid_to_name)

    # ── Q1: Value by source ─────────────────────────────────────────────────
    q1_raw = wk.groupby(['owner','source'])['team_value'].sum().unstack(fill_value=0)
    for s in sources:
        if s not in q1_raw.columns:
            q1_raw[s] = 0.0
    q1_raw['Total'] = q1_raw[sources].sum(axis=1)
    pos_denom = q1_raw[sources].clip(lower=0).sum(axis=1).replace(0, np.nan)
    q1_out = []
    for owner in q1_raw.index:
        row = {'owner': owner, 'total': round(float(q1_raw.loc[owner,'Total']), 1)}
        for s in sources:
            val = float(q1_raw.loc[owner, s])
            pct = (max(val, 0) / float(pos_denom[owner])
                   if pd.notna(pos_denom[owner]) else 0) * 100
            key = s.lower().replace('-','_').replace(' ','_')
            row[key]          = round(val, 1)
            row[f'{key}_pct'] = round(pct, 1)
        q1_out.append(row)
    q1_out.sort(key=lambda x: -x['total'])

    # ── Q2: Trade winners ───────────────────────────────────────────────────
    rs = roster.merge(raw_stats, on=['cbsid','week'], how='left')
    _ptype_map = dict(zip(p_meta['cbsid'], p_meta['ptype']))

    def _player_post_value(cbsid: int, recv_oid: int, from_week: int) -> float:
        mask = ((rs['cbsid'] == cbsid) & (rs['owner_id'] == recv_oid)
                & (rs['week'] >= from_week))
        if not mask.any():
            return 0.0
        ps    = rs[mask][SCOLS].sum()
        ptype = _ptype_map.get(cbsid, 'p')
        # Apply two-way player logic to partial-season stats too
        if ptype == 'p' and float(ps.get('IP', 0)) < 5 and float(ps.get('AB', 0)) >= 50:
            ptype = 'h'
        rdf = pd.DataFrame([ps])
        rdf['cbsid'] = cbsid
        rdf = rdf.rename(columns={'Ha': 'HA', 'BBa': 'BB'})
        rdf['PA'] = rdf['AB'] + rdf['BB']
        rdf['Primary_Pos'] = 'DH' if ptype == 'h' else 'SP'
        rdf['Name'] = ''
        rdf['ptype'] = ptype
        try:
            if ptype == 'h':
                z_df = fp.calculate_z_scores(rdf, 'hitting')
                return round((float(z_df['total_z'].iloc[0]) - rl_h) * conv, 1)
            else:
                z_df = fp.calculate_z_scores(rdf, 'pitching')
                return round((float(z_df['total_z'].iloc[0]) - rl_p) * conv, 1)
        except Exception:
            return 0.0

    q2_out = None
    if year == 2025:
        tr_csv = pd.read_csv('2025-trades.csv')
        tr_csv['owner_id'] = tr_csv['owner'].map(_CSV_TO_OID_2025)
        events = []
        for _, row in tr_csv.iterrows():
            if pd.isna(row.get('owner_id')):
                continue
            recv_oid = int(row['owner_id'])
            try:
                for p in _ast.literal_eval(row['parsed_transactions']):
                    # Fix 2: position-aware cbsid lookup
                    cid = _lookup_cbsid(p['Name'], p.get('Position', ''))
                    events.append({
                        'effective': row['Effective'],
                        'period':    int(row['period']),
                        'recv_oid':  recv_oid,
                        'send_name': p.get('Trade_Partner', ''),
                        'player':    p['Name'],
                        'cbsid':     cid,
                    })
            except Exception:
                pass
        ev_df = pd.DataFrame(events).dropna(subset=['cbsid'])
        ev_df['cbsid']    = ev_df['cbsid'].astype(int)
        ev_df['send_oid'] = ev_df['send_name'].map(_CSV_TO_OID_2025)
        ev_df['team_pair'] = ev_df.apply(
            lambda r: tuple(sorted([str(r['recv_oid']), str(r.get('send_oid', ''))])), axis=1)
        unique_ev = (ev_df[['effective','period','team_pair']].drop_duplicates()
                     .sort_values('effective').reset_index(drop=True))
        q2_trades = []
        for _, ut in unique_ev.iterrows():
            mask = ((ev_df['effective'] == ut['effective'])
                    & (ev_df['team_pair'] == ut['team_pair']))
            by_recv: dict = {}
            for _, tr in ev_df[mask].iterrows():
                oid = tr['recv_oid']
                by_recv.setdefault(oid, {'cbsids': [], 'period': tr['period'], 'players': []})
                by_recv[oid]['cbsids'].append(int(tr['cbsid']))
                by_recv[oid]['players'].append(tr['player'])
            if len(by_recv) < 2:
                continue
            sides = []
            for oid, info in by_recv.items():
                unique_cbsids = list(dict.fromkeys(info['cbsids']))  # deduplicate
                val = sum(_player_post_value(c, oid, info['period']) for c in unique_cbsids)
                sides.append({'team': oid_to_name.get(oid, str(oid)),
                              'players': ', '.join(dict.fromkeys(info['players'])),
                              'value': round(val, 1)})
            sides.sort(key=lambda x: -x['value'])
            q2_trades.append({'date': ut['effective'],
                              'winner': sides[0]['team'], 'sides': sides})
        q2_out = q2_trades

    elif year == 2023:
        with open('trades_2023.json') as f:
            t23 = json.load(f)
        seen: dict = {}
        for recv_name, trade_list in t23.items():
            recv_oid = _FULLNAME_TO_OID.get(recv_name)
            if recv_oid is None:
                continue
            for trade in trade_list:
                week      = trade['week']
                send_name = trade['partner']
                send_oid  = _FULLNAME_TO_OID.get(send_name)
                key = (week, frozenset([recv_oid, send_oid or 0]))
                if key not in seen:
                    seen[key] = {
                        'week': week,
                        'teams': ({recv_oid: trade['received'], send_oid: trade['traded']}
                                  if send_oid else {recv_oid: trade['received']}),
                    }
        q2_23 = []
        for (week, _), evt in sorted(seen.items()):
            sides = []
            for recv_oid, cbsids in evt['teams'].items():
                if recv_oid is None:
                    continue
                val = sum(_player_post_value(c, recv_oid, week) for c in cbsids)
                names_list = [
                    (p_meta[p_meta['cbsid'] == cid]['Name'].iloc[0]
                     if len(p_meta[p_meta['cbsid'] == cid]) else str(cid))
                    for cid in cbsids
                ]
                sides.append({'team': oid_to_name.get(recv_oid, str(recv_oid)),
                              'players': ', '.join(names_list),
                              'value': round(val, 1)})
            if len(sides) < 2:
                continue
            sides.sort(key=lambda x: -x['value'])
            q2_23.append({'date': f'Week {week}', 'winner': sides[0]['team'], 'sides': sides})
        q2_out = q2_23

    # ── Q3: Draft surplus (Fix 4: full-season value, no pro-rating) ─────────
    # Use the complete season actual_value attributed to the drafting team only.
    # No fractions — whether a player was later traded or dropped doesn't affect this.
    owned = all_proj[all_proj['Owner'].notna()].copy()
    owned['owner_id'] = owned['Owner'].map(name_to_oid)
    owned = owned.dropna(subset=['cbsid', 'owner_id'])
    owned['cbsid']    = owned['cbsid'].astype(int)
    owned['owner_id'] = owned['owner_id'].astype(int)

    # drafted_rows: (cbsid, owner_id) pairs where that owner drafted the player
    drafted_rows = (wk[wk['source'] == 'Draft'][['cbsid','owner_id']]
                    .drop_duplicates('cbsid'))
    # full-season value per player (FantasyProjections-derived)
    full_vals = st[['cbsid','actual_value']].copy()

    drafted = owned.merge(drafted_rows, on=['cbsid','owner_id'], how='inner')
    drafted = drafted.merge(full_vals, on='cbsid', how='left')
    drafted['actual_value'] = drafted['actual_value'].fillna(0)
    drafted['surplus']      = (drafted['actual_value'] - drafted['Paid']).round(1)
    drafted['team_name']    = drafted['owner_id'].map(oid_to_name)
    if 'Pos' not in drafted.columns:
        drafted['Pos'] = drafted.get('Primary_Pos', pd.Series('', index=drafted.index))

    team_q3 = (drafted.groupby('team_name')
               .agg(total_paid=('Paid','sum'), proj_value=('Value','sum'),
                    act_value=('actual_value','sum'), surplus=('surplus','sum'))
               .round(1).reset_index()
               .sort_values('surplus', ascending=False)
               .to_dict(orient='records'))

    auction = drafted[drafted['Paid'] > 0].copy()

    def _to_records(df):
        d = df.copy()
        if 'Pos' not in d.columns:
            d['Pos'] = d['Primary_Pos'].fillna('') if 'Primary_Pos' in d.columns else ''
        return (d[['Name','team_name','Pos','Paid','actual_value','surplus']]
                .rename(columns={'team_name': 'owner', 'actual_value': 'team_value'})
                .round(1).to_dict(orient='records'))

    result = {
        'year':    year,
        'sources': sources,
        'conv':    round(conv, 2),
        'q1':      q1_out,
        'q2':      q2_out,
        'q3': {
            'teams':         team_q3,
            'players_best':  _to_records(auction.nlargest(50, 'surplus')),
            'players_worst': _to_records(auction.nsmallest(30, 'surplus')),
            'all_players':   _to_records(auction),
        },
    }
    _analysis_cache[year] = result
    return result


@app.get('/analysis', response_class=HTMLResponse)
async def analysis_view(request: Request):
    default_yr = min(_ANALYSIS_DEFAULT_YEAR, max(ANALYSIS_YEARS))
    return templates.TemplateResponse(
        'analysis.html',
        {'request': request, 'years': ANALYSIS_YEARS, 'default_year': default_yr})


@app.get('/analysis/data')
async def analysis_data(year: int = datetime.now().year - 1):
    if year not in ANALYSIS_YEARS:
        raise HTTPException(status_code=400, detail=f"Year must be one of {ANALYSIS_YEARS}")
    try:
        data = compute_analysis(year)
        return json.loads(json.dumps(data, default=str))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())