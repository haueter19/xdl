import pandas as pd
import math
from datetime import datetime
from sqlalchemy import create_engine

tm_dollars = 260
tm_players = 23
engine = create_engine('sqlite:///fantasy_data.db', echo=False)

class Fantasy_Projections():
    def __init__(self, yr=datetime.now().year, n_teams=12, tm_players=23, tm_dollars=260, player_split=.6) -> None:
        self.yr = yr
        self.n_teams = n_teams
        self.tm_players = tm_players
        self.tm_dollars = tm_dollars
        self.player_split = player_split
        self.pitcher_split = 1 - player_split
        self.tot_dollars = n_teams * tm_dollars
        self.tot_players = n_teams * tm_players
        self.tot_hitters = n_teams * 14
        self.tot_pitchers = n_teams * 9
        self.drafted_by_pos = {
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
        self.quals = None
        #self.qual_p = None
        self.proj_systems = ['atc', 'thebatx', 'dc', 'steamer', 'zips']
        self.pos_hierarchy = ['C', '2B', '3B', 'OF', 'SS', '1B', 'DH', 'SP', 'RP', 'P']
        self.keepers_url = 'https://docs.google.com/spreadsheets/d/1dwDC2uMsfVRYeDECKLI0Mm_QonxkZvTkZTfBgnZo7-Q/edit#gid=1723951361'

    def load_id_map(self):
        player_id_url = 'https://docs.google.com/spreadsheets/d/1JgczhD5VDQ1EiXqVG-blttZcVwbZd5_Ne_mefUGwJnk/pubhtml?gid=0&single=true'
        ids = pd.read_html(player_id_url, header=1)[0]
        ids.drop(columns=['1', 'Unnamed: 9'], inplace=True)
        ids = ids[ids['PLAYERNAME'].notna()]
        self.ids = ids
        return ids

    def find_primary_pos(self, p):
        pos_list = p.split('/')
        for i in self.pos_hierarchy:
            if i in pos_list:
                return i

    def get_qual_avgs(self, yr):
        final_h = pd.read_csv('data/'+str(yr)+'-final-stats-h.csv').sort_values('PA', ascending=False)
        final_h = final_h[final_h['PA']>440]
        lgBA = final_h['H'].sum()/final_h['AB'].sum()
        final_h['zlgBA'] = final_h.apply(lambda x: x['H']-(x['AB']*(lgBA)), axis=1)
        quals_h = final_h[['H', 'AB', 'PA', 'G', 'zlgBA', 'R', 'RBI', 'HR', 'SB']].describe().to_dict()
        
        final_p = pd.read_csv('data/'+str(yr)+'-final-stats-p.csv').sort_values('IP', ascending=False)
        final_p['playerid'] = final_p['playerid'].astype(str)
        final_p['Sv+Hld'] = final_p['SV']+final_p['HLD']
        val_p = pd.read_csv('data/'+str(yr)+'-fangraphs-auction-calculator-p.csv')
        val_p['playerid'] = val_p['playerid'].astype(str)
        val_p['Primary_Pos'] = val_p['POS'].apply(lambda x: self.find_primary_pos(x))
        final_p = final_p.merge(val_p[['playerid', 'Primary_Pos']], on='playerid', how='inner')
        final_p = final_p[(final_p['Primary_Pos']=='RP') & (final_p['IP'].between(48,90) & (final_p['Sv+Hld']>5)) | (final_p['Primary_Pos']=='SP') & (final_p['IP']>140)]
        lgERA = final_p['ER'].sum()/final_p['IP'].sum()*9
        lgWHIP = (final_p['BB'].sum()+final_p['H'].sum())/final_p['IP'].sum()
        final_p['zlgERA'] = final_p.apply(lambda x: ((x['ER']*9) - (x['IP']*lgERA))*-1, axis=1)
        final_p['zlgWHIP'] = final_p.apply(lambda x: ((x['H']+x['BB'])-(x['IP']*lgWHIP))*-1, axis=1)
        quals_p = final_p[['BB', 'H', 'ER', 'IP', 'SO', 'W', 'Sv+Hld', 'zlgERA', 'zlgWHIP']].describe().to_dict()
        quals_p['HA'] = quals_p.pop('H')
        quals_h.update(quals_p)
        self.quals = quals_h
        return quals_h


    def big_board(self, row, stat, qual):
        if stat == 'BA':
            ba_pts = row['H']-(row['AB']*(qual['H']['mean']/qual['AB']['mean']))
            zBA = (ba_pts-qual['zlgBA']['mean'])/qual['zlgBA']['std']
            #return ((row['AB'] * (((row['H']/row['AB'])-qual_avgs['AVG'][0])/qual_avgs['AVG'][1])) - qual_avgs['zlgBA'][0])/qual_avgs['zlgBA'][1]
            return zBA
        if stat == 'BA_ly':
            ba_pts = row['H_ly']-(row['AB_ly']*(qual['H']['mean']/qual['AB']['mean']))
            zBA = (ba_pts-qual['zlgBA']['mean'])/qual['zlgBA']['std']
            #return ((row['AB'] * (((row['H']/row['AB'])-qual_avgs['AVG'][0])/qual_avgs['AVG'][1])) - qual_avgs['zlgBA'][0])/qual_avgs['zlgBA'][1]
            return zBA
        elif stat=='ERA':
            pts = ((row['ER']*9) - ((row['IP']*qual['ER']['mean']*9)/qual['IP']['mean'])) * -1
            zERA = (pts-qual['zlgERA']['mean'])/qual['zlgERA']['std']
            return zERA
        elif stat=='ERA_ly':
            pts = ((row['ER_ly']*9) - ((row['IP_ly']*qual['ER']['mean']*9)/qual['IP']['mean'])) * -1
            zERA = (pts-qual['zlgERA']['mean'])/qual['zlgERA']['std']
            return zERA
        elif stat=='WHIP':
            pts = ((row['HA']+row['BB'])-(row['IP']*((qual['HA']['mean']+qual['BB']['mean'])/qual['IP']['mean']))) * -1
            zWHIP = (pts-qual['zlgWHIP']['mean'])/qual['zlgWHIP']['std']
            return zWHIP
        elif stat=='WHIP_ly':
            pts = ((row['HA_ly']+row['BB_ly'])-(row['IP_ly']*((qual['HA']['mean']+qual['BB']['mean'])/qual['IP']['mean']))) * -1
            zWHIP = (pts-qual['zlgWHIP']['mean'])/qual['zlgWHIP']['std']
            return zWHIP
        else:
            if stat[-3:] == '_ly':
                return (row[stat] - qual[stat[:-3]]['mean']) / qual[stat[:-3]]['std']
            else:
                return (row[stat] - qual[stat]['mean']) / qual[stat]['std']

    def calc_z(self, df, kind):
        if kind=='h':
            for stat in ['R', 'HR', 'RBI', 'SB', 'BA', 'R_ly', 'HR_ly', 'RBI_ly', 'SB_ly', 'BA_ly']:
                df['z'+stat] = df.apply(lambda row: self.big_board(row, stat, self.quals), axis=1)
            df['BIGAAh'] = df['zR']+df['zRBI']+df['zHR']+df['zSB']+df['zBA']
            df['z_h_ly'] = df['zR_ly']+df['zRBI_ly']+df['zHR_ly']+df['zSB_ly']+df['zBA_ly']
            return df
        else:
            for stat in ['ERA', 'WHIP', 'SO', 'W', 'Sv+Hld', 'ERA_ly', 'WHIP_ly', 'SO_ly', 'W_ly', 'Sv+Hld_ly']:
                df['z'+stat] = df.apply(lambda row: self.big_board(row, stat, self.quals), axis=1)
            df['BIGAAp'] = df['zERA']+df['zWHIP']+df['zW']+df['zSO']+df['zSv+Hld']
            df['z_p_ly'] = df['zERA_ly']+df['zWHIP_ly']+df['zW_ly']+df['zSO_ly']+df['zSv+Hld_ly']
            return df
        
    
    def make_projections(self, yr):
        proj_system_list = []
        for proj_file in self.proj_systems:
            try:
                temp = pd.read_csv('data/'+str(yr)+'-'+proj_file+'-proj-h.csv', encoding="latin-1")
                proj_system_list.append(temp)
            except:
                pass

        h = pd.concat(proj_system_list).sort_values('PlayerId')
        h.rename(columns={'PlayerId':'playerid'},inplace=True)
        # Add Fangraphs auction values
        try:
            val_h = pd.read_csv('data/'+str(yr)+'-fangraphs-auction-calculator-h.csv')
            val_h.rename(columns={'PlayerId':'playerid', 'POS':'Pos'},inplace=True)
            h = h.merge(val_h[['playerid', 'Pos', 'Dollars']])
        except:
            h['Dollars'] = 0
        # Add CBS auction values
        try:
            cbs = pd.read_csv('data/'+str(yr)+'-cbs-values.csv', encoding="latin-1")
            h = h.merge(cbs[['playerid', 'CBSNAME', 'CBS']], on='playerid', how='left')
        except:
            h['CBS'] = 0

        h.drop(columns=['wOBA', 'CS', 'Fld', 'BsR', 'ADP'],inplace=True)
        h['Primary_Pos'] = h.apply(lambda x: self.find_primary_pos(x['Pos']), axis=1)
        
        proj = pd.pivot_table(h, index='playerid', values=['G', 'PA', 'AB', 'H', 'HR', 'R', 'RBI', 'SB'], aggfunc='mean').merge(h[['playerid', 'Name', 'CBSNAME', 'Team', 'Pos', 'Primary_Pos', 'Dollars', 'CBS']], on='playerid', how='inner').drop_duplicates()
        proj['sorter'] = proj['HR']+proj['R']+proj['RBI']+proj['H']+proj['SB']
        proj['BA'] = proj['H']/proj['AB']
        proj = proj.drop_duplicates(subset='playerid')

        # Pitching section
        proj_system_list = []
        for proj_file in self.proj_systems:
            try:
                temp = pd.read_csv('data/'+str(yr)+'-'+proj_file+'-proj-p.csv', encoding="latin-1")
                proj_system_list.append(temp)
            except:
                pass
        p = pd.concat(proj_system_list).sort_values('PlayerId')
        p.rename(columns={'PlayerId':'playerid'},inplace=True)

        # Add Fangraphs auction values
        try:
            val_p = pd.read_csv('data/'+str(yr)+'-fangraphs-auction-calculator-p.csv')
            val_p.rename(columns={'PlayerId':'playerid', 'POS':'Pos'},inplace=True)
            p = p.merge(val_p[['playerid', 'Pos', 'Dollars']])
        except:
            p['Dollars'] = 0
        
        try:
            p = p.merge(cbs[['playerid', 'CBSNAME', 'CBS']], on='playerid', how='left')
        except:
            p['CBS'] = 0

        p.rename(columns={'H':'HA'},inplace=True)
        p['Sv+Hld'] = p['SV']+p['HLD']
        p['Primary_Pos'] = p['Pos'].apply(lambda x: ', '.join(x.split('/')))

        pproj = pd.pivot_table(p, index='playerid', values=['GS', 'G', 'IP', 'ER', 'HA', 'SO', 'BB', 'W', 'SV', 'HLD', 'Sv+Hld'], aggfunc='mean').merge(p[['playerid', 'Name', 'CBSNAME', 'Team', 'Pos', 'Dollars', 'CBS']], on='playerid', how='inner').drop_duplicates()
        pproj['sorter'] = pproj['SO']+(pproj['Sv+Hld']*4)+pproj['W']
        pproj['Primary_Pos'] = pproj.apply(lambda x: self.find_primary_pos(x['Pos']), axis=1)
        
        pproj['IP'].fillna(0, inplace=True)
        for i in ['PA', 'AB', 'G', 'H', 'HR', 'R', 'RBI', 'SB']:
            proj[i].fillna(0,inplace=True)
            proj[i] = proj[i].apply(lambda x: int(x))
        for i in ['GS', 'G', 'HA', 'SO', 'ER', 'BB', 'W', 'SV', 'HLD', 'Sv+Hld']:
            pproj[i].fillna(0,inplace=True)
            pproj[i] = pproj[i].apply(lambda x: int(x))
        pproj['ERA'] = pproj['ER']/pproj['IP']*9
        pproj['WHIP'] = (pproj['HA']+pproj['BB'])/pproj['IP']
        pproj = pproj.drop_duplicates(subset='playerid')

        proj.CBS.fillna(0,inplace=True)
        pproj.CBS.fillna(0,inplace=True)

        ids = self.load_id_map()

        h = proj.merge(ids[['IDFANGRAPHS', 'TEAM', 'CBSNAME']], left_on=['playerid', 'Team'], right_on=['IDFANGRAPHS', 'TEAM'], how='left').drop(columns=['IDFANGRAPHS', 'TEAM'])
        h.loc[h['CBSNAME_y'].notna(), 'Name'] = h.loc[h['CBSNAME_y'].notna()]['CBSNAME_y']
        h['Name'].fillna(h['CBSNAME_x'],inplace=True)
        h['Name'].fillna(h['CBSNAME_y'],inplace=True)
        h.drop(columns=['CBSNAME_x', 'CBSNAME_y'],inplace=True)

        p = pproj.merge(ids[['IDFANGRAPHS', 'TEAM', 'CBSNAME']], left_on=['playerid', 'Team'], right_on=['IDFANGRAPHS', 'TEAM'], how='left').drop(columns=['IDFANGRAPHS', 'TEAM'])
        p.loc[p['CBSNAME_y'].notna(), 'Name'] = p.loc[p['CBSNAME_y'].notna()]['CBSNAME_y']
        p['Name'].fillna(p['CBSNAME_x'],inplace=True)
        p['Name'].fillna(p['CBSNAME_y'],inplace=True)
        p.drop(columns=['CBSNAME_x', 'CBSNAME_y'],inplace=True)

        # Establish previous year qualifying averages for the major stat categories
        quals = self.get_qual_avgs(self.yr-1)

        # Adding StatCast data
        scb = pd.read_csv('data/'+str(yr-1)+'-statcast-h.csv')
        scb.rename(columns={'IDfg':'playerid', 'PA':'PA_ly', 'HR':'HR_ly', 'R':'R_ly', 'RBI':'RBI_ly', 'SB':'SB_ly', 'AVG':'BA_ly'},inplace=True)
        scb['playerid'] = scb['playerid'].astype(str)
        scp = pd.read_csv('data/'+str(yr-1)+'-statcast-p.csv')
        scp.rename(columns={'IDfg':'playerid', 'IP':'IP_ly', 'W':'W_ly', 'ERA':'ERA_ly', 'WHIP':'WHIP_ly', 'SV':'SV_ly', 'HLD':'HLD_ly', 'SO':'SO_ly'},inplace=True)
        scp['playerid'] = scp['playerid'].astype(str)
        scp['Sv+Hld_ly'] = scp['SV_ly']+scp['HLD_ly']
        sc = pd.concat([scb,scp])
        h = h.merge(sc[['playerid', 'Age', 'PA_ly', 'H_ly', 'AB_ly', 'HR_ly', 'SB_ly', 'R_ly', 'RBI_ly', 'BA_ly', 'BB%', 'K%', 'Contact%', 'O-Contact%', 'Z-Contact%', 'O-Swing%', 'Z-Swing%', 'Swing%', 'Hard%+', 'Hard%', 'HardHit%', 'EV', 'maxEV', 'LA', 'Barrels', 'Barrel%', 'Events', 'xBA', 'xSLG', 'xwOBA', 'IP_ly', 'ER_ly', 'HA_ly', 'BB_ly', 'HBP_ly', 'ERA_ly', 'WHIP_ly', 'Sv+Hld_ly', 'W_ly', 'SO_ly', 'CSW%', 'SIERA', 'FIP', 'xFIP', 'xERA', 'ERA-', 'FBv', 'HR/9', 'BB/9']], on='playerid',how='left')
        p = p.merge(sc[['playerid', 'Age', 'PA_ly', 'H_ly', 'AB_ly', 'HR_ly', 'SB_ly', 'R_ly', 'RBI_ly', 'BA_ly', 'BB%', 'K%', 'Contact%', 'O-Contact%', 'Z-Contact%', 'O-Swing%', 'Z-Swing%', 'Swing%', 'Hard%+', 'Hard%', 'HardHit%', 'EV', 'maxEV', 'LA', 'Barrels', 'Barrel%', 'Events', 'xBA', 'xSLG', 'xwOBA', 'IP_ly', 'ER_ly', 'HA_ly', 'BB_ly', 'HBP_ly', 'ERA_ly', 'WHIP_ly', 'Sv+Hld_ly', 'W_ly', 'SO_ly', 'CSW%', 'SIERA', 'FIP', 'xFIP', 'xERA', 'ERA-', 'FBv', 'HR/9', 'BB/9']], on='playerid',how='left')
        
        h.loc[(h['playerid'].duplicated()) & (h['Primary_Pos'].isin(['C', '1B', '2B', '3B', 'SS', 'OF', 'DH'])), 'keep'] = 0
        h['keep'].fillna(1,inplace=True)
        h = h[h['keep']==1]

        p.loc[(p['playerid'].duplicated()) & (p['Primary_Pos'].isin(['SP', 'RP', 'P'])), 'keep'] = 0
        p['keep'].fillna(1,inplace=True)
        p = p[p['keep']==1]
        
        print('Doing the BIGAA calculations...')
        h = self.calc_z(h, 'h')
        p = self.calc_z(p, 'p')

        print('Making adjustments by position...')
        c_adjust = abs(h[h['Primary_Pos']=='C'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['C']]['BIGAAh'])
        h.loc[h['Primary_Pos']=='C', 'Pos_adj'] = c_adjust
        ci_adjust = abs(h[h['Primary_Pos'].isin(['1B', '3B'])].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['1B']+self.drafted_by_pos['3B']]['BIGAAh'])
        h.loc[h['Primary_Pos'].isin(['1B', '3B']), 'Pos_adj'] = ci_adjust
        mi_adjust = abs(h[h['Primary_Pos'].isin(['2B', 'SS'])].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['SS']+self.drafted_by_pos['2B']]['BIGAAh'])
        h.loc[h['Primary_Pos'].isin(['2B', 'SS']), 'Pos_adj'] = mi_adjust
        of_adjust = abs(h[h['Primary_Pos'].isin(['OF', 'DH'])].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['OF']]['BIGAAh'])
        h.loc[h['Primary_Pos'].isin(['OF', 'DH']), 'Pos_adj'] = of_adjust

        sp_adjust = abs(p[p['Primary_Pos']=='SP'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['SP']]['BIGAAp'])
        p.loc[p['Primary_Pos']=='SP', 'Pos_adj'] = sp_adjust
        rp_adjust = abs(p[p['Primary_Pos']=='RP'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['RP']]['BIGAAp'])
        p.loc[p['Primary_Pos']=='RP', 'Pos_adj'] = rp_adjust
        
        # Apply Positional adjustment
        h['z'] = h['BIGAAh'] + h['Pos_adj']
        p['z'] = p['BIGAAp'] + p['Pos_adj']

        conv = (self.tm_dollars/self.tm_players)*(self.tot_players/(h[h['z']>0]['z'].sum()+p[p['z']>0]['z'].sum()))
        print('\nTotal z:',h[h['z']>0]['z'].sum()+p[p['z']>0]['z'].sum())
        print('\nH/P split:',h[h['z']>0]['z'].sum()/(h[h['z']>0]['z'].sum()+p[p['z']>0]['z'].sum()))
        print('Conversion to $:',conv)
        
        h['Value'] = h['z']*conv
        h['Value_ly'] = h['z_h_ly']*conv
        
        p['Value'] = p['z']*conv
        p['Value_ly'] = p['z_p_ly']*conv

        b = pd.concat([h,p])
        b['Outs'] = b['IP']*3
        b['K/9'] = b['SO']*9/(b['Outs']/3)
        
        print('Adding keepers')
        kdf = pd.read_csv('data/'+str(self.yr)+'-keepers.csv')
        kdf['playerid'] = kdf['playerid'].astype(str)
        b = b.merge(kdf, on='playerid', how='left')
        b['Paid'].fillna(0,inplace=True)
        b["Paid"] = pd.to_numeric(b.Paid, downcast='integer')
        b['Keeper'].fillna(0,inplace=True)

        
        self.data = b.sort_values('Value', ascending=False)
        return b.sort_values('Value', ascending=False)
    
    def upload(self, tbl='players'):
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///fantasy_data.db', echo=False)
        self.data.to_sql(tbl, engine, if_exists='replace')
        return



def owners(conv, n_teams=12):
    tot_dollars = n_teams * tot_dollars
    tot_players = n_teams * tm_players
    df = pd.read_sql('players', engine)
    owners_df = df.groupby('Owner').agg({'Name':'count', 'Paid':'sum', 'z':'sum', 'H':'sum', 'AB':'sum', 'HR':'sum', 'R':'sum', 'RBI':'sum', 'SB':'sum', 'W':'sum', 'Sv+Hld':'sum', 'SO':'sum'}).reset_index()
    owners_df.rename(columns={'Name':'Drafted'},inplace=True)
    owners_df['$/unit'] = owners_df['Paid']/owners_df['z']
    owners_df['$ Left'] = tm_dollars - owners_df['Paid']
    owners_df['$ Left / Plyr'] = owners_df['$ Left'] / (tm_players -owners_df['Drafted'])
    owners_df['Cash Sitch'] = owners_df['$ Left / Plyr'] / (((tot_dollars - owners_df.Paid.sum()) + owners_df['Paid']) / ((tot_players - owners_df.Drafted.sum()) + owners_df['Drafted']))
    owners_df['Value'] = (owners_df['z']*conv)-owners_df['Paid']
    owners_df['BA'] = owners_df['H']/owners_df['AB']
    owners_df['Pts'] = 0
    for i in ['BA', 'HR', 'R', 'RBI', 'SB', 'W', 'Sv+Hld', 'SO']:
        owners_df['Pts'] += owners_df[i].rank()
    owners_df['Rank'] = owners_df['Pts'].rank()
    return df.sort_values('z', ascending=False), owners_df


def check_roster_pos(roster, name, team_name, pos, eligible):
    eligible_at = eligible.split('/')
    eligibility = []
    for p in eligible.split('/'):
        if p=='C':
            eligibility.extend(['C'])
        if p=='1B':
            eligibility.extend(['1B', 'CI'])
        if p=='2B':
            eligibility.extend(['2B', 'MI'])
        if p=='3B':
            eligibility.extend(['3B', 'CI'])
        if p=='SS':
            eligibility.extend(['SS', 'MI'])
        if p=='OF':
            eligibility.extend(['OF1', 'OF2', 'OF3', 'OF4', 'OF5'])
        if p in ['SP', 'RP']:
            eligibility.extend(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'])
        
    eligibility = list(dict.fromkeys(eligibility))
    if 'SP' in eligible_at or 'RP' in eligible_at: 
        pos_list = eligibility
    else:
        pos_list = eligibility+['DH1', 'DH2']
    for p in pos_list:
        if roster.loc[p, team_name]==0:
            roster.loc[p, team_name] = name
            return p
    
    return pos_list


def next_closest_in_tier(df, pos, playerid):
    try:
        i = df[(df['Primary_Pos']==pos) & (df['playerid']==playerid) & (df['Owner'].isna())].index[0]
        val = df[(df['Primary_Pos']==pos) & (df['Owner'].isna()) & (df['playerid']==playerid)]['Value'].iloc[0]
        return df[df['playerid']==playerid]['Value'].iloc[0] - df[(df['Primary_Pos']==pos) & (df['Owner'].isna()) & (df['Value']<=val)].iloc[1]['Value']
    except:
        return 0


def create_statcast_csv(yr):
    from pybaseball import batting_stats, pitching_stats
    scb = batting_stats(yr, qual=1)
    scb.rename(columns={'IDfg':'playerid', 'PA':'PA_ly', 'HR':'HR_ly', 'R':'R_ly', 'RBI':'RBI_ly', 'SB':'SB_ly', 'AVG':'BA_ly'},inplace=True)
    scb['playerid'] = scb['playerid'].astype(str)
    scb.to_csv('data/'+str(yr)+'-statcast-h.csv')
    
    scp = pitching_stats(yr, qual=0)
    scp.rename(columns={'IDfg':'playerid', 'IP':'IP_ly', 'W':'W_ly', 'ERA':'ERA_ly', 'WHIP':'WHIP_ly', 'SV':'SV_ly', 'HLD':'HLD_ly', 'SO':'SO_ly'},inplace=True)
    scp['playerid'] = scp['playerid'].astype(str)
    scp['Sv+Hld_ly'] = scp['SV_ly']+scp['HLD_ly']
    scp.sort_values('IP_ly')
    scp.to_csv('data/'+str(yr)+'-statcast-p.csv')
    sc = pd.concat([scb,scp])
    return sc


def show_all_tables():
    # First check available tables
    return pd.read_sql("SELECT name FROM sqlite_master", engine)

def create_new_table(yr):
    # Create new table <name>
    tbl_name = 'players'+str(yr)
    pd.read_sql("CREATE TABLE "+tbl_name+" AS SELECT * FROM players",engine)
    return f"tbl_name created"

def drop_table(tbl_name):
    # Drop table if needed
    pd.read_sql('DROP TABLE '+tbl_name,engine)
    return f"tbl_name dropped"

def check_old_table(tbl_name):
    # Check the old table with new name
    return pd.read_sql("SELECT * from "+tbl_name, engine)

def fix_downloaded_cbs_values(year, save=False):
    # First get spreadsheet from CBS auction values page
    # Then parse name out of first column
    # Save it to data/ folder as yyyy-cbs-values.csv
    ids = Fantasy_Projections.load_id_map()
    cbs = pd.read_csv('data/'+str(year)+'-cbs-values.csv')
    cbs = cbs.merge(ids[['CBSNAME', 'TEAM', 'IDFANGRAPHS']], left_on=['Name', 'Team'], right_on=['CBSNAME','TEAM'], how='left')
    cbs.rename(columns={'IDFANGRAPHS':'playerid'},inplace=True)
    if save:
        cbs[['playerid', 'Name', 'Pos', 'Team', 'CBS']].to_csv('data/'+str(year)+'-cbs-values.csv',index=False)
    return cbs
