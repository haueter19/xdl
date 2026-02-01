import numpy as np
import pandas as pd
import math
from datetime import datetime
import re
import requests
from sqlalchemy import create_engine
from scipy.stats import poisson


tm_players, tm_dollars = 23, 260

engine = create_engine('sqlite:///fantasy_data.db', echo=False)


class Fantasy_Projections():
    def __init__(self, hitting_data=None, pitching_data=None, yr=datetime.now().year, n_teams=12, tm_players=23, tm_dollars=260, player_split=.6) -> None:
        self.hitting_data = hitting_data
        self.pitching_data = pitching_data
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
        
        self.proj_systems = ['atc', 'thebat', 'fangraphsdc', 'steamer']
        self.pos_hierarchy = ['C', '2B', '3B', 'SS', 'OF', '1B', 'DH', 'SP', 'RP', 'P']
        self.keepers_url = 'https://docs.google.com/spreadsheets/d/1dwDC2uMsfVRYeDECKLI0Mm_QonxkZvTkZTfBgnZo7-Q/edit#gid=1723951361'

    def load_id_map(self):
        """
        Function to download curated list of players with associated fantasy system IDs
        """
        player_id_url = 'https://docs.google.com/spreadsheets/d/1JgczhD5VDQ1EiXqVG-blttZcVwbZd5_Ne_mefUGwJnk/pubhtml?gid=0&single=true'
        ids = pd.read_html(player_id_url, header=1)[0]
        ids.drop(columns=['1', 'Unnamed: 9'], inplace=True)
        ids = ids[ids['PLAYERNAME'].notna()]
        self.ids = ids
        return ids

    def find_primary_pos(self, p):
        """
        Function to find the position of most value
        :param: p, str, list of positions separated by a slash (/)
        
        :returns: highest position from the pos_hierarchy list
        """
        pos_list = re.split('[,/]', p)
        for i in self.pos_hierarchy:
            if i in pos_list:
                return i

    def get_qual_avgs(self, previous_season_stats_hitter, previous_season_stats_pitcher, yr=datetime.now().year, min_pa=440, sp_min_ip=130, rp_ip_range=[45,95], min_sv_hld=5):
        """
        Accepts a dataset and some parameters for filtering that dataset. Finds the average and standard deviation for various hitting 
        and pitching categories. The idea is to get a baseline for what an average good player did the previous season. Projections will
        be compared to this baseline. 
        """
        # Checks to make sure the supplied dataframe has the appropriate columns to do the analysis
        # If it is not a dataframe, then it will attempt to load from a local storage
        if isinstance(previous_season_stats_hitter,pd.DataFrame):
            for stat in ['PA', 'AB', 'H', 'HR', 'R', 'RBI', 'SB']:
                assert stat in previous_season_stats_hitter.columns
        else:
            previous_season_stats_hitter = pd.read_csv('data/'+str(yr-1)+'-final-stats-h.csv', encoding='latin1')
            previous_season_stats_hitter['PA'] = previous_season_stats_hitter['AB']+previous_season_stats_hitter['BB']
            previous_season_stats_hitter = previous_season_stats_hitter.query('PA>0')
            previous_season_stats_hitter = previous_season_stats_hitter.sort_values('PA', ascending=False)
            previous_season_stats_hitter.rename(columns={'Positions':'Pos'},inplace=True)
            previous_season_stats_hitter['Primary_Pos'] = previous_season_stats_hitter['Pos'].apply(lambda x: self.find_primary_pos(x))

        previous_season_stats_hitter = previous_season_stats_hitter[previous_season_stats_hitter['PA']>min_pa]
        lgBA = previous_season_stats_hitter['H'].sum()/previous_season_stats_hitter['AB'].sum()
        previous_season_stats_hitter['zlgBA'] = previous_season_stats_hitter.apply(lambda x: x['H']-(x['AB']*(lgBA)), axis=1)
        quals_h = previous_season_stats_hitter[['H', 'AB', 'PA', 'zlgBA', 'R', 'RBI', 'HR', 'SB']].describe().to_dict()
        
        if isinstance(previous_season_stats_pitcher,pd.DataFrame):
            for stat in ['BB', 'HA', 'ER', 'IP', 'SO', 'QS', 'SvHld']:
                assert stat in previous_season_stats_pitcher.columns
        else:
            previous_season_stats_pitcher = pd.read_csv('data/'+str(yr-1)+'-final-stats-p.csv', encoding='latin1')
            previous_season_stats_pitcher.rename(columns={'INNs':'IP', 'S':'SV', 'HD':'HLD'},inplace=True)
            previous_season_stats_pitcher = previous_season_stats_pitcher.query('IP>0')
            previous_season_stats_pitcher['SvHld'] = previous_season_stats_pitcher['SV']+previous_season_stats_pitcher['HLD']
            previous_season_stats_pitcher = previous_season_stats_pitcher.sort_values('IP', ascending=False)
            previous_season_stats_pitcher.rename(columns={'Positions':'Pos'},inplace=True)
            previous_season_stats_pitcher['Primary_Pos'] = previous_season_stats_pitcher['Pos'].apply(lambda x: 'RP' if x == None else self.find_primary_pos(x))
        
        #previous_season_stats_pitcher['playerid'] = previous_season_stats_pitcher['playerid'].astype(str)
        #val_p = pd.read_csv('data/'+str(yr)+'-fangraphs-auction-calculator-p.csv')
        #val_p['playerid'] = val_p['playerid'].astype(str)
        #val_p['Primary_Pos'] = val_p['POS'].apply(lambda x: self.find_primary_pos(x))
        #previous_season_stats_pitcher = previous_season_stats_pitcher.merge(val_p[['playerid', 'Primary_Pos']], on='playerid', how='inner')
        
        previous_season_stats_pitcher = previous_season_stats_pitcher[(previous_season_stats_pitcher['Primary_Pos']=='RP') & (previous_season_stats_pitcher['IP'].between(rp_ip_range[0],rp_ip_range[1]) & (previous_season_stats_pitcher['SvHld']>min_sv_hld)) | (previous_season_stats_pitcher['Primary_Pos']=='SP') & (previous_season_stats_pitcher['IP']>sp_min_ip)]
        lgERA = previous_season_stats_pitcher['ER'].sum()/previous_season_stats_pitcher['IP'].sum()*9
        lgWHIP = (previous_season_stats_pitcher['BB'].sum()+previous_season_stats_pitcher['HA'].sum())/previous_season_stats_pitcher['IP'].sum()
        previous_season_stats_pitcher['zlgERA'] = previous_season_stats_pitcher.apply(lambda x: ((x['ER']*9) - (x['IP']*lgERA))*-1, axis=1)
        previous_season_stats_pitcher['zlgWHIP'] = previous_season_stats_pitcher.apply(lambda x: ((x['HA']+x['BB'])-(x['IP']*lgWHIP))*-1, axis=1)
        quals_p = previous_season_stats_pitcher[['BB', 'HA', 'ER', 'IP', 'SO', 'QS', 'SvHld', 'zlgERA', 'zlgWHIP']].describe().to_dict()
        #quals_p['HA'] = quals_p.pop('H')
        quals_h.update(quals_p)
        self.quals = quals_h
        return quals_h


    def big_board(self, row, stat, qual):
        # Handle rate stats different from counting stats
        # Rate stats
        if stat == 'BA':
            ba_pts = row['H']-(row['AB']*(qual['H']['mean']/qual['AB']['mean']))
            zBA = (ba_pts-qual['zlgBA']['mean'])/qual['zlgBA']['std']
            #return ((row['AB'] * (((row['H']/row['AB'])-qual_avgs['AVG'][0])/qual_avgs['AVG'][1])) - qual_avgs['zlgBA'][0])/qual_avgs['zlgBA'][1]
            return zBA
        elif stat == 'BA_ly':
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
        # Counting stats
        else:
            if stat[-3:] == '_ly':
                return (row[stat] - qual[stat[:-3]]['mean']) / qual[stat[:-3]]['std']
            else:
                return (row[stat] - qual[stat]['mean']) / qual[stat]['std']

    def calc_z(self, df, kind):
        """
        Calculate z-scores for each stat and each row of the projections dataframe
        """
        if kind=='h':
            for stat in ['BA', 'HR', 'R', 'RBI', 'SB', 'BA_ly', 'HR_ly', 'R_ly', 'RBI_ly', 'SB_ly']:
                df['z'+stat] = df.apply(lambda row: self.big_board(row, stat, self.quals), axis=1)
            df['BIGAAh'] = df['zR']+df['zRBI']+df['zHR']+df['zSB']+df['zBA']
            df['z_h_ly'] = df['zR_ly']+df['zRBI_ly']+df['zHR_ly']+df['zSB_ly']+df['zBA_ly']
            return df
        else:
            for stat in ['ERA', 'WHIP', 'SO', 'QS', 'SvHld', 'ERA_ly', 'WHIP_ly', 'SO_ly', 'QS_ly', 'SvHld_ly']:
                df['z'+stat] = df.apply(lambda row: self.big_board(row, stat, self.quals), axis=1)
            df['BIGAAp'] = df['zERA']+df['zWHIP']+df['zQS']+df['zSO']+df['zSvHld']
            df['z_p_ly'] = df['zERA_ly']+df['zWHIP_ly']+df['zQS_ly']+df['zSO_ly']+df['zSvHld_ly']
            return df
    

    def preprocess_hitting_projections(self):
        """
        Function to load various csv files containing projections for the upcoming season, clean the data, and save it to the class. 
        """
        # Loop through possible projection csv files. Load, if they exist, into a list and then concatenate into a dataframe
        h = pd.DataFrame()
        for proj_file in self.proj_systems:
            try:
                temp = pd.read_csv(f'data/{datetime.now().year}-{proj_file}-proj-h.csv', encoding="latin-1")
                temp.rename(columns={'SO':'K'},inplace=True)
                if 'HBP' not in temp.columns:
                    temp['HBP'] = 0
                temp['sys'] = proj_file
                h = pd.concat([h, temp])
                print(f'Found {proj_file}')
            except:
                print(f"Did not find file {proj_file}")
                pass

        h = h.drop(columns='ï»¿Name')
        h.rename(columns={'NameASCII':'Name', 'PlayerId':'playerid'},inplace=True)

        # Load players table from database
        players = pd.read_sql('SELECT * FROM players', engine)
        # Merge the player ID map with the projections dataframe
        h = h.merge(players[['cbsid', 'CBSNAME', 'IDFANGRAPHS']], left_on='playerid', right_on='IDFANGRAPHS', how='left').sort_values('PA', ascending=False)
        # Add in cbs projections
        cbs = pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year}-cbs-proj-h.csv')
        cbs['sys'] = 'cbs'
        cbs.rename(columns={'Positions':'Pos'},inplace=True)
        cbs = cbs.merge(players[['cbsid', 'IDFANGRAPHS']], on='cbsid', how='left')
        # Merge the cbs projections with the other projections
        h = pd.concat([h,cbs]).copy()
        h.fillna({'CBS':0, 'playerid':h['IDFANGRAPHS'], 'Name':h['CBSNAME']},inplace=True)
        h.drop(columns=['InterSD', 'InterSK', 'IntraSD'],inplace=True)
        # Compute TB
        h['TB'] = h['1B'] + (h['2B']*2) + (h['3B']*3) + (h['HR']*4)
        # There are thousands of projections. I'm going to filter out players who are not projected to get at least 1 AB
        h = h[(h['AB']>1)]
        # If I ever want to save these and explore, can use this line
        h_without_cbsid = h[h['cbsid'].isna()]
        # Drop rows where cbsid is null
        h = h[h['cbsid'].notna()]
        # Collapse projections into one per cbsid and take the average for each stat
        # Then merge with original dataframe to get the player's name and team
        # Drop duplicates by cbsid since the previous merge will duplicate rows
        proj = pd.pivot_table(h, index='cbsid', values=['G', 'PA', 'AB', 'H', 'HR', 'R', 'RBI', 'SB', 'BB', 'HBP', 'K', 'SF', 'SH', 'TB', 'Vol', 'Skew', 'Dim', 'wRC+'], aggfunc='mean')\
            .merge(h[['cbsid', 'playerid', 'Name', 'Team']], on='cbsid', how='inner').drop_duplicates('cbsid') # drop by cbsid b/c when this is the same, the data avgs for the entries are the same
        # Helps with sorting later when we need to find the top 12 players at a position.
        proj['sorter'] = proj['HR']+proj['R']+proj['RBI']+proj['H']+proj['SB']
        
        # Calculate some rate stats
        proj['BA'] = proj['H']/proj['AB']
        proj['OBP'] = (proj['H']+proj['BB']+proj['HBP']) / (proj['AB']+proj['BB']+proj['HBP']+proj['SF'])
        proj['SLG'] = proj['TB'] / proj['AB']
        proj['OPS'] = proj['OBP'] + proj['SLG']
        proj['K%'] = proj['K']/proj['PA']
        proj['BB%'] = proj['BB']/proj['PA']
        # Merge with the CBS projections. Because I use a CBS system, every ID and Name and Position should come from there. 
        # This step brings in the CBS designated position and rank
        proj = proj.merge(cbs[['cbsid', 'Pos', 'Rank']], how='inner', on='cbsid').drop_duplicates('cbsid')
        # Add CBS auction values. Serves as a reference point to see if my projections seem reasonable
        cbs_auction_values = pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year}-cbs-auction-values.csv')[['cbsid', 'CBSNAME', 'CBS']]
        proj = proj.merge(cbs_auction_values[['cbsid', 'CBS']], on='cbsid', how='left')
        proj.fillna({'CBS':0},inplace=True)
        # Merge data with Fangraphs Auction Calculator values
        proj = proj.merge(pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year}-fangraphs-auction-calculator-h.csv')[['Dollars', 'PlayerId']], left_on='playerid', right_on='PlayerId', how='left')
        proj.fillna({'Dollars':0, 'ADP':0},inplace=True)
        proj.drop(columns='PlayerId',inplace=True)
        proj = proj.sort_values('sorter', ascending=False)
        # Find primary position
        proj['Primary_Pos'] = proj['Pos'].apply(lambda x: self.find_primary_pos(x) if type(x) != float else x)

        lyh = pd.read_csv('data/'+str(self.yr-1)+'-final-stats-h.csv', encoding='latin1')
        for col in ['PA', 'AB', 'H', 'HR', 'R', 'RBI', 'SB', 'BB', 'K', 'AVG', 'HBP']:
            if col not in lyh.columns:
                lyh[col] = 0
        lyh.rename(columns={'PA':'PA_ly', 'AB':'AB_ly', 'HR':'HR_ly', 'R':'R_ly', 'RBI':'RBI_ly', 'SB':'SB_ly', 'BB':'BB_ly', 'K':'K_ly', 'H':'H_ly', 'AVG':'BA_ly', 'HBP':'HBP_ly'},inplace=True)
        proj = proj.merge(lyh[['cbsid', 'PA_ly', 'H_ly', 'AB_ly', 'HR_ly', 'SB_ly', 'R_ly', 'RBI_ly', 'BA_ly', 'BB_ly', 'HBP_ly', 'K_ly']], on='cbsid',how='left')

        # Merge with StatCast data if it exists
        try:
            stat_cast = pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year-1}-statcast.csv')
            sc_ly = stat_cast[stat_cast['year']==datetime.now().year-1][['cbsid', 'MLBID', 'player_name', 'year', 'player_age', 'woba', 'xwoba', 'woba_diff', 'xba', 'barrel_batted_rate', 'sprint_speed', 'exit_velocity_avg', 'K/9', 'K-BB%', 'ff_avg_speed', 'fastball_avg_speed', 'fastball_avg_break_z_induced', 'whiff_percent', 'home_run', 'pa', 'r_total_stolen_base', 'r_run', 'b_rbi', 'batting_avg', 'p_quality_start', 'p_SvHld', 'p_strikeout', 'p_out', 'p_era', 'p_whip']]
            sc_ly.rename(columns={col:col+'_ly' for col in sc_ly.columns if col not in ['cbsid', 'MLBID', 'player_name', 'year']},inplace=True)

            sc_2ly = stat_cast[stat_cast['year']==datetime.now().year-2][['cbsid', 'MLBID', 'player_name', 'year', 'player_age', 'woba', 'xwoba', 'woba_diff', 'xba', 'barrel_batted_rate', 'sprint_speed', 'exit_velocity_avg', 'K/9', 'K-BB%', 'ff_avg_speed', 'fastball_avg_speed', 'fastball_avg_break_z_induced', 'whiff_percent', 'home_run', 'pa', 'r_total_stolen_base', 'r_run', 'b_rbi', 'batting_avg', 'p_quality_start', 'p_SvHld', 'p_strikeout', 'p_out', 'p_era', 'p_whip']]
            sc_2ly.rename(columns={col:col+'_2ly' for col in sc_2ly.columns if col not in ['cbsid', 'MLBID', 'player_name', 'year']},inplace=True)

            proj.rename(columns={'PA_ly':'pa_ly_'},inplace=True)
            proj = proj.merge(sc_ly, on='cbsid', how='left').merge(sc_2ly, on='cbsid', how='left').drop_duplicates('cbsid')
            proj.fillna({'MLBID_x':proj['MLBID_y'], 'year_x':proj['year_y']},inplace=True)
            proj.drop(columns=['MLBID_y', 'player_name_x', 'player_name_y', 'year_y'],inplace=True)
            proj.rename(columns={'MLBID_x':'MLBID', 'year_x':'year'},inplace=True)
            proj['Age'] = proj['player_age_ly']+1
        except:
            print('There is a problem with the StatCast data')
            pass

        # Save hitting projections to the class
        self.hitting_data = proj.sort_values('sorter', ascending=False)
        return proj
    
    def preprocess_pitching_projections(self):
        # Loop through possible projection csv files. Load, if they exist, into a list and then concatenate into a dataframe
        p = pd.DataFrame()
        for proj_file in self.proj_systems:
            try:
                temp = pd.read_csv(f'data/{datetime.now().year}-{proj_file}-proj-p.csv', encoding="latin-1")
                temp.rename(columns={'SO':'K'},inplace=True)
                temp['sys'] = proj_file
                p = pd.concat([p, temp])
                print(f'Found {proj_file}')
            except:
                print(f"Did not find file {proj_file}")
                pass

        p = p.drop(columns='ï»¿Name')
        p.rename(columns={'NameASCII':'Name', 'PlayerId':'playerid'},inplace=True)
        
        players = pd.read_sql('SELECT * FROM players', engine)
        p = p.merge(players[['cbsid', 'CBSNAME', 'IDFANGRAPHS']], left_on='playerid', right_on='IDFANGRAPHS', how='left').sort_values('IP', ascending=False)

        # Add in cbs projections
        temp = pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year}-cbs-proj-p.csv')
        temp['sys'] = 'cbs'
        temp.rename(columns={'Positions':'Pos', 'INNs':'IP', 'S':'SV', 'HD':'HLD'},inplace=True)
        temp['ER'] = temp['ERA']/9*temp['IP']
        temp = temp.merge(players[['cbsid', 'IDFANGRAPHS']], on='cbsid', how='left')

        p = pd.concat([p,temp]).copy()
        p = p.drop_duplicates()

        p.fillna({'CBS':0, 'playerid':p['IDFANGRAPHS'], 'Name':p['CBSNAME']},inplace=True)
        p.drop(columns=['InterSD', 'InterSK', 'IntraSD'],inplace=True)
        p.rename(columns={'H':'HA', 'K':'SO'},inplace=True)
        p['SvHld'] = p['SV']+p['HLD']

        p = p[(p['IP']>1)]

        proj = pd.pivot_table(p, index='cbsid', values=['GS', 'G', 'IP', 'TBF', 'ER', 'HA', 'SO', 'BB', 'QS', 'SV', 'HLD', 'SvHld', 'Vol', 'Skew', 'Dim'], aggfunc='mean')\
            .merge(p[['cbsid', 'playerid', 'Name', 'Team']], on='cbsid', how='inner').drop_duplicates('cbsid') # drop by cbsid b/c when this is the same, the data avgs for the entries are the same
        proj['sorter'] = proj['SO']+(proj['SvHld']*4)+proj['QS']
        for i in ['IP', 'GS', 'G', 'HA', 'SO', 'ER', 'BB', 'QS', 'SV', 'HLD', 'SvHld']:
            proj.fillna({i:0},inplace=True)
            proj[i] = proj[i].apply(lambda x: int(x))
        proj['ERA'] = proj['ER']/proj['IP']*9
        proj['WHIP'] = (proj['HA']+proj['BB'])/proj['IP']
        proj['K%'] = round(proj['SO']/proj['TBF'],4)
        proj['BB%'] = round(proj['BB']/proj['TBF'],4)
        proj['K-BB%'] = round(proj['K%']-proj['BB%'],4)
        proj['K/9'] = round(proj['SO']/proj['IP']*9,4)
        # Merge data with CBS auction values
        cbs_auction_values = pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year}-cbs-auction-values.csv')[['cbsid', 'CBSNAME', 'CBS']]
        proj = proj.merge(cbs_auction_values[['cbsid', 'CBS']], on='cbsid', how='left')
        proj.fillna({'CBS':0},inplace=True)
        # Merge data with Fangraphs Auction Calculator values
        proj = proj.merge(pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year}-fangraphs-auction-calculator-p.csv')[['Dollars', 'PlayerId']], left_on='playerid', right_on='PlayerId', how='left')
        proj.fillna({'Dollars':0, 'ADP':0},inplace=True)
        proj.drop(columns='PlayerId',inplace=True)
        proj = proj.sort_values('sorter', ascending=False)
        
        proj = proj.merge(temp[['cbsid', 'Pos', 'Rank']], how='left', on='cbsid').drop_duplicates('cbsid')

        proj.loc[(proj['Pos'].isna()) & (proj['GS']>0), 'Pos'] = 'SP'
        proj.fillna({'Pos':'RP'},inplace=True)
        proj['Primary_Pos'] = proj['Pos'].apply(lambda x: 'SP' if 'SP' in x else 'RP')

        lyp = pd.read_csv('data/'+str(self.yr-1)+'-final-stats-p.csv', encoding='latin1')
        for col in ['INNs', 'ER', 'HA', 'BB', 'SO', 'QS', 'SV', 'HLD']:
            if col not in lyp.columns:
                lyp[col] = 0
        lyp.rename(columns={'INNs':'IP_ly', 'QS':'QS_ly', 'ER':'ER_ly', 'ERA':'ERA_ly', 'WHIP':'WHIP_ly', 'HA':'HA_ly', 'BB':'BB_ly', 'SV':'SV_ly', 'HLD':'HLD_ly', 'SO':'SO_ly'},inplace=True)
        lyp['SvHld_ly'] = lyp['SV_ly']+lyp['HLD_ly']

        proj = proj.merge(lyp[['cbsid', 'IP_ly', 'ER_ly', 'HA_ly', 'BB_ly', 'ERA_ly', 'WHIP_ly', 'SvHld_ly', 'QS_ly', 'SO_ly']], on='cbsid',how='left')

        # Merge with StatCast data if it exists
        try:
            stat_cast = pd.read_csv(rf'C:\GitHub\xdl\data\{datetime.now().year-1}-statcast.csv')
            sc_ly = stat_cast[stat_cast['year']==datetime.now().year-1][['cbsid', 'MLBID', 'player_name', 'year', 'player_age', 'woba', 'xwoba', 'woba_diff', 'xba', 'barrel_batted_rate', 'sprint_speed', 'exit_velocity_avg', 'K/9', 'K-BB%', 'ff_avg_speed', 'fastball_avg_speed', 'fastball_avg_break_z_induced', 'whiff_percent', 'home_run', 'pa', 'r_total_stolen_base', 'r_run', 'b_rbi', 'batting_avg', 'p_quality_start', 'p_SvHld', 'p_strikeout', 'p_out', 'p_era', 'p_whip']]
            sc_ly.rename(columns={col:col+'_ly' for col in sc_ly.columns if col not in ['cbsid', 'MLBID', 'player_name', 'year']},inplace=True)

            sc_2ly = stat_cast[stat_cast['year']==datetime.now().year-2][['cbsid', 'MLBID', 'player_name', 'year', 'player_age', 'woba', 'xwoba', 'woba_diff', 'xba', 'barrel_batted_rate', 'sprint_speed', 'exit_velocity_avg', 'K/9', 'K-BB%', 'ff_avg_speed', 'fastball_avg_speed', 'fastball_avg_break_z_induced', 'whiff_percent', 'home_run', 'pa', 'r_total_stolen_base', 'r_run', 'b_rbi', 'batting_avg', 'p_quality_start', 'p_SvHld', 'p_strikeout', 'p_out', 'p_era', 'p_whip']]
            sc_2ly.rename(columns={col:col+'_2ly' for col in sc_2ly.columns if col not in ['cbsid', 'MLBID', 'player_name', 'year']},inplace=True)

            proj.rename(columns={'PA_ly':'pa_ly_'},inplace=True)
            proj = proj.merge(sc_ly, on='cbsid', how='left').merge(sc_2ly, on='cbsid', how='left').drop_duplicates('cbsid')
            proj.fillna({'MLBID_x':proj['MLBID_y'], 'year_x':proj['year_y']},inplace=True)
            proj.drop(columns=['MLBID_y', 'player_name_x', 'player_name_y', 'year_y'],inplace=True)
            proj.rename(columns={'MLBID_x':'MLBID', 'year_x':'year'},inplace=True)
            proj['Age'] = proj['player_age_ly']+1
        except:
            print('There is a problem with the StatCast data')
            pass

        self.pitching_data = proj.sort_values('sorter', ascending=False)
        return proj
    

    def calc_value(self, previous_season_hitting=None, previous_season_pitching=None):
        """
        Calculate the auction value of the provided stats
        """

        # If hitting and pitching data was not provided, that means we need to create it using the prep_projection_data function
        if self.hitting_data is None:
            self.preprocess_hitting_projections()
        if self.pitching_data is None:
            self.preprocess_pitching_projections()

        #h = self.hitting_data
        #p = self.pitching_data

        # Establish previous year qualifying averages for the major stat categories
        quals = self.get_qual_avgs(previous_season_hitting, previous_season_pitching)
        
        print('Doing the BIGAA calculations...')
        h = self.calc_z(self.hitting_data, 'h')
        p = self.calc_z(self.pitching_data, 'p')

        print('Making adjustments by position...')
        adj_dict = {
            'C':abs(h[h['Primary_Pos']=='C'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['C']]['BIGAAh']),
            '1B':abs(h[h['Primary_Pos']=='1B'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['1B']]['BIGAAh']),
            '2B':abs(h[h['Primary_Pos']=='2B'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['2B']]['BIGAAh']),
            '3B':abs(h[h['Primary_Pos']=='3B'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['3B']]['BIGAAh']),
            'SS':abs(h[h['Primary_Pos']=='SS'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['SS']]['BIGAAh']),
            'OF':abs(h[h['Primary_Pos']=='OF'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['OF']]['BIGAAh']),
            #'DH':abs(h[h['Primary_Pos']=='DH'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['DH']]['BIGAAh']),
            'SP':abs(p[p['Primary_Pos']=='SP'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['SP']]['BIGAAp']),
            'RP':abs(p[p['Primary_Pos']=='RP'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['RP']]['BIGAAp']),
            #'P':abs(p[p['Primary_Pos']=='P'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['P']]['BIGAAp']),
        }
        """
        adj_dict = {
            'C':abs(h[h['Pos'].str.contains('C')].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['C']]['BIGAAh']),
            '1B':abs(h[h['Pos'].str.contains('1B')].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['1B']]['BIGAAh']),
            '2B':abs(h[h['Pos'].str.contains('2B')].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['2B']]['BIGAAh']),
            '3B':abs(h[h['Pos'].str.contains('3B')].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['3B']]['BIGAAh']),
            'SS':abs(h[h['Pos'].str.contains('SS')].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['SS']]['BIGAAh']),
            'OF':abs(h[h['Pos'].str.contains('OF')].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['OF']]['BIGAAh']),
            #'DH':abs(h[h['Primary_Pos']=='DH'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['DH']]['BIGAAh']),
            'SP':abs(p[p['Pos'].str.contains('SP')].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['SP']]['BIGAAp']),
            'RP':abs(p[p['Pos'].str.contains('RP')].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['RP']]['BIGAAp']),
            #'P':abs(p[p['Primary_Pos']=='P'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['P']]['BIGAAp']),
        }
        """
        adj_dict['DH'] = adj_dict[min(adj_dict)]
        adj_dict['P'] = min([adj_dict['SP'], adj_dict['RP']])
        self.pos_adjust = adj_dict
        #c_adjust = abs(h[h['Primary_Pos']=='C'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['C']]['BIGAAh'])
        #h.loc[h['Primary_Pos']=='C', 'Pos_adj'] = c_adjust
        #_1b_adjust = abs(h[h['Primary_Pos']=='1B'].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['1B']]['BIGAAh'])
        #ci_adjust = abs(h[h['Primary_Pos'].isin(['1B', '3B'])].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['1B']+self.drafted_by_pos['3B']]['BIGAAh'])
        #h.loc[h['Primary_Pos'].isin(['1B', '3B']), 'Pos_adj'] = ci_adjust
        #mi_adjust = abs(h[h['Primary_Pos'].isin(['2B', 'SS'])].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['SS']+self.drafted_by_pos['2B']]['BIGAAh'])
        #h.loc[h['Primary_Pos'].isin(['2B', 'SS']), 'Pos_adj'] = mi_adjust
        #of_adjust = abs(h[h['Primary_Pos'].isin(['OF', 'DH'])].sort_values('BIGAAh',ascending=False).iloc[self.drafted_by_pos['OF']]['BIGAAh'])
        #h.loc[h['Primary_Pos'].isin(['OF', 'DH']), 'Pos_adj'] = of_adjust

        #sp_adjust = abs(p[p['Primary_Pos']=='SP'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['SP']]['BIGAAp'])
        #p.loc[p['Primary_Pos']=='SP', 'Pos_adj'] = sp_adjust
        #rp_adjust = abs(p[p['Primary_Pos']=='RP'].sort_values('BIGAAp',ascending=False).iloc[self.drafted_by_pos['RP']]['BIGAAp'])
        #p.loc[p['Primary_Pos']=='RP', 'Pos_adj'] = rp_adjust

        # Apply Positional adjustment
        h['Pos_adj'] = h['Pos'].apply(lambda x: max([adj_dict[i] for i in x.split(',')]))
        p['Pos_adj'] = p['Pos'].apply(lambda x: max([adj_dict[i] for i in x.split(',')]))
        
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
        try:
            kdf = pd.read_csv('data/'+str(self.yr)+'-keepers.csv')
            b = b.merge(kdf, on='cbsid', how='left')
            b.fillna({'Paid':0},inplace=True)
            b["Paid"] = pd.to_numeric(b.Paid, downcast='integer')
            b.fillna({'Keeper':0},inplace=True)
        except:
            print('No keepers found')
        
        b = b.groupby('cbsid').agg({c:"max" for c in b.columns if c != 'cbsid'}).reset_index()
        self.data = b.sort_values('Value', ascending=False)
        print('Completed applying auction values')
        return b.sort_values('Value', ascending=False)
    

    def upload(self, tbl):
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///fantasy_data.db', echo=False)
        self.data.to_sql(tbl, engine, if_exists='replace')
        return





# Dynamic max_bid dictionary
max_bid_mult = {a:b for a, b in zip(list(range(1,46)), [3,2.5,2.2,1.9,1.8,1.7,1.6,1.55,1.5,1.4,1.36,1.35,1.34,1.33,1.32,1.31,1.3,1.29,1.28,1.27,1.26,1.25,1.24,1.23,1.22,1.21,1.2,1.19,1.18,1.17,1.16,1.15,1.14,1.13,1.12,1.11,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1])}

#owner_sort = [i[1] for i in enumerate(owners.keys())]

pos_elig = {
    'C':['C', 'DH1', 'DH2'],
    '1B':['1B', 'CI', 'DH1', 'DH2'],
    '2B':['2B', 'MI', 'DH1', 'DH2'],
    '3B':['3B', 'CI', 'DH1', 'DH2'],
    'SS':['SS', 'MI', 'DH1', 'DH2'],
    'OF':['OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'DH1', 'DH2'],
    'DH':['DH1', 'DH2'],
    'SP':['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'],
    'RP':['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'],
    'P':['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
}

def calculate_max_bid(player_value):
    """
    Dynamically calculate the max bid factor based on the player's value.
    """
    if player_value > 45:
        return 1.1 * player_value
    else:
        return max_bid_mult.get(player_value) * player_value
    


def simulate_auction(player_data, owners_dict, rosters, timidness=0.1, nominating_team=None):
    """
    Simulate an auction draft where teams bid on a player.
    
    Parameters:
    - player_value: Estimated value of the player.
    - owners: dict of owners
    - team_budgets: List of budgets for each team.
    - max_bid_factor: Maximum bid as a multiple of the player's value.
    - timidness: Higher values make teams more timid about bidding as bids increase.
    
    Returns:
    - A list of bids (0 means no bid) for each team.
    """
    player_value = int(max(player_data['CBS'], player_data['Dollars']))
    pos_elig = {
        'C':['C', 'DH1', 'DH2'],
        '1B':['1B', 'CI', 'DH1', 'DH2'],
        '2B':['2B', 'MI', 'DH1', 'DH2'],
        '3B':['3B', 'CI', 'DH1', 'DH2'],
        'SS':['SS', 'MI', 'DH1', 'DH2'],
        'OF':['OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'DH1', 'DH2'],
        'DH':['DH1', 'DH2'],
        'SP':['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'],
        'RP':['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'],
        'P':['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
    }
    elig_list = []
    for pos in player_data['Pos'].split(','):
        elig_list.extend(pos_elig[pos])
    elig_list = list(set(elig_list))
    max_bid = calculate_max_bid(player_value)
    bids = []

    for k,v in owners_dict.items():
        # Checks to see if the team has one of the player's positions available
        if rosters.map(lambda x: 0 if isinstance(x, str) and any(char.isalpha() for char in x) else 1).loc[elig_list, k].sum() <= 0:
            print(f"{k} ineligible to bid on player")
            bids.append(0)
        else:
            # Set budget and team's max bid as constraints
            budget = v['$ Left']
            team_max_bid = 260 if pd.isnull(v['max_bid']) else math.floor(v['max_bid'])
            print(k, budget, team_max_bid)

            # Probability of bidding decreases as the bid approaches max_bid
            bid_probability = max(0.1, 1 - timidness * (max_bid / player_value - 1))
            if np.random.random() > bid_probability or team_max_bid < player_value * 0.5:
                # Skip bidding due to budget or preference
                # Skip bidding due to budget or preference
                if k == nominating_team:
                    bids.append(1)
                else:
                    bids.append(0)
            else:
                # Generate a bid with some randomness
                # Do not allow a bid higher than team's max bid
                #bid = np.random.uniform(player_value * 0.8, min(team_max_bid, max_bid))
                pmf = poisson.pmf(np.arange(0, 50), 5)
                bid = np.random.choice(np.arange(len(pmf)), size=1, p=pmf)
                bid = min((player_value - 8) + bid, max_bid)
                bids.append(min(bid, budget))  # Respect team budgets
                # Make sure bids don't go under 0
                if bid < 0:
                    bid = 0
                # Make sure nominating team bids 1
                if k == nominating_team:
                    bid = max(1, bid)
                bids.append(min(bid, budget))  # Respect team budgets
    # Round bids to integers for realism
    return [int(bid) for bid in bids]



def find_bid_winner(bids, offer):
    orig_order = [i[1] for i in enumerate(owners.keys())] #list(owners.keys())
    owners_key = {tm:i for i, tm in enumerate(orig_order)}
    bid_order = orig_order[orig_order.index(offer):]+orig_order[:orig_order.index(offer)] #bid_order[offer:]+bid_order[:offer]
    bids = bids[owners_key[offer]:] + bids[:owners_key[offer]]
    current_bid = 0
    bid_complete = 0
    max_bid_round = max(bids)
    
    while current_bid < 60:
        if current_bid == max_bid_round:
            break
        if bid_complete == 1:
            break
        for n, owner in enumerate(bid_order):
            if len([i for i, val in enumerate(bids) if val >= current_bid]) < 2:
                #print(f'Bidding is complete. {bid_winner} wins with bid of {current_bid}')
                bid_complete = 1
                break
            elif current_bid >= bids[n]:
                #print(f"current bid is {current_bid}. {owner}'s top bid of {bids[n]} has been met, therefore this owner is done, current bid remains the same")
                pass
            else:
                bid_winner = owner
                current_bid += 1
                #print(f"current bid is {current_bid-1}, {len([i for i, val in enumerate(bids) if val >= current_bid])} -- {owner} max bid is {bids[n]} therefore {owner} raises bid to {current_bid}")
    return bid_winner, current_bid



def complete_bid(player_data):    
    player_value = max(player_data['CBS'], player_data['Dollars'])
    timidness = 0.3  # Adjust timidness to control bidding behavior
    
    bids = simulate_auction(player_data, owners_df.to_dict(orient='index'), rosters, timidness=timidness)
    
    print("Generated Bids:", bids)
    #plot_auction_bids(bids, player_value)
    
    winning_owner, winning_bid = find_bid_winner(bids, offer)
    offer = winning_owner
    print(winning_owner, winning_bid)
    
    # Update database
    url = 'http://localhost:8000/draft/update_bid'
    params = {'cbsid': cbsid, 'price': winning_bid, 'owner': winning_owner}
    r = requests.get(url, params)
    
    if r.status_code == 200:
        h = pd.read_sql(f"SELECT * FROM players{datetime.now().year} WHERE cbsid IS NOT NULL", engine)
        owners_df = pd.read_html(r.text)[2]
        owners_df.fillna({'$ Left':260, 'Max Bid':237, 'Drafted':0},inplace=True)
        owners_df = owners_df.set_index('Owner').reindex(owner_sort)
        owners_df.fillna({'$ Left':260, 'Max Bid':237, 'Drafted':0, '$ Left/Plyr':11.3},inplace=True)
        owners_df['$ Left'] = owners_df['$ Left'].astype(int)
        rosters = pd.read_html(r.text)[3]
        rosters = rosters.set_index('Pos')
    return h, owners_df, rosters



def owners(conv, n_teams=12, tm_players=23):
    tot_dollars = n_teams * 260
    tot_players = n_teams * tm_players
    df = pd.read_sql('players', engine)
    owners_df = df.groupby('Owner').agg({'Name':'count', 'Paid':'sum', 'z':'sum', 'H':'sum', 'AB':'sum', 'HR':'sum', 'R':'sum', 'RBI':'sum', 'SB':'sum', 'QS':'sum', 'Sv+Hld':'sum', 'SO':'sum'}).reset_index()
    owners_df.rename(columns={'Name':'Drafted'},inplace=True)
    owners_df['$/unit'] = owners_df['Paid']/owners_df['z']
    owners_df['$ Left'] = tm_dollars - owners_df['Paid']
    owners_df['$ Left / Plyr'] = owners_df['$ Left'] / (tm_players -owners_df['Drafted'])
    owners_df['Cash Sitch'] = owners_df['$ Left / Plyr'] / (((tot_dollars - owners_df.Paid.sum()) + owners_df['Paid']) / ((tot_players - owners_df.Drafted.sum()) + owners_df['Drafted']))
    owners_df['Value'] = (owners_df['z']*conv)-owners_df['Paid']
    owners_df['BA'] = owners_df['H']/owners_df['AB']
    owners_df['Pts'] = 0
    for i in ['BA', 'HR', 'R', 'RBI', 'SB', 'QS', 'Sv+Hld', 'SO']:
        owners_df['Pts'] += owners_df[i].rank()
    owners_df['Rank'] = owners_df['Pts'].rank()
    return df.sort_values('z', ascending=False), owners_df



def OLD_check_roster_pos(roster, name, team_name, pos, eligible):
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
        if p in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
            eligibility.extend([p])
        
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
    """
    If pybaseball is installed, running this function will download that year's statcast data and save it to data/ folder as yyyy-statcast-h.csv and yyyy-statcast-p.csv
    
    :param yr: Description
    """
    from pybaseball import batting_stats, pitching_stats
    scb = batting_stats(yr, qual=1)
    scb.rename(columns={'IDfg':'playerid', 'PA':'PA_ly', 'HR':'HR_ly', 'R':'R_ly', 'RBI':'RBI_ly', 'SB':'SB_ly', 'AVG':'BA_ly'},inplace=True)
    scb['playerid'] = scb['playerid'].astype(str)
    scb.to_csv('data/'+str(yr)+'-statcast-h.csv')
    
    scp = pitching_stats(yr, qual=0)
    scp.rename(columns={'IDfg':'playerid', 'IP':'IP_ly', 'QS':'QS_ly', 'ERA':'ERA_ly', 'WHIP':'WHIP_ly', 'SV':'SV_ly', 'HLD':'HLD_ly', 'SO':'SO_ly'},inplace=True)
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



def sort_by_other_list(list_to_sort, order_list):
    """Sorts a list based on the order of another list."""

    order_dict = {value: index for index, value in enumerate(order_list)}
    return sorted(list_to_sort, key=lambda x: order_dict.get(x, float('inf')))



def get_eligible_positions(pos, pos_order):
    # Expand player eligibility 
    eligibility = []
    for position in pos.split(','):
        if position=='C':
            eligibility.extend(['C', 'DH1', 'DH2'])
        if position=='1B':
            eligibility.extend(['1B', 'CI', 'DH1', 'DH2'])
        if position=='2B':
            eligibility.extend(['2B', 'MI', 'DH1', 'DH2'])
        if position=='3B':
            eligibility.extend(['3B', 'CI', 'DH1', 'DH2'])
        if position=='SS':
            eligibility.extend(['SS', 'MI', 'DH1', 'DH2'])
        if position=='OF':
            eligibility.extend(['OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'DH1', 'DH2'])
        if position=='DH':
            eligibility.extend(['DH1', 'DH2'])
        if position in ['SP', 'RP', 'P']:
            eligibility.extend(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'])
        if position in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
            eligibility.extend([position])
    
    # Remove duplicates and sort
    eligibility = list(set(list(dict.fromkeys(eligibility))))
    eligibility = sort_by_other_list(eligibility, pos_order)
    return eligibility


pos_order = ['C', 'MI', 'CI', 'DH1', 'DH2', '2B', '3B', 'SS', '1B', 'OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']

def check_roster_pos(player, roster, df, pos_order, is_draft_operation=False):
    """
    Place a player on the roster by finding a valid permutation of positions
    """
    print(f"Function called for: {player['Name']} (cbsid: {player['cbsid']})")
    
    # Skip duplicate processing only during draft operations, not during general roster building
    if is_draft_operation:
        # Check if this player is being processed for the first time in this draft operation
        player_id = f"{player['cbsid']}_{player['Owner']}"
        if hasattr(check_roster_pos, '_current_draft_player') and check_roster_pos._current_draft_player == player_id:
            print(f"Skipping duplicate processing for: {player['Name']} during draft")
            return []
        check_roster_pos._current_draft_player = player_id
    
    # Check if this player is already on the roster
    team_roster = roster[player['Owner']]
    existing_players = team_roster[team_roster != 0].values
    if player['Name'] in existing_players:
        print(f"{player['Name']} is already on the roster, skipping re-rostering")
        return []  # Return empty list to indicate no action needed
    
    eligibility = get_eligible_positions(player['Pos'], pos_order)
    print(f"Eligible positions for {player['Name']}: {eligibility}")
    
    # Get current team roster
    team_roster = roster[player['Owner']]
    
    # Find current players and their original positions
    current_players = team_roster[team_roster != 0]
    
    # Get all current rostered positions
    rostered_positions = current_players.index.tolist()
    rostered_positions = [p.strip().upper() for p in rostered_positions]
    print(f"Currently rostered positions for {player['Owner']}: {rostered_positions}")
    
    # First check rostered_positions for a free spot
    for position in eligibility:
        position = position.strip().upper()
        print(f"Checking position: {position}")
        if position not in rostered_positions:
            print(f"Found free spot for {player['Name']} at {position}")
            result = [{player['Name']: position}]
            print(f"Returning: {result}")
            return result
            
    print('No free spot found initially, looking for bumps')
    
    # Create a list of possible moves, sorted by optimal placement
    possible_moves = []
    
    # Loop through the drafted player's eligible positions
    for p in eligibility:
        # Find name of player currently in each roster spot
        used_pos_player_name = roster.loc[p, player['Owner']]
        
        # Skip empty spots (should be caught above, but just in case)
        if used_pos_player_name == 0 or not used_pos_player_name:
            print(f"Position {p} is actually free, assigning {player['Name']}")
            return [{player['Name']: p}]
            
        # Find the other positions this player could potentially move to
        player_pos_data = df[df['Name'] == used_pos_player_name]['Pos']
        if player_pos_data.empty:
            print(f"Warning: Could not find position data for {used_pos_player_name}")
            continue
            
        used_pos_player_positions = get_eligible_positions(player_pos_data.iloc[0], pos_order)
        print(f"{p} is currently used by {used_pos_player_name}. {used_pos_player_name} is also eligible at {used_pos_player_positions}")
        
        # Calculate versatility (number of positions a player can play)
        current_player_versatility = len(used_pos_player_positions)
        new_player_versatility = len(eligibility)
        
        # Loop through the already rostered player's eligible positions
        for p2 in used_pos_player_positions:
            p2 = p2.strip().upper()
            # If the rostered player can move to an open position
            if p2 not in rostered_positions:
                # Add this move to possible moves with score based on versatility difference
                # Higher score = better move (player with more positions should be in more flexible spot)
                score = current_player_versatility - new_player_versatility
                possible_moves.append({
                    'score': score,
                    'move': [{player['Name']: p}, {used_pos_player_name: p2}]
                })
    
    # Sort possible moves by score (highest first)
    possible_moves.sort(key=lambda x: x['score'], reverse=True)
    
    # If we found any possible moves, return the best one
    if possible_moves:
        best_move = possible_moves[0]['move']
        print(f"Best move found: {best_move}")
        return best_move
    
    print(f'Unable to roster {player["Name"]}')
    return [{player['Name']: None}]


def WORKS_check_roster_pos(player, roster, df, pos_order, is_draft_operation=False):
    """
    Place a player on the roster by finding a valid permutation of positions
    
    Args:
    roster: DataFrame with index of all possible positions
    name: str, player name
    team_name: str, name of team drafting player
    eligible: str, comma-separated positions the player is eligible to play
    pos_order: list of positions sorted by the order to try to place the player
    
    Returns:
    str or None: Assigned position, or None if no placement possible
    """
    print(f"Function called for: {player['Name']} (cbsid: {player['cbsid']})")

    # Skip duplicate processing only during draft operations, not during general roster building
    if is_draft_operation:
        # Check if this player is being processed for the first time in this draft operation
        player_id = f"{player['cbsid']}_{player['Owner']}"
        if hasattr(check_roster_pos, '_current_draft_player') and check_roster_pos._current_draft_player == player_id:
            print(f"Skipping duplicate processing for: {player['Name']} during draft")
            return []
        check_roster_pos._current_draft_player = player_id

    eligibility = get_eligible_positions(player['Pos'], pos_order)
    print(f"Eligible positions for {player['Name']}: {eligibility}")
    
    # Get current team roster
    team_roster = roster[player['Owner']]
    
    # Find current players and their original positions
    current_players = team_roster[team_roster != 0]
    
    # Get all current rostered positions
    rostered_positions = current_players.index.tolist()
    rostered_positions = [p.strip().upper() for p in rostered_positions]

    # Combine current players with new player
    #all_players = list(current_players.values) + [name]
    print(f"Function called for: {player['Name']}")
    # First check rostered_positions for a free spot
    for position in eligibility:
        position = position.strip().upper()
        #print(f"Checking position: {position}")
        if position not in rostered_positions:
            #roster.loc[position, player['Owner']] = player['Name']
            # End function looping b/c we found a place to put the player
            print(f"Found free spot for {player['Name']} at {position}")
            result = [{player['Name']: position}]
            print(f"About to return: {result}")
            return result
        
    print("No eligible positions found, function is ending without return.")

    # Loop through the drafted player's eligible positions
    #bump = False
    for p in eligibility:
        # Find name of player currently in each roster spot
        used_pos_player_name = roster.loc[p, player['Owner']]

        # Skip empty spots (should be caught above, but just in case)
        if used_pos_player_name == 0 or not used_pos_player_name:
            print(f"Position {p} is actually free, assigning {player['Name']}")
            return [{player['Name']: p}]
        
        # Find the other positions this player could potentially move to
        player_pos_data = df[df['Name'] == used_pos_player_name]['Pos']
        if player_pos_data.empty:
            print(f"Warning: Could not find position data for {used_pos_player_name}")
            continue

        # Find the other positions this player could potentially move to
        used_pos_player_positions = get_eligible_positions(df[df['Name']==used_pos_player_name]['Pos'].iloc[0], pos_order)
        print(f"{p} is currently used by {used_pos_player_name}. {used_pos_player_name} is also eligible at {used_pos_player_positions}")
        # Loop through the already rostered player's eligible positions
        for p2 in used_pos_player_positions:
            p2 = p2.strip().upper()
            # If the rostered player can move to an open position, put him there and put the newly drafted player in his spot
            if p2 not in rostered_positions:
                print(f"{used_pos_player_name} is eligible to move to {p2}. Moving {used_pos_player_name} to {p2}. Rostering {player['Name']} in {p}")
                #roster.loc[p2, player['Owner']] = used_pos_player_name
                #roster.loc[p, player['Owner']] = player['Name']
                # Set bump to True
                #bump = True
                print(f"{used_pos_player_name} is eligible to move to {p2}. Moving {used_pos_player_name} to {p2}. Rostering {player['Name']} in {p}")
                # Get out of the loop now that we found someone who can move
                return [{player['Name']:p}, {used_pos_player_name:p2}]
    
    print(f'Unable to roster {player["Name"]}')
    return [{player['Name']:None}]
        

