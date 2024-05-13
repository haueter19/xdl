import pandas as pd
import math
from datetime import datetime
from sqlalchemy import create_engine

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
        
        self.proj_systems = ['atc', 'thebat', 'dc', 'steamer', 'zips']
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
        pos_list = p.split('/')
        for i in self.pos_hierarchy:
            if i in pos_list:
                return i

    def get_qual_avgs(self, previous_season_stats_hitter, previous_season_stats_pitcher, yr=datetime.now().year, min_pa=440, sp_min_ip=140, rp_ip_range=[48,90], min_sv_hld=5):
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
            previous_season_stats_hitter = pd.read_csv('data/'+str(yr-1)+'-final-stats-h.csv', encoding='latin1').sort_values('PA', ascending=False)
        
        previous_season_stats_hitter = previous_season_stats_hitter[previous_season_stats_hitter['PA']>min_pa]
        lgBA = previous_season_stats_hitter['H'].sum()/previous_season_stats_hitter['AB'].sum()
        previous_season_stats_hitter['zlgBA'] = previous_season_stats_hitter.apply(lambda x: x['H']-(x['AB']*(lgBA)), axis=1)
        quals_h = previous_season_stats_hitter[['H', 'AB', 'PA', 'zlgBA', 'R', 'RBI', 'HR', 'SB']].describe().to_dict()
        
        if isinstance(previous_season_stats_pitcher,pd.DataFrame):
            for stat in ['BB', 'HA', 'ER', 'IP', 'SO', 'W', 'Sv+Hld']:
                assert stat in previous_season_stats_pitcher.columns
        else:
            previous_season_stats_pitcher = pd.read_csv('data/'+str(yr-1)+'-final-stats-p.csv', encoding='latin1').sort_values('IP', ascending=False)
        
        previous_season_stats_pitcher['playerid'] = previous_season_stats_pitcher['playerid'].astype(str)
        previous_season_stats_pitcher['Sv+Hld'] = previous_season_stats_pitcher['SV']+previous_season_stats_pitcher['HLD']
        val_p = pd.read_csv('data/'+str(yr)+'-fangraphs-auction-calculator-p.csv')
        val_p['playerid'] = val_p['playerid'].astype(str)
        val_p['Primary_Pos'] = val_p['POS'].apply(lambda x: self.find_primary_pos(x))
        previous_season_stats_pitcher = previous_season_stats_pitcher.merge(val_p[['playerid', 'Primary_Pos']], on='playerid', how='inner')
        previous_season_stats_pitcher = previous_season_stats_pitcher[(previous_season_stats_pitcher['Primary_Pos']=='RP') & (previous_season_stats_pitcher['IP'].between(rp_ip_range[0],rp_ip_range[1]) & (previous_season_stats_pitcher['Sv+Hld']>min_sv_hld)) | (previous_season_stats_pitcher['Primary_Pos']=='SP') & (previous_season_stats_pitcher['IP']>sp_min_ip)]
        lgERA = previous_season_stats_pitcher['ER'].sum()/previous_season_stats_pitcher['IP'].sum()*9
        lgWHIP = (previous_season_stats_pitcher['BB'].sum()+previous_season_stats_pitcher['HA'].sum())/previous_season_stats_pitcher['IP'].sum()
        previous_season_stats_pitcher['zlgERA'] = previous_season_stats_pitcher.apply(lambda x: ((x['ER']*9) - (x['IP']*lgERA))*-1, axis=1)
        previous_season_stats_pitcher['zlgWHIP'] = previous_season_stats_pitcher.apply(lambda x: ((x['HA']+x['BB'])-(x['IP']*lgWHIP))*-1, axis=1)
        quals_p = previous_season_stats_pitcher[['BB', 'HA', 'ER', 'IP', 'SO', 'W', 'Sv+Hld', 'zlgERA', 'zlgWHIP']].describe().to_dict()
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
            for stat in ['ERA', 'WHIP', 'SO', 'W', 'Sv+Hld', 'ERA_ly', 'WHIP_ly', 'SO_ly', 'W_ly', 'Sv+Hld_ly']:
                df['z'+stat] = df.apply(lambda row: self.big_board(row, stat, self.quals), axis=1)
            df['BIGAAp'] = df['zERA']+df['zWHIP']+df['zW']+df['zSO']+df['zSv+Hld']
            df['z_p_ly'] = df['zERA_ly']+df['zWHIP_ly']+df['zW_ly']+df['zSO_ly']+df['zSv+Hld_ly']
            return df
        
    
    def prep_projection_data(self):
        """
        Function to load various csv files containing projections for the next season. Data is expected to be downloaded from Fangraphs. 
        """
        # Loop through possible projection csv files. Load, if they exist, into a list and then concatenate into a dataframe
        proj_system_list = []
        for proj_file in self.proj_systems:
            try:
                temp = pd.read_csv('data/'+str(self.yr)+'-'+proj_file+'-proj-h.csv', encoding="latin-1")

                proj_system_list.append(temp)
            except:
                print(f"Did not find file {proj_file}")
                pass
    
        h = pd.concat(proj_system_list).sort_values('PlayerId')
        h.rename(columns={'PlayerId':'playerid'},inplace=True)
    
        # Add Fangraphs auction values. Serves as a reference point to see if my projections seem reasonable
        try:
            val_h = pd.read_csv('data/'+str(self.yr)+'-fangraphs-auction-calculator-h.csv')
            val_h.rename(columns={'PlayerId':'playerid', 'POS':'Pos'},inplace=True)
            h = h.merge(val_h[['playerid', 'Pos', 'Dollars']])
        except:
            h['Dollars'] = 0
        
        # Add CBS auction values. Another reference point. Assumption is that lazy owners might only consult this guide.
        try:
            cbs = pd.read_csv('data/'+str(self.yr)+'-cbs-values.csv', encoding="latin-1")
            h = h.merge(cbs[['playerid', 'CBS']], on='playerid', how='left')
        except:
            h['CBS'] = 0

        h.drop(columns=['wOBA', 'CS', 'Fld', 'BsR', 'ADP'],inplace=True)        
        h = h.rename(columns={'ï»¿Name':'Name'})
        h['Primary_Pos'] = h.apply(lambda x: self.find_primary_pos(x['Pos']), axis=1)

        # With the csv's loaded, we now must collapse the players into one stat line. This pivot table takes the averages
        proj = pd.pivot_table(h, index='playerid', values=['G', 'PA', 'AB', 'H', 'HR', 'R', 'RBI', 'SB'], aggfunc='mean')\
            .merge(h[['playerid', 'Name', 'Team', 'Pos', 'Primary_Pos', 'Dollars', 'CBS']], on='playerid', how='inner').drop_duplicates()
        # Helps with sorting later when we need to find the top 12 players at a position. 
        proj['sorter'] = proj['HR']+proj['R']+proj['RBI']+proj['H']+proj['SB']
        proj['BA'] = proj['H']/proj['AB']
        proj = proj.drop_duplicates(subset='playerid')
        proj.CBS.fillna(0,inplace=True)

        # Pitching section ---- repeat similar process
        proj_system_list = []
        for proj_file in self.proj_systems:
            try:
                temp = pd.read_csv('data/'+str(self.yr)+'-'+proj_file+'-proj-p.csv', encoding="latin-1")
                proj_system_list.append(temp)
            except:
                pass
        p = pd.concat(proj_system_list).sort_values('PlayerId')
        p.rename(columns={'PlayerId':'playerid'},inplace=True)
        
        # Add Fangraphs auction values
        try:
            val_p = pd.read_csv('data/'+str(self.yr)+'-fangraphs-auction-calculator-p.csv')
            val_p.rename(columns={'PlayerId':'playerid', 'POS':'Pos'},inplace=True)
            p = p.merge(val_p[['playerid', 'Pos', 'Dollars']], how='inner')
        except:
            p['Dollars'] = 0
        
        
        # Add CBS values
        try:
            p = p.merge(cbs[['playerid', 'CBS']], on='playerid', how='left')
        except:
            p['CBS'] = 0
        

        # Little clean up
        p.rename(columns={'H':'HA'},inplace=True)
        p['Sv+Hld'] = p['SV']+p['HLD']
        p['Primary_Pos'] = p['Pos'].apply(lambda x: ', '.join(x.split('/')))
        p = p.rename(columns={'ï»¿Name':'Name'})


        # Collapse pitchers into one per row
        pproj = pd.pivot_table(p, index='playerid', values=['GS', 'G', 'IP', 'ER', 'HA', 'SO', 'BB', 'W', 'SV', 'HLD', 'Sv+Hld'], aggfunc='mean')\
            .merge(p[['playerid', 'Name', 'Team', 'Pos', 'Dollars', 'CBS']], on='playerid', how='inner').drop_duplicates()
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
        pproj.CBS.fillna(0,inplace=True)


        ######## I should refactor this section. Pull this out and use a different function to apply CBSID to data downloaded from Fangraphs ###########
        # 
        ids = self.load_id_map()

        h = proj.merge(ids[['IDFANGRAPHS', 'TEAM', 'CBSNAME']], left_on=['playerid', 'Team'], right_on=['IDFANGRAPHS', 'TEAM'], how='left').drop(columns=['IDFANGRAPHS', 'TEAM'])
        #h.loc[h['CBSNAME_y'].notna(), 'Name'] = h.loc[h['CBSNAME_y'].notna()]['CBSNAME_y']
        #h['Name'].fillna(h['CBSNAME_x'],inplace=True)
        #h['Name'].fillna(h['CBSNAME_y'],inplace=True)
        #h.drop(columns=['CBSNAME_x', 'CBSNAME_y'],inplace=True)

        p = pproj.merge(ids[['IDFANGRAPHS', 'TEAM', 'CBSNAME']], left_on=['playerid', 'Team'], right_on=['IDFANGRAPHS', 'TEAM'], how='left').drop(columns=['IDFANGRAPHS', 'TEAM'])
        #p.loc[p['CBSNAME_y'].notna(), 'Name'] = p.loc[p['CBSNAME_y'].notna()]['CBSNAME_y']
        #p['Name'].fillna(p['CBSNAME_x'],inplace=True)
        #p['Name'].fillna(p['CBSNAME_y'],inplace=True)
        #p.drop(columns=['CBSNAME_x', 'CBSNAME_y'],inplace=True)

        # Adding StatCast data
        try:
            scb = pd.read_csv('data/'+str(self.yr-1)+'-statcast-h.csv')
            scb.rename(columns={'IDfg':'playerid', 'PA':'PA_ly', 'HR':'HR_ly', 'R':'R_ly', 'RBI':'RBI_ly', 'SB':'SB_ly', 'AVG':'BA_ly'},inplace=True)
            scb['playerid'] = scb['playerid'].astype(str)
            scp = pd.read_csv('data/'+str(self.yr-1)+'-statcast-p.csv')
            scp.rename(columns={'IDfg':'playerid', 'IP':'IP_ly', 'W':'W_ly', 'ERA':'ERA_ly', 'WHIP':'WHIP_ly', 'SV':'SV_ly', 'HLD':'HLD_ly', 'SO':'SO_ly'},inplace=True)
            scp['playerid'] = scp['playerid'].astype(str)
            scp['Sv+Hld_ly'] = scp['SV_ly']+scp['HLD_ly']
            sc = pd.concat([scb,scp])
            h = h.merge(sc[['playerid', 'Age', 'PA_ly', 'H_ly', 'AB_ly', 'HR_ly', 'SB_ly', 'R_ly', 'RBI_ly', 'BA_ly', 'BB%', 'K%', 'Contact%', 'O-Contact%', 'Z-Contact%', 'O-Swing%', 'Z-Swing%', 'Swing%', 'Hard%+', 'Hard%', 'HardHit%', 'EV', 'maxEV', 'LA', 'Barrels', 'Barrel%', 'Events', 'xBA', 'xSLG', 'xwOBA', 'IP_ly', 'ER_ly', 'HA_ly', 'BB_ly', 'HBP_ly', 'ERA_ly', 'WHIP_ly', 'Sv+Hld_ly', 'W_ly', 'SO_ly', 'CSW%', 'SIERA', 'FIP', 'xFIP', 'xERA', 'ERA-', 'FBv', 'HR/9', 'BB/9']], on='playerid',how='left')
            p = p.merge(sc[['playerid', 'Age', 'PA_ly', 'H_ly', 'AB_ly', 'HR_ly', 'SB_ly', 'R_ly', 'RBI_ly', 'BA_ly', 'BB%', 'K%', 'Contact%', 'O-Contact%', 'Z-Contact%', 'O-Swing%', 'Z-Swing%', 'Swing%', 'Hard%+', 'Hard%', 'HardHit%', 'EV', 'maxEV', 'LA', 'Barrels', 'Barrel%', 'Events', 'xBA', 'xSLG', 'xwOBA', 'IP_ly', 'ER_ly', 'HA_ly', 'BB_ly', 'HBP_ly', 'ERA_ly', 'WHIP_ly', 'Sv+Hld_ly', 'W_ly', 'SO_ly', 'CSW%', 'SIERA', 'FIP', 'xFIP', 'xERA', 'ERA-', 'FBv', 'HR/9', 'BB/9']], on='playerid',how='left')
        except:
            lyh = pd.read_csv('data/'+str(self.yr-1)+'-final-stats-h.csv', encoding='latin1')
            lyh.rename(columns={'PlayerID':'playerid', 'PA':'PA_ly', 'AB':'AB_ly', 'HR':'HR_ly', 'R':'R_ly', 'RBI':'RBI_ly', 'SB':'SB_ly', 'BB':'BB_ly', 'H':'H_ly', 'AVG':'BA_ly', 'HBP':'HBP_ly'},inplace=True)
            lyh['playerid'] = lyh['playerid'].astype(str)

            lyp = pd.read_csv('data/'+str(self.yr-1)+'-final-stats-p.csv', encoding='latin1')
            lyp.rename(columns={'PlayerID':'playerid', 'IP':'IP_ly', 'W':'W_ly', 'ER':'ER_ly', 'ERA':'ERA_ly', 'WHIP':'WHIP_ly', 'HA':'HA_ly', 'SV':'SV_ly', 'HLD':'HLD_ly', 'SO':'SO_ly'},inplace=True)
            lyp['playerid'] = lyp['playerid'].astype(str)
            lyp['Sv+Hld_ly'] = lyp['SV_ly']+lyp['HLD_ly']

            ly = pd.concat([lyh,lyp])
            h = h.merge(ly[['playerid', 'PA_ly', 'H_ly', 'AB_ly', 'HR_ly', 'SB_ly', 'R_ly', 'RBI_ly', 'BA_ly', 'BB_ly', 'HBP_ly']], on='playerid',how='left')
            p = p.merge(ly[['playerid', 'PA_ly', 'H_ly', 'AB_ly', 'HR_ly', 'SB_ly', 'R_ly', 'RBI_ly', 'BA_ly', 'IP_ly', 'ER_ly', 'HA_ly', 'BB_ly', 'HBP_ly', 'ERA_ly', 'WHIP_ly', 'Sv+Hld_ly', 'W_ly', 'SO_ly']], on='playerid',how='left')

        #
        h.loc[(h['playerid'].duplicated()) & (h['Primary_Pos'].isin(['C', '1B', '2B', '3B', 'SS', 'OF', 'DH'])), 'keep'] = 0
        h['keep'].fillna(1,inplace=True)
        h = h[h['keep']==1]

        p.loc[(p['playerid'].duplicated()) & (p['Primary_Pos'].isin(['SP', 'RP', 'P'])), 'keep'] = 0
        p['keep'].fillna(1,inplace=True)
        p = p[p['keep']==1]

        self.hitting_data = h
        self.pitching_data = p
        return h, p

    def calc_value(self, previous_season_hitting=None, previous_season_pitching=None):
        """
        Calculate the auction value of the provided stats
        """

        # If hitting and pitching data was not provided, that means we need to create it using the prep_projection_data function
        if self.hitting_data is None:
            self.prep_projection_data()

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
        h['Pos_adj'] = h['Pos'].apply(lambda x: max([adj_dict[i] for i in x.split('/')]))
        p['Pos_adj'] = p['Pos'].apply(lambda x: max([adj_dict[i] for i in x.split('/')]))
        
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
        print('Completed applying auction values')
        return b.sort_values('Value', ascending=False)
    
    def upload(self, tbl):
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///fantasy_data.db', echo=False)
        self.data.to_sql(tbl, engine, if_exists='replace')
        return





def owners(conv, n_teams=12, tm_players=23):
    tot_dollars = n_teams * 260
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
