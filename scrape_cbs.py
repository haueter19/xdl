from datetime import datetime
import time
import re
from bs4 import BeautifulSoup as bs4
from selenium.webdriver import Chrome
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import fantasy_utils as fu
fp = fu.Fantasy_Projections()

proj_url = 'https://xdl.baseball.cbssports.com/stats/stats-main'
elig_url = 'https://xdl.baseball.cbssports.com/teams/eligibility/'

eligibility_url_dict = {"9 Grand Kids":'30', 'Brewbirds':'3', 'Charmer':'36', 'Dirty Birds':'41', "Harveys Wallbangers":'4', 'Lil Trump & the Ivanabees':'27', 'Lima Time!':'38', 
                        'Roid Ragers':'44', 'Trouble with the Curve':'1', 'Ugly Spuds':'29', 'Wiscompton Wu-Tang':'42', 'Young Guns':'45'}

name_change = {"9 Gran...":'Madness', 'Brewbi...':'Brewbirds', 'Charme...':'Charmer', 'Dirty ...':'Dirty Birds', 'Harvey...':'Harvey', 'Lil Tr...':'Lil Trump', 
               'Lima T...':'Lima Time', 'Roid R...':'Roiders', 'Troubl...':'Trouble', 'Ugly S...':'Ugly Spuds', 'Wiscom...':'Wu Tang', 'Young ...':'Young Guns'}
position_priority = ['C', '2B', '3B', 'SS', 'OF', '1B', 'MI', 'CI', 'DH', 'SP', 'RP']

opening_week = int(datetime.strftime(pd.Timestamp('2023-03-30'), '%W')) - 1

def clean_table(df, num):
    df = df.copy()
    if num % 2 == 0:
        df.columns = df.columns.droplevel().droplevel()  
    else:
        df.columns = df.columns.droplevel()
    
    df.drop(columns='Action', inplace=True)
    df = df.iloc[:df.index.stop-1]
    df['Team'] = df.Player.apply(lambda x: x.split()[-1])
    df['Pos'] = df.Player.apply(lambda x: x.split()[-3])
    df['Player'] = df.Player.apply(lambda x: ' '.join(x.split()[:-3]))
    df['Owner'] = df['Avail'].replace(name_change)
    df.loc[df['Rank']<=50, 'top_50'] = 1
    df.loc[df['Rank']<=100, 'top_100'] = 1
    df.loc[df['Rank']<=276, 'top_276'] = 1
    df.loc[df['Rank']>276, 'above_276'] = 1
    if num % 2 == 0:
        df.loc[:13, 'Start'] = 1
    else:
        df.loc[:8, 'Start'] = 1
        df.rename(columns={'K':'SO', 'BB':'BBa', 'H':'Ha'},inplace=True)
        df['Sv+Hld'] = df['S'] + df['HD']
        df['ER'] = df['INNs']*df['ERA']/9
    
    df.fillna(0,inplace=True)
    return df



def process_eligibility(val, driver):
    driver.get(elig_url+val)
    time.sleep(2.2)
    html = driver.page_source
    soup = bs4(html, 'html.parser')
    elig = pd.read_html(str(soup.find_all('table')))[0]
    elig = elig.iloc[:-1,:]
    elig['Player'] = elig['Player'].apply(lambda x: x[:re.search(r'RP|SP|1B|2B|3B|SS|LF|CF|RF|DH|C\s', x).span()[0]])
    elig.fillna(0,inplace=True)
    elig.loc[:,1:] = elig.iloc[:,1:].astype('int32')
    return elig



def stitch_positions(row):
    pos_code = row[position_priority+['P']]>=5
    return list(pos_code[pos_code].index)



def parse_projections_data(master):
    df = pd.DataFrame()
    for n in range(len(master)):
        df = pd.concat([df, clean_table(master[n], n)])
    
    df.fillna(0,inplace=True)
    df.rename(columns={'AVG':'BA', 'INNs':'IP'}, inplace=True)
    df.drop(columns=['1B', '2B', '3B'],inplace=True)
    df = df.reset_index()
    
    elig = pd.read_csv('data/eligibility.csv')
    df = df.merge(elig, on='Player', how='left')

    df.loc[(df['Pos'].isin(['LF', 'CF', 'RF'])) & (df['OF']<5), 'OF'] = 5
    df.loc[(df['Pos'].isin(['1B'])), '1B'] = 5
    df.loc[(df['Pos'].isin(['2B'])), '2B'] = 5
    df.loc[(df['Pos'].isin(['3B'])), '3B'] = 5
    df.loc[(df['Pos'].isin(['SS'])), 'SS'] = 5
    df.loc[(df['Pos'].isin(['C'])), 'C'] = 5
    df.loc[(df['Pos'].isin(['SP', 'RP'])), 'P'] = 5

    
    df.loc[(df['SP']>=5) | (df['RP']>=5) | (df['P']>=5), 'type'] = 'p'
    df['type'].fillna('h',inplace=True)
    df.loc[(df['Player']=='Shohei Ohtani') & (df['AB']>=1), 'type'] = 'h'

    h_mask = (df['Start']==1) & (df['type']=='h')
    hitters = (df['type']=='h')
    p_mask = (df['Start']==1) & (df['type']=='p')
    pitchers = (df['type']=='p')

    df.loc[hitters, 'DH'] = 5
    df.loc[pitchers, 'P'] = 5
    df['P'].fillna(0,inplace=True)

    lgBA = df[h_mask]['H'].sum()/df[h_mask]['AB'].sum()
    df.loc[hitters, 'zlgBA'] = (df[hitters]['H'] - (df[hitters]['AB'] * lgBA))

    lgERA = df[p_mask]['ER'].sum()/df[p_mask]['IP'].sum()*9
    df.loc[pitchers, 'zlgERA'] = (((df[pitchers]['ER']*9) - (df[pitchers]['IP']*lgERA))*-1)

    lgWHIP = (df[p_mask]['BBa'].sum()+df[p_mask]['Ha'].sum())/df[p_mask]['IP'].sum()
    df.loc[pitchers, 'zlgWHIP'] = (((df.loc[pitchers]['Ha']+df.loc[pitchers]['BBa'])-(df.loc[pitchers]['IP']*lgWHIP))*-1)
    
    q = df[h_mask][['HR', 'RBI', 'R', 'SB', 'H', 'AB']].describe().T[['mean', 'std']].T.to_dict()
    q['BA'] = {'mean':df[h_mask]['H'].sum()/df[h_mask]['AB'].sum(), 'std':df[h_mask]['BA'].std()}
    q.update(df[p_mask][['W', 'SO', 'Sv+Hld', 'IP', 'ER', 'BBa', 'Ha']].describe().T[['mean', 'std']].T.to_dict())
    q['ERA'] = {'mean':(df[p_mask]['ER'].sum()/df[p_mask]['IP'].sum())*9, 'std':(df[p_mask]['ER']/df[p_mask]['IP']*9).std()}
    q['WHIP'] = {'mean':(df[p_mask]['BBa'].sum()+df[p_mask]['Ha'].sum())/df[p_mask]['IP'].sum(), 'std':((df[p_mask]['BBa']+df[p_mask]['Ha'])/df[p_mask]['IP']).std()}
    q['zlgBA'] = {'mean':df[h_mask]['zlgBA'].mean(), 'std':df[h_mask]['zlgBA'].std()}
    q['zlgERA'] = {'mean':df[p_mask]['zlgERA'].mean(), 'std':df[p_mask]['zlgERA'].std()}
    q['zlgWHIP'] = {'mean':df[p_mask]['zlgWHIP'].mean(), 'std':df[p_mask]['zlgWHIP'].std()}

    for stat in ['R', 'HR', 'RBI', 'SB', 'zlgBA']:
        df.loc[hitters, 'z'+stat] = df[hitters].apply(lambda row: fp.big_board(row, stat, q), axis=1)

    df.loc[hitters & (df['zlgBA'].isna()), 'zlgBA'] = df['zlgBA'].min()-.01
    df.loc[hitters & (df['zzlgBA'].isna()), 'zzlgBA'] = df['zzlgBA'].min()-.01
    df.loc[hitters, 'BIGAAh'] = df[hitters]['zR'] + df[hitters]['zRBI'] + df[hitters]['zHR'] + df[hitters]['zSB'] + df[hitters]['zzlgBA']

    for stat in ['W', 'SO', 'Sv+Hld', 'zlgERA', 'zlgWHIP']:
        df.loc[pitchers, 'z'+stat] = df[pitchers].apply(lambda row: fp.big_board(row, stat, q), axis=1)

    df.loc[pitchers & (df['zzlgERA'].isna()), 'zzlgERA'] = df['zzlgERA'].min()-.01
    df.loc[pitchers & (df['zzlgWHIP'].isna()), 'zzlgWHIP'] = df['zzlgWHIP'].min()-.01
    df.loc[pitchers, 'BIGAAp'] = df[pitchers]['zW']+df[pitchers]['zSO']+df[pitchers]['zSv+Hld']+df[pitchers]['zzlgERA']+df[pitchers]['zzlgWHIP']

    df['BIGAAh'].fillna(0,inplace=True)
    df['BIGAAp'].fillna(0,inplace=True)
    df['z'] = df['BIGAAh']+df['BIGAAp']
    
    ohtani_idx = df[df['Player']=='Shohei Ohtani'].index
    df.loc[ohtani_idx[0], 'z'] = df.loc[ohtani_idx[0],'BIGAAh'] + 0
    df.loc[ohtani_idx[1], 'z'] = 0 + df.loc[ohtani_idx[1],'BIGAAp']
    df.loc[ohtani_idx[1], ['AB', 'R', 'H', 'HR', 'RBI', 'BB', 'K', 'SB', 'CS', 'BA', 'OBP', 'SLG', 'zlgBA', 'zR', 'zHR', 'zRBI', 'zSB', 'zzlgBA', 'BIGAAh']] = df.loc[ohtani_idx[0]][['AB', 'R', 'H', 'HR', 'RBI', 'BB', 'K', 'SB', 'CS', 'BA', 'OBP', 'SLG', 'zlgBA', 'zR', 'zHR', 'zRBI', 'zSB', 'zzlgBA', 'BIGAAh']]
    df = df.loc[df.index!=ohtani_idx[0]].copy()

    df['all_pos'] = df.apply(lambda x: stitch_positions(x), axis=1)
    return df


def login():
    driver = webdriver.Chrome("C:\\ProgramData\\Anaconda3\\WebDriver\\bin\\chromedriver.exe")
    driver.get('https://www.cbssports.com/login?master_product=150&xurl=https%3A%2F%2Fwww.cbssports.com%2Flogin')
    search_box = driver.find_element_by_id('userid')
    search_box.send_keys('gostros09')
    search_box = driver.find_element_by_id('password')
    search_box.send_keys('Segneri9A')
    search_box.submit()
    return driver



def scrape_proj(driver):
    time.sleep(2)
    driver.get(proj_url)

    # This clicks the Teams filter and opens it
    time.sleep(3)
    driver.find_element(By.XPATH, "//li[@class=' fantasyTeams dropdown selected_arrow large opt ']").click()

    # This chooses the All Teams filter
    time.sleep(4)
    driver.find_element(By.XPATH, "//a[@aria-label='All Teams ']").click()

    time.sleep(3)
    html = driver.page_source
    soup = bs4(html, 'html.parser')
    cur_proj = pd.read_html(str(soup.find_all('table')[1:25]))

    # This clicks on the Timeframe pulldown menu
    time.sleep(4)
    driver.find_element(By.XPATH, "//div[@class='pageFilter subFilters dropdown large']").click()
    # This chooses the Rest of Season option
    time.sleep(4)
    driver.find_element(By.XPATH, "//li[@value='/stats/stats-main/restofseason']").click()

    time.sleep(5)
    html = driver.page_source
    soup = bs4(html, 'html.parser')
    ros_proj = pd.read_html(str(soup.find_all('table')[1:25]))
    return cur_proj, ros_proj



def scrape_eligibility(driver):    
    elig = pd.DataFrame()
    for key, val in eligibility_url_dict.items():
        driver.get(elig_url+val)
        time.sleep(3)
        html = driver.page_source
        soup = bs4(html, 'html.parser')
        temp = pd.read_html(str(soup.find_all('table')))[0]
        temp = temp.iloc[:-1,:]
        temp['Player'] = temp['Player'].apply(lambda x: x[:re.search(r'RP|SP|1B|2B|3B|SS|LF|CF|RF|DH|C\s', x).span()[0]])
        temp.fillna(0,inplace=True)
        temp.iloc[:,1:] = temp.iloc[:,1:].astype('int32')
        
        elig = pd.concat([elig, process_eligibility(val, driver)])
    
    elig.to_csv('eligibility.csv', index=False)
    return elig



if __name__=='__main__':
    driver = login()
    cur_week = 'period_'+str(int(datetime.strftime(datetime.now(), '%W')) - opening_week)

    #elig = scrape_eligibility(driver)
    #print('created eligibility.csv')
    
    cur_proj, ros_proj = scrape_proj(driver)
    cur_df = parse_projections_data(cur_proj)
    cur_df.to_csv('data/'+cur_week+'_projections.csv', index=False)
    print(f'created {cur_week}_projections.csv')
    
    ros_df = parse_projections_data(ros_proj)
    ros_df.to_csv('data/'+cur_week+'_ros_projections_.csv', index=False)
    driver.quit()
    print(f'created {cur_week}_ros_projections.csv')
