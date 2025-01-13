from bs4 import BeautifulSoup as bs4
from datetime import datetime
import math
import os
import pandas as pd
import re
from selenium.webdriver import Chrome
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import shutil
from sqlalchemy import create_engine
import time

class Scraper():
    def __init__(self):
        self.chromedriver_path = r"C:\ProgramData\Anaconda3\WebDriver\bin\chromedriver.exe"
        self.download_path = r'C:\Users\pddnh\Downloads'
        self.downloaded_fangraphs_filename = 'fangraphs-leaderboard-projections.csv'
        self.destination_path = r"C:\Users\pddnh\Documents\GitHub\xdl\data"
        
    def cbs_login(self):
        service = Service(executable_path=self.chromedriver_path)
        driver = webdriver.Chrome(service=service)
        driver.get('https://www.cbssports.com/login?master_product=150&xurl=https%3A%2F%2Fwww.cbssports.com%2Flogin')
        time.sleep(2)
        login_form = driver.find_element(By.ID, 'name')
        login_form.send_keys('gostros09')
        login_form = driver.find_element(By.NAME, 'password')
        login_form.send_keys('Segneri9A')
        time.sleep(4)
        # Wait until the "Continue" button is present and clickable
        continue_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]"))
        )
        # Click the "Continue" button
        continue_button.click()
        return driver
    
    def fangraphs_login(self):
        service = Service(executable_path=self.chromedriver_path)
        driver = webdriver.Chrome(service=service)
        driver.get('https://blogs.fangraphs.com/wp-login.php')
        driver.implicitly_wait(3)
        login_form = driver.find_element(By.ID, 'loginform')
        login = driver.find_element(By.ID, 'user_login')
        login.send_keys('Haueter19')
        time.sleep(1.2)
        pw = driver.find_element(By.ID, 'user_pass')
        pw.send_keys('Segneri9@')
        time.sleep(1.4)
        login_form.submit()
        return driver
    
    def cbs_auction_values(self):
        try:
            driver = self.cbs_login()
        except:
            print('Error logging in to CBS')
            return
        time.sleep(5)
        driver.get('https://xdl.baseball.cbssports.com/features/projected-salaries')
        time.sleep(5)
        # Get the HTML of the page
        html = driver.page_source
        soup = bs4(html, 'html.parser')
        # Find the specific table you want
        table = soup.find('table', {'class': 'data'})  # Use appropriate attributes to locate the table
        # Use pandas to read the HTML table
        df = pd.read_html(str(table), header=1, skiprows=0, extract_links='body')[0]  # [0] to get the first DataFrame if there are multiple
        # Extract the first element of each tuple in the MultiIndex
        new_header = [tup[0] for tup in df.columns]
        # Set the new header
        df.columns = new_header
        # Apply a lambda function to each column to extract the first element of each tuple
        df = df.apply(lambda col: [v[0] if v[1] is None else v for v in col])
        # Extract the player name and ID
        df['player'] = df['Name'].apply(lambda x: x[0])
        df['id'] = df['Name'].apply(lambda x: x[1])
        # Define the regex pattern
        pattern = r'^(?P<FullName>[A-Za-z\.\'\-\s]+) (?P<Positions>[A-Z0-9,]+) • (?P<Team>[A-Z]+)$'
        # Apply the regex pattern to the DataFrame
        df[['CBSNAME', 'Positions', 'Team']] = df['player'].str.extract(pattern)
        # Split the FullName into FirstName and LastName
        df[['FirstName', 'LastName']] = df['CBSNAME'].str.rsplit(n=1, expand=True)
        # Extract CBSID
        df['cbsid'] = df['id'].apply(lambda x: int(x.replace('/players/playerpage/', '')))
        # Change dollar values column to int
        for col in ['Mixed', 'AL-Only', 'NL-Only']:
            df[col] = df[col].apply(lambda x: int(x.replace('$','').replace('-','0')))
        # Rename the Mixed column
        df.rename(columns={'Mixed':'CBS'},inplace=True)
        # Output the DataFrame to a CSV in the data/ folder
        cols = ['cbsid','CBSNAME','FirstName','LastName','Positions','Pos','Team','CBS','AL-Only','NL-Only','Name','player','id']
        df[cols].to_csv(f'{self.destination_path}/{datetime.now().year}-cbs-auction-values.csv', index=False)
        driver.quit()
        return df
    

    def get_fangraphs_projections(self, system_name, stats_type, statgroup='fantasy', fantasypreset='roto5x5'):
        # Launch the scraper
        try:
            driver = self.fangraphs_login()
        except:
            print('Error logging in to Fangraphs')
            return
        
        driver.implicitly_wait(10)
        # Define the URL and get the page
        #https://www.fangraphs.com/projections?pos=all&stats=bat&type=steamer
        url = f"https://www.fangraphs.com/projections?pos=all&type={system_name}&statgroup={statgroup}&fantasypreset={fantasypreset}&stats={stats_type}"
        driver.get(url)
        time.sleep(1.9)
        # Click on the export button
        export_button = driver.find_element(By.CLASS_NAME, 'data-export')
        export_button.click()
        # Set filename
        filename = f"{datetime.now().year}-{system_name}-proj-{'h' if stats_type=='bat' else 'p'}.csv"
        time.sleep(10)
        # Rename downloaded file appropriately
        os.rename(os.path.join(self.download_path,self.downloaded_fangraphs_filename), os.path.join(self.download_path,filename))
        # Move file to data/ folder
        shutil.move(os.path.join(self.download_path,filename), os.path.join(self.destination_path,filename))
        return f'{filename} saved in {self.destination_path}'