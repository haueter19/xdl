# XDL Fantasy Baseball Analysis
### xdl is a fantasy baseball league
### This project attempts to analyze various aspects of the league including the draft, trades, and weekly sit/start decisions

The results of the analysis are displayed in a streamlit application that can be accessed from this [link](https://xdl-summary.streamlit.app/)

### About the league
In 2023, there were 12 teams in the league. Each team drafted 23 players via auction (14 hitters and 9 pitchers) plus an additional 10 bench players. Each team starts with $260 to spend. The league follows a weekly change structure and there were 26 weeks in the season. One player is allowed to be picked up each week (with a corresponding player dropped). Scoring is based on starting players' stats each week. Hitting statistical categories are R, RBI, HR, SB, and BA. Pitching categories are W, K, Saves plus Holds, ERA, and WHIP.

### Data Prep
I got the data in various manners. The league's web site contains each team's weekly roster and whether or not a player was started or on the bench. I wrote a script using the Selenium package in Python to scrape data from the web site for each of the 12 teams' 26 weeks. I also scraped the stats for all players every week. I was able to get the auction cost of each player from this process as well. With the data in hand, I used Pandas, SQLAlchemy, and SQLite to create a database with tables to store everything. Views were created which show stats by week or year, often by team owner. I also created views to calculate z-scores for yearl and weekly data. Finally, I built an optimization function to sort through a team's roster and build the best possible, legal line-up. I could use this optimization function to analyze how well a particular team used their roster on a weekly basis.

### Draft Assistant Interface
This is a web page served by a FastAPI backend. I bring together various projection systems to create my own. Then I calculate auction values. On draft day, I can use this to track players that are still available based on position and tier. I can see the money situation to better understand if other teams are over spending or under spending. And I can see a quick list of comps for any player. 
