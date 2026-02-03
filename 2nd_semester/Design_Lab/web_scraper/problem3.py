import sqlite3
import os
import requests
from bs4 import BeautifulSoup
import re
from multiprocessing import Process, current_process

DB_NAME = "EPL.db"
URL = "https://en.wikipedia.org/wiki/Premier_League"


class EPLScraper:
    def __init__(self):
        self.db_name = DB_NAME
        self.url = URL
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        self.conn = sqlite3.connect(self.db_name)
        self.c = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def create_table(self):
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS EPL_season2425_clubdetails (
                club_name TEXT PRIMARY KEY,
                club_url TEXT,
                position INTEGER,
                first_season TEXT,
                season_in_top_division INTEGER,
                season_in_PL INTEGER,
                no_of_seasons_in_current_spell INTEGER,
                top_division_title INTEGER,
                most_recent_top_division_title INTEGER,
                first_team_players INTEGER,
                players_on_loan INTEGER,
                num_managers INTEGER,
                avg_manager_win_percentage REAL,
                DONE_OR_NOT_DONE INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def fetch_base_page(self):
        response = requests.get(self.url, headers=self.headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def print_table(self):
        self.c.execute("SELECT * FROM EPL_season2425_clubdetails")
        for row in self.c.fetchall():
            print(row)
        self.conn.close()

    def fill_common_data(self, soup):
        tables = soup.find_all('table')[9]
        rows = tables.find_all('tr')[1:]
        season_data = []
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 8:
                club_name = columns[0].find('a')['title']
                club_slug = columns[0].find('a')['href'].split('/')[-1]
                club_url = f"https://en.wikipedia.org/wiki/{club_slug}"
                position = columns[1].get_text(strip=True)
                first_season = columns[3].get_text(strip=True)
                season_in_top_division = columns[4].get_text(strip=True)
                season_in_PL = columns[5].get_text(strip=True)
                season_in_current_spell = columns[7].get_text(strip=True)
                top_division_title = columns[8].get_text(strip=True)
                most_recent_top_division_title = columns[9].get_text(strip=True)
                first_team_players = -1
                players_on_loan = -1
                num_managers = -1
                avg_manager_win_percentage = -1
                season_data.append({
                    'Club Name': club_name,
                    'Club URL': club_url,
                    'Current Standing': position,
                    'First Season': first_season,
                    'Season in Top Division': season_in_top_division,
                    'Season in PL': season_in_PL,
                    'Season in Current Spell': season_in_current_spell,
                    'Top Division Title': top_division_title,
                    'Most Recent Top Division Title': most_recent_top_division_title,
                    'First Team Players': first_team_players,
                    'Players on Loan': players_on_loan,
                    'Number of Managers': num_managers,
                    'Average Manager Win Percentage': avg_manager_win_percentage
                })
        return season_data

    def populate_club_specific_details(self, data):
        for row in data:
            club_url = row['Club URL']
            response = requests.get(club_url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            #Find First-team squad with id="First-team_squad" text and select the just next table
            first_team_squad = soup.find('h3', {'id': 'First-team_squad'}).find_next('table')
            #Count the total no of rows combining the table inside if any
            first_team_players = len(first_team_squad.find_all('tr'))
            print(f"First Team Players: {first_team_players}")


if __name__ == "__main__":
    scraper = EPLScraper()

    #---Fetch base page---
    soup = scraper.fetch_base_page()

    #---Create table---
    scraper.create_table()

    #---Scrape club data and populate the table with common data---
    table_data = scraper.fill_common_data(soup)

    #---Scraper each club page and populate the table with club specific data---
    scraper.populate_club_specific_details(table_data)

    #Print the heading with a tab
    print("Club Name\tClub URL\tCurrent Standing\tFirst Season\tSeason in Top Division\tSeason in PL\tSeason in Current Spell\tTop Division Title\tMost Recent Top Division Title\tFirst Team Players\tPlayers on Loan\tNumber of Managers\tAverage Manager Win Percentage")
    for row in data:
        print(f"{row['Club Name']}\t{row['Club URL']}\t{row['Current Standing']}\t{row['First Season']}\t{row['Season in Top Division']}\t{row['Season in PL']}\t{row['Season in Current Spell']}\t{row['Top Division Title']}\t{row['Most Recent Top Division Title']}\t{row['First Team Players']}\t{row['Players on Loan']}\t{row['Number of Managers']}\t{row['Average Manager Win Percentage']}")


    #---Print table---
    