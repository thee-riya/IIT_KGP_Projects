import requests
from bs4 import BeautifulSoup
import sqlite3
import re  # Add this for regular expression functions

DB_NAME = "EPL.db"
URL = "https://en.wikipedia.org/wiki/Premier_League"

class EPLDataCollector:
    def __init__(self):
        self.url = URL
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        self.conn = sqlite3.connect(DB_NAME)
        self.c = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def clean_club_name(self, club_name):
        # Clean the club name by removing any text in square brackets
        return re.sub(r'\[.*?\]', '', club_name).strip()

    def fetch_page(self):
        response = requests.get(self.url, headers=self.headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def create_tables(self):
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS EPL_season2425 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                club_name TEXT,
                current_standing INTEGER,
                first_season INTEGER,
                season_in_top_division INTEGER,
                season_in_PL INTEGER,
                season_in_current_spell INTEGER,
                top_division_title INTEGER,
                most_recent_top_division_title INTEGER
            )
        """)

        self.c.execute("""
            CREATE TABLE IF NOT EXISTS EPL_season2425_managers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manager_name TEXT,
                nationality TEXT,
                club_name TEXT,
                appointed TEXT,
                time_as_manager TEXT
            )
        """) 

        self.conn.commit()

    def extract_club_data(self, soup):
        table = soup.find_all('table')[15]
        rows = table.find_all('tr')[1:]
        club_data = []
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 4:
                manager = row.find('th').get_text(strip=True)
                nationality = columns[0].get_text(strip=True)
                club = columns[1].get_text(strip=True)
                appointed = columns[2].get_text(strip=True)
                time_as_manager = columns[3].get_text(strip=True)
                
                # Clean the club name
                club = self.clean_club_name(club)
                
                club_data.append({
                    'Manager': manager,
                    'Nationality': nationality,
                    'Club': club,
                    'Appointed': appointed,
                    'Time as manager': time_as_manager
                })
                
        return club_data

    def extract_season_data(self, soup):
        tables = soup.find_all('table')[9]
        rows = tables.find_all('tr')[1:]
        season_data = []
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 8:
                club_name = columns[0].get_text(strip=True)
                current_standing = columns[1].get_text(strip=True)
                first_season = columns[3].get_text(strip=True)
                season_in_top_division = columns[4].get_text(strip=True)
                season_in_PL = columns[5].get_text(strip=True)
                season_in_current_spell = columns[7].get_text(strip=True)
                top_division_title = columns[8].get_text(strip=True)
                most_recent_top_division_title = columns[9].get_text(strip=True)
                
                # Clean the club name
                club_name = self.clean_club_name(club_name)
                
                season_data.append({
                    'Club Name': club_name,
                    'Current Standing': current_standing,
                    'First Season': first_season,
                    'Season in Top Division': season_in_top_division,
                    'Season in PL': season_in_PL,
                    'Season in Current Spell': season_in_current_spell,
                    'Top Division Title': top_division_title,
                    'Most Recent Top Division Title': most_recent_top_division_title
                })
        return season_data

    def store_manager_data_in_db(self, data):
        for row in data:
            self.c.execute("""
                INSERT INTO EPL_season2425_managers (manager_name, nationality, club_name, appointed, time_as_manager)
                VALUES (?, ?, ?, ?, ?)
            """, (row['Manager'], row['Nationality'], row['Club'], row['Appointed'], row['Time as manager']))
        self.conn.commit()

    def store_season_data_in_db(self, data):
        for row in data:
            self.c.execute("""
                INSERT INTO EPL_season2425 (club_name, current_standing, first_season, season_in_top_division, season_in_PL, season_in_current_spell, top_division_title, most_recent_top_division_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (row['Club Name'], row['Current Standing'], row['First Season'], row['Season in Top Division'], row['Season in PL'], row['Season in Current Spell'], row['Top Division Title'], row['Most Recent Top Division Title']))
        self.conn.commit()

    def query_earliest_appointed_manager(self):
        # Get the earliest appointed manager
        self.c.execute("SELECT manager_name, club_name, appointed FROM EPL_season2425_managers ORDER BY appointed ASC LIMIT 1")
        earliest_manager = self.c.fetchone()
        
        if earliest_manager:
            manager_name, club_name, _ = earliest_manager
            # Get the last top division title won by the club
            self.c.execute("""
                SELECT most_recent_top_division_title 
                FROM EPL_season2425 WHERE club_name = ?
            """, (club_name,))
            title = self.c.fetchone()
            return earliest_manager, title
        return None, None

    # Query B: Find nationality of the manager with the most top division titles
    def query_nationality_of_manager_with_most_titles(self):
        self.c.execute("""
            SELECT club_name, MAX(top_division_title) 
            FROM EPL_season2425 
            GROUP BY club_name 
            ORDER BY MAX(top_division_title) DESC LIMIT 1
        """)
        club_with_most_titles = self.c.fetchone()
        
        if club_with_most_titles:
            club_name, _ = club_with_most_titles
            # Find the manager of this club
            self.c.execute("""
                SELECT nationality 
                FROM EPL_season2425_managers 
                WHERE club_name = ? 
                LIMIT 1
            """, (club_name,))
            nationality = self.c.fetchone()
            return nationality
        return None

    # Query C: Find the club with most consecutive seasons and earliest appointed manager
    def query_club_with_most_consecutive_seasons(self):
        self.c.execute("""
            SELECT club_name, MAX(season_in_top_division)
            FROM EPL_season2425
            GROUP BY club_name
            ORDER BY MAX(season_in_top_division) DESC LIMIT 1
        """)
        club_with_most_consecutive_seasons = self.c.fetchone()

        if club_with_most_consecutive_seasons:
            club_name, _ = club_with_most_consecutive_seasons
            # Get the earliest-appointed manager for this club
            self.c.execute("""
                SELECT manager_name, appointed 
                FROM EPL_season2425_managers 
                WHERE club_name = ?
                ORDER BY appointed ASC LIMIT 1
            """, (club_name,))
            earliest_manager = self.c.fetchone()
            if earliest_manager:
                manager_name, appointed = earliest_manager
                return club_name, manager_name, appointed
        return None, None, None


if __name__ == "__main__":
    eplObj = EPLDataCollector()

    #---Fetch and store the HTML data---
    soup = eplObj.fetch_page()
    #---Create table schema---
    eplObj.create_tables()
    #---Fetch the data---
    club_data = eplObj.extract_club_data(soup)
    #---Store the data in db---
    eplObj.store_manager_data_in_db(club_data)
    #---Fetch the data---
    season_data = eplObj.extract_season_data(soup)
    #---Store the data in db---
    eplObj.store_season_data_in_db(season_data)

    #---Query A: Earliest-appointed manager and last top division title---
    print("Query A:")
    earliest_manager, title = eplObj.query_earliest_appointed_manager()
    if earliest_manager:
        print(f"Earliest-appointed Manager: {earliest_manager[0]}")
        print(f"Club: {earliest_manager[1]}")
        print(f"Appointed: {earliest_manager[2]}")
        if title:
            print(f"Last Top Division Title: {title[0]}")
    else:
        print("No data found for earliest-appointed manager.")

    #---Query B: Nationality of manager with the most top-division titles---
    print("\nQuery B:")
    nationality = eplObj.query_nationality_of_manager_with_most_titles()
    if nationality:
        print(f"Nationality of the manager with the most top-division titles: {nationality[0]}")

    #---Query C: Club with most consecutive seasons and earliest-appointed manager---
    print("\nQuery C:")
    most_consecutive_club, earliest_manager, appointed = eplObj.query_club_with_most_consecutive_seasons()
    if most_consecutive_club:
        print(f"Club with most consecutive seasons: {most_consecutive_club}")
        if earliest_manager:
            print(f"Earliest-appointed Manager: {earliest_manager}")
            print(f"Appointed: {appointed}")