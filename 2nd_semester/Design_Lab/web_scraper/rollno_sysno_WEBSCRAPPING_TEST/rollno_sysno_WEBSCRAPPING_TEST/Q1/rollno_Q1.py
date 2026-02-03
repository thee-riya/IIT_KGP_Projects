import sqlite3
from bs4 import BeautifulSoup
# import other modules as required


with open(f"html_pages/UCL_2012-13.html", "r") as f: # read the data from all the files. This is the sample for one file
    text=f.read()

soup = BeautifulSoup(text, 'html.parser')

# write the remaining logic here to scrap the data








# store the data in the database









# QUERIES:
# Q1

# Q2

# Q3

# Q4

# Q5