import requests 
import os

file_url_countries = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-countries.txt"
  
r = requests.get(file_url_countries) 
if not os.path.exists('./data/countries.txt'):  
    os.makedirs('./data')
    with open("./data/countries.txt", "wb") as countries_txt: 
        countries_txt.write(r.content)
    print("File 'countries.txt' successfully downloaded")
    
else: 
    print("File 'countries.txt' already exists")

file_url_stations = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.csv"

r = requests.get(file_url_stations)
if not os.path.exists("./data/stations.csv"):
    with open("./data/stations.csv", "wb") as stations_txt:
        stations_txt.write(r.content)
    print("File 'stations.csv' successfully downloaded")
    
else:
    print("File 'stations.csv' already exists")

file_url_inventory = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"

r = requests.get(file_url_inventory)
if not os.path.exists("./data/inventory.txt"):
    with open("./data/inventory.txt", "wb") as inventory_txt:
        inventory_txt.write(r.content)
    print("File 'inventory.txt' successfully downloaded")
    
else:
    print("File 'inventory.txt' already exists")