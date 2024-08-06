import requests
import webbrowser
from bs4 import BeautifulSoup

def extract_all_links(site):
    html = requests.get(site).text
    soup = BeautifulSoup(html, 'html.parser').find_all('a')
    links = [link.get('href') for link in soup]
    return links

def get_pdf_links(query,count):
    # "https://www.mdpi.com/search?sort=pubdate&page_no=1&page_count=50&year_from=1996&year_to=2024&q={query}&view=default"
    site_link = f"https://www.mdpi.com/search?q={query}"
    all_links = extract_all_links(site_link)
    pdf_links=[]
    current=0
    for link in all_links:
        print("link---------------------",link)
        if link!= None and "pdf" in link:
            pdf_links.append(link)
            current+=1
        if current>count:
            break    
    return pdf_links

def download_pdf(query,count):
    pdf_links = get_pdf_links(query,count)
    constant="https://www.mdpi.com"
    for link in pdf_links:
        print("Downloading", constant+link)
        webbrowser.open(constant+link) 

download_pdf("Artificial Intelligence",2)