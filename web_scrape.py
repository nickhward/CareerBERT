from serpapi import GoogleSearch
import pandas as pd
from tqdm import tqdm

class JobScraper:
    def __init__(self, api_key, job_titles, google_domain, pages_to_scrape):
        self.api_key = api_key
        self.job_titles = job_titles
        self.google_domain = google_domain
        self.pages_to_scrape = pages_to_scrape

    def get_jobs(self, page_start):
        """Scrape job postings from a specific start page."""
        params = {
            "engine": "google_jobs",
            "q": self.job_titles,
            "google_domain": self.google_domain,
            "api_key": self.api_key,
            "start": page_start
        }

        client = GoogleSearch(params)
        data = client.get_dict()

        # Add a 'job_link' field to each job
        for job in data['jobs_results']:
            job['job_link'] = "https://www.google.com/search?q=" + job['job_id']

        return data['jobs_results']

    def scrape_jobs(self):
        """Scrape job postings from multiple pages."""
        all_jobs = []
        for i in tqdm(range(self.pages_to_scrape), desc='Pages'):  # Number of pages to scrape
            all_jobs.extend(self.get_jobs(i * 10))  # Each page has about 10 results
        return all_jobs

    def save_to_csv(self, filename, data):
        """Save scraped data to a CSV file."""
        df = pd.DataFrame(data)
        df.to_csv(filename)

if __name__ == '__main__':
    # Initialize scraper
    scraper = JobScraper(
        api_key='YOUR_API_KEY', 
        job_titles="Data Science|Data Engineer|Data Analyst", 
        google_domain="google.com", 
        pages_to_scrape=5
    )
    
    # Scrape jobs
    scraped_jobs = scraper.scrape_jobs()
    
    # Save to CSV
    scraper.save_to_csv("job_data/jobs.csv", scraped_jobs)