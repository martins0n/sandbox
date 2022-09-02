from bs4 import BeautifulSoup

import scrapy
import nltk

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = [
        'https://kashin.livejournal.com/?skip=0',
    ]
    page = 0

    def parse(self, response):
        if response.status != 200:
            return
        
    
        for article in response.css('blockquote'):
            href = article.css('a').attrib.get('href')
            title = ''
            content = article.get()
            content = BeautifulSoup(content).get_text()
            yield dict(
                href=href,
                title=title,
                content=content
            )
    
        self.page += 20
        next_page = 'https://kashin.livejournal.com/?skip={}'.format(self.page)
        
        yield response.follow(next_page, self.parse)
