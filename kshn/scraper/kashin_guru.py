from bs4 import BeautifulSoup

import scrapy
import nltk

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = [
        'http://kashin.guru/kshn/',
    ]
    page = 1

    def parse(self, response):
        if response.status != 200:
            return
        
    
        for article in response.css('.td-module-title'):
            href = article.css('a').attrib['href']
            title = article.css('a').attrib['title']
            request = scrapy.Request(href,
                callback=self.parse_article,
                cb_kwargs=dict(href=href,title=title)
            )
            yield request
    
        self.page += 1
        next_page = 'http://kashin.guru/kshn/page/{}/'.format(self.page)
        
        yield response.follow(next_page, self.parse)
        
    def parse_article(self, response, href, title):
        content = ' '.join(response.css('.td-post-content').css('p').extract())
        content = BeautifulSoup(content).get_text()
        
        yield dict(
            href=href,
            title=title,
            content=content
        )