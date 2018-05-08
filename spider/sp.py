# wikipedia crawl download reference weight
# Created by X.X.  05/08/2018

import scrapy

def get_info(page):
    index = page.rfind('_')
    agent = page[index+1:]
    page = page[:index]

    index = page.rfind('_')
    whart = page[index+1:]
    page = page[:index]

    index = page.rfind('_')
    link = page[index+1:]
    keywords = page[:index]
    return keywords, link, whart, agent

def get_keyword(page):
    return get_info(page)[0]
def get_link(page):
    return get_info(page)[1]
def get_access(page):
    return get_info(page)[2]
def get_agent(page):
    return get_info(page)[3]


def resemble_link(page):
    # EX: 2NE1_zh.wikipedia.org_all-access_spider
    keyword = get_keyword(page)
    link = get_link(page)
    linked = "http://"+link+"/wiki/"+keyword

    return linked

class WikiSpider(scrapy.Spider):
    name = "wikis"

    urls = []

    def __init__(self, start_index=1, size = 1,  *args, **kwargs):
        super(WikiSpider, self).__init__(*args, **kwargs)
        with open("data/train_2.csv") as f:
            for i, line in enumerate(f):
                if i < int(start_index):
                    continue
                elif i >= int(start_index) and i < int(start_index) + int(size):
                    page = line.split(",")[0][1:-1]
                    t = (i, page, resemble_link(page))
                    self.urls.append(t)
                else:
                    break
                

    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(url=url[2], callback=self.parse, meta={'xx-data-key': url[1], 'xx-data-index': url[0]})

    def parse(self, response):
        key = response.meta.get('xx-data-key')
        index = response.meta.get('xx-data-index')

        refs = response.xpath('//ol[@class="references"]/li').extract()

        yield {
            "index":index-1,
            "key":key,
            "ref-weight":len(refs)
        }