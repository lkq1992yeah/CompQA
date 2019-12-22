import os
import re
import urllib2


def gen_wiki_link_graph():
    path = '/home/xusheng/wikipedia/zh-extracted/'
    title_matcher = re.compile('<doc id=.*?>')
    link_matcher = re.compile('<a .*?>')
    in_link = dict()
    out_link = dict()
    entities = set()
    dirs = os.listdir(path)
    for dir in dirs:
        if len(dir) != 2:
            continue
        dir = path + dir
        files = os.listdir(dir)
        for file in files:
            if not file.endswith("zhs"):
                continue
            file = dir + '/' + file
            with open(file, 'r') as f:
                content = f.read()
                titles = title_matcher.finditer(content)
                for title in titles:
                    title_text = get_title(title.group(0))
                    article_begin = title.end(0)
                    article_end = content.find('</doc>', article_begin)
                    article = content[article_begin:article_end]
                    links = link_matcher.findall(article)
                    links = [get_link(link) for link in links]
                    out_link[title_text] = links
    with open('wiki_link_graph.txt', 'w') as f:
        for title, link in out_link.items():
            if len(link) == 0:
                continue
            f.write(title + '\t')
            for l in link:
                f.write('\t' + urllib2.unquote(l)) 
            f.write('\n')

def get_title(tag):
    start = tag.find('title=')
    end = tag.rfind('\"')
    return tag[start+7:end]

def get_link(tag):
    start = tag.find('href=')
    end = tag.rfind('\"')
    return tag[start+6:end]

class WikiGraph:
    def __init__(self, fname):
        self.out_link = dict()
        self.in_link = dict()
        count = 0
        with open(fname, 'r') as f:
            for line in f:
                count = count + 1
                # if count == 10000:
                #     break
                title_and_links = line.lower()[:-1].split('\t\t')
                if len(title_and_links) != 2:
                    continue
                title, links = title_and_links[0], title_and_links[1]
                links = links.split('\t')
                if not title in self.out_link:
                    self.out_link[title] = set(links)
                for link in links:
                    if not link in self.in_link:
                        self.in_link[link] = set()
                    self.in_link[link].add(title)
        self.size = len(self.out_link.keys())
    
    def get_out_link(self, entity):
        if entity in self.out_link:
            return self.out_link[entity]
        else:
            return set()

    def get_in_link(self, entity):
        if entity in self.in_link:
            return self.in_link[entity]
        else:
            return set()
                
                    

if __name__ == '__main__':
    gen_wiki_link_graph()
