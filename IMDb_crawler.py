# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.common.exceptions import ElementNotVisibleException
import time
from imdb import IMDb
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
#loading 需要的模組

def ParseReviews(asin):
    mid=str(asin)
    imdb=IMDb()
    #使用selenium模擬chrome搜尋IMDB評論頁面
    chrome_path = "C:\selenium_driver_chrome\chromedriver.exe"
    web = webdriver.Chrome(chrome_path)
    web.get("https://www.imdb.com/title/tt"+mid+"/reviews")
    
    #打開所有load-more button
    load='ipl-load-more__button'
    def check(load):
        try:
            web.find_element_by_class_name(load)
        except NoSuchElementException:  
            return False
        except ElementNotVisibleException:
            return False
        return True
    while(check(load)==True):
        try:
            web.find_element_by_class_name(load).click()
            time.sleep(10)
        except NoSuchElementException:  
            break
        except ElementNotVisibleException:
            break
    #打開所有show-more button
    hidediv1=web.find_elements_by_xpath('//*[@id="main"]/section/div[2]/div[2]/div[12]/div/div[1]/div[4]/div/div')
    for item in hidediv1:
        try:
            item.location_once_scrolled_into_view
            time.sleep(5)
            
            item.click()
            time.sleep(10)
        except ElementNotVisibleException:
            pass
    #打開所有 spoiler-warning__control button
    hidediv2=web.find_elements_by_xpath('//*[@id="main"]/section/div[2]/div[2]/div[10]/div[2]/div[2]/div[1]/div/div')
    for item in hidediv2:
        try:
            item.location_once_scrolled_into_view
            time.sleep(5)
            item.click()
            time.sleep(10)
        except ElementNotVisibleException:
            pass


    
    res = web.page_source
    soup = BeautifulSoup(res, "html.parser")
    movie_name =soup.select(".parent")[0].text.replace('\n', '')
    movie=imdb.get_movie(mid)   
    genres_list=[]
   
   


    for genres in movie['genres']:
        genres_dict={
            'genres':genres
        }
        genres_list.append(genres_dict)
    
    reviews_list = []
    reviews_dict = {}
    """
    append 所有需要的User評論資料到review list中
    """
    for item in soup.select(".lister-item-content"):
        try:
            review_title = item.select(".title")[0].text.replace('\n', '')
            review_rating = item.select(".rating-other-user-rating")[0].text.replace('\n', '')
            review_author =item.select(".display-name-link")[0].text.replace('\n', '')
            review_date=item.select(".review-date")[0].text.replace('\n', '')
            review_content=item.select(".text,show-more__control")[0].text.replace('\n', '')
        except IndexError:
            review_title = 'Null'
            review_rating = 'Null'
            review_author ='Null'
            review_date='Null'
            review_content='Null'
        reviews_dict={
        'name':movie_name,

        'id':mid,
        'review_author': review_author,
        'review_title': review_title,
        'review_rating':review_rating,
        'review_date': review_date,
        'review_content':review_content
         }
        reviews_list.append(reviews_dict)
        data={

        'reviews': reviews_list,

         }

    web.quit()
    #網頁關閉
    return data

def ReadAsin():
    movielist=pd.read_csv("data.csv",dtype={'imdbId': object})
    #將MovieID存於CSV檔案中 ，逐列讀取
    AsinList=movielist['imdbId']
    print(AsinList)
    for asin in AsinList:
        extracted_data=[]
        print("Downloading and processing page from IMDb")
        a=str(asin)
        extracted_data.append(ParseReviews(a))
        #寫檔
        filename=('json data location'+a+'.json')
        with open(filename, 'w') as f:
            json.dump(extracted_data,f)   

if __name__ == '__main__':
   ReadAsin()
