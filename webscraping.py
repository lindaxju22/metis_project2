#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:00:22 2019

@author: lindaxju
"""

import re
import time
import pickle
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from selenium import webdriver
import chromedriver_binary
from bs4 import BeautifulSoup

#%%
#######################################
# Create dictionary of guards and IDs #
#######################################
def get_top_players(seconds, url):
    driver_leaders= webdriver.Chrome()
    driver_leaders.get(url)
    time.sleep(seconds)
    
    inf_adjust_season_selector = '//select[@name="Season"]/option[@label="2018-19"]'
    driver_leaders.find_element_by_xpath(inf_adjust_season_selector).click()
    time.sleep(seconds)
    
    inf_adjust_seasontype_selector = '//select[@name="SeasonType"]/option[@label="Regular Season"]'
    driver_leaders.find_element_by_xpath(inf_adjust_seasontype_selector).click()
    time.sleep(seconds)
    
    inf_adjust_advfilter_selector = '//a[@class="stats-filters-advanced__toggle"]'
    driver_leaders.find_element_by_xpath(inf_adjust_advfilter_selector).click()
    time.sleep(seconds)
    
    inf_adjust_pos_selector = '//select[@name="PlayerPosition"]/option[@label="Guard"]'
    driver_leaders.find_element_by_xpath(inf_adjust_pos_selector).click()
    time.sleep(seconds)
    
    inf_adjust_pos_selector = '//a[@class="run-it"]'
    driver_leaders.find_element_by_xpath(inf_adjust_pos_selector).click()
    time.sleep(seconds+2)
    
    # show all games
    driver_leaders.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select/option[1]').click()
    time.sleep(seconds)
    
    soup_leaders = BeautifulSoup(driver_leaders.page_source, 'html.parser')
    
    leaders_list = list(soup_leaders.find_all(class_='player'))
    leaders_list = [str(i) for i in leaders_list]
    subs = 'href'
    scraped_leaders = [i for i in leaders_list if subs in i]
    
    driver_leaders.close()
    
    leaders = defaultdict(int)
    
    for leader in scraped_leaders:
        player_name = re.search('/">(.*)</a>', leader).group(1)
        player_id = re.search(r'\d+', leader).group()
        leaders[player_name] = int(player_id)

    return leaders
#%%
####################
# Scrape game logs #
####################
def get_game_log(seconds, url):
    driver_player = webdriver.Chrome()
    driver_player.get(url)
    time.sleep(seconds)
    
    inf_adjust_season_selector = '//select[@name="Season"]/option[@label="2018-19"]'
    driver_player.find_element_by_xpath(inf_adjust_season_selector).click()
    time.sleep(seconds)
    
    inf_adjust_seasontype_selector = '//select[@name="SeasonType"]/option[@label="Regular Season"]'
    driver_player.find_element_by_xpath(inf_adjust_seasontype_selector).click()
    time.sleep(seconds+2)
    
    # show all games
    try:
        driver_player.find_element_by_xpath('/html/body/main/div[2]/div/div/div[3]/div/div/div/nba-stat-table/div[1]/div/div/select/option[1]').click()
        time.sleep(seconds+2)
    except:
        pass # doing nothing on exception

    table_player = driver_player.find_element_by_class_name('nba-stat-table__overflow')
    
    player_stats = []
    
    for line_id, lines in enumerate(table_player.text.split('\n')):
        if line_id != 0:
            player_stats.append( [i for i in lines.split(' ')] )
            
    driver_player.close()
            
    df = pd.DataFrame({'month': [i[0] for i in player_stats],
                       'day': [i[1] for i in player_stats],
                       'year': [int(i[2]) for i in player_stats],
                       'team': [i[4] for i in player_stats],
                       'HvA': [i[5] for i in player_stats],
                       'opp': [i[6] for i in player_stats],
                       'wl': [i[7] for i in player_stats],
                       'min': [int(i[8]) for i in player_stats],
                       'pts': [int(i[9]) for i in player_stats],
                       'fgm': [int(i[10]) for i in player_stats], 
                       'fga': [int(i[11]) for i in player_stats],
                       'fgpct': [float(i[12]) for i in player_stats],
                       '3pm': [int(i[13]) for i in player_stats],
                       '3pa': [int(i[14]) for i in player_stats],
                       '3ppct': [float(i[15]) for i in player_stats],
                       'ftm': [int(i[16]) for i in player_stats],
                       'fta': [int(i[17]) for i in player_stats],
                       'ftpct': [float(i[18]) for i in player_stats],
                       'oreb': [int(i[19]) for i in player_stats],
                       'dreb': [int(i[20]) for i in player_stats],
                       'reb': [int(i[21]) for i in player_stats],
                       'ast': [int(i[22]) for i in player_stats],
                       'stl': [int(i[23]) for i in player_stats],
                       'blk': [int(i[24]) for i in player_stats],
                       'tov': [int(i[25]) for i in player_stats],
                       'pf': [int(i[26]) for i in player_stats],
                       'pm': [int(i[27]) for i in player_stats]
                        })
    
    return df
#%%
###################
# Clean game logs #
###################
def clean_game_log(df, player, roll_window):
    idx = 0
    new_col = player
    
    df_player = df.copy()
    df_player.insert(loc=idx, column='player', value=new_col)
    
    idx = 4
    new_col = pd.to_datetime(df_player['year'].astype(str) + '-' +
                             df_player['month'].astype(str) + '-' +
                             df_player['day'].str[:2].astype(str))
    df_player.insert(loc=idx, column='date', value=new_col)
    
    df_player.sort_values(by=['date'],inplace=True)
    
    df_player = df_player.reset_index(drop=True)
    
    idx = 6
    new_col = (df['HvA'] == '@')
    df_player.insert(loc=idx, column='home', value=new_col)
    
    df_player['fan_score'] = df_player['pts'] + 1.2 * df_player['reb'] + 1.5 * df_player['ast'] + 3 * df_player['stl'] + 3 * df_player['blk'] - df_player['tov'] 
    df_player['days_rest'] = df_player['date'].diff().dt.days
    
    cols_SMA_list = ['min','pts','fgm','fga','fgpct','3pm',
                 '3pa','3ppct','ftm','fta','ftpct','oreb',
                 'dreb','reb','ast','stl','blk','tov',
                 'pf','pm','fan_score']
    
    for col in cols_SMA_list:
        new_col = col + '_SMA'
        df_player[str(new_col)] = df_player[col].rolling(window=roll_window).mean().shift(1)
        
    return(df_player)
#%%
#########################
# Scrape advanced stats #
#########################
def get_advanced_stats(seconds, url):
    driver_player_adv = webdriver.Chrome()
    driver_player_adv.get(url)
    time.sleep(seconds)
    
    inf_adjust_season_selector = '//select[@name="Season"]/option[@label="2018-19"]'
    driver_player_adv.find_element_by_xpath(inf_adjust_season_selector).click()
    time.sleep(seconds)
    
    inf_adjust_seasontype_selector = '//select[@name="SeasonType"]/option[@label="Regular Season"]'
    driver_player_adv.find_element_by_xpath(inf_adjust_seasontype_selector).click()
    time.sleep(seconds+2)
    
    # show all games
    try:
        driver_player_adv.find_element_by_xpath('/html/body/main/div[2]/div/div/div[3]/div/div/div/nba-stat-table/div[1]/div/div/select/option[1]').click()
        time.sleep(seconds+2)
    except:
        pass # doing nothing on exception
        
    table_player_adv = driver_player_adv.find_element_by_class_name('nba-stat-table__overflow')
    
    player_stats_adv = []

    for line_id, lines in enumerate(table_player_adv.text.split('\n')):
        if line_id != 0:
            player_stats_adv.append( [i for i in lines.split(' ')] )
            
    driver_player_adv.close()
    
    df_adv = pd.DataFrame({'month': [i[0] for i in player_stats_adv],
                       'day': [i[1] for i in player_stats_adv],
                       'year': [int(i[2]) for i in player_stats_adv],
                       'team': [i[4] for i in player_stats_adv],
                       'HvA': [i[5] for i in player_stats_adv],
                       'opp': [i[6] for i in player_stats_adv],
                       'wl': [i[7] for i in player_stats_adv],
                       'min': [int(i[8]) for i in player_stats_adv],
                       'offrtg': [float(i[9]) for i in player_stats_adv],
                       'defrtg': [float(i[10]) for i in player_stats_adv],
                       'netrtg': [float(i[11]) for i in player_stats_adv],
                       'astpct': [float(i[12]) for i in player_stats_adv],
                       'ast_to': [float(i[13]) for i in player_stats_adv],
                       'ast_ratio': [float(i[14]) for i in player_stats_adv],
                       'orebpct': [float(i[15]) for i in player_stats_adv],
                       'drebpct': [float(i[16]) for i in player_stats_adv],
                       'rebpct': [float(i[17]) for i in player_stats_adv],
                       'to_ratio': [float(i[18]) for i in player_stats_adv],
                       'efgpct': [float(i[19]) for i in player_stats_adv],
                       'tspct': [float(i[20]) for i in player_stats_adv],
                       'usgpct': [float(i[21]) for i in player_stats_adv],
                       'pace': [float(i[22]) for i in player_stats_adv],
                       'pie': [float(i[23]) for i in player_stats_adv],
                      })
    return df_adv
    
#%%
########################
# Clean advanced stats #
########################
def clean_adv_stats(df, roll_window):
    df_player_adv = df.copy()
    
    idx = 3
    new_col = pd.to_datetime(df_player_adv['year'].astype(str) + '-' +
                             df_player_adv['month'].astype(str) + '-' +
                             df_player_adv['day'].str[:2].astype(str))
    df_player_adv.insert(loc=idx, column='date', value=new_col)
    
    df_player_adv.sort_values(by=['date'],inplace=True)
    
    df_player_adv = df_player_adv.reset_index(drop=True)
    
    cols_adv_SMA_list = ['offrtg','defrtg','netrtg','usgpct','pace','pie']
    
    for col in cols_adv_SMA_list:
        new_col = col + '_SMA'
        df_player_adv[str(new_col)] = df_player_adv[col].rolling(window=roll_window).mean().shift(1)
        
    return df_player_adv
#%%
#######################################
# Combine game log and advanced stats #
#######################################
def combine_game_log_adv_stats(df_game_log, df_adv_stats):
    df_player_cleaned = df_game_log.copy()
    df_player_cleaned.drop(columns=['month','day','year','HvA','wl','min','pts',
                                    'fgm','fga','fgpct','3pm','3pa','3ppct','ftm',
                                    'fta','ftpct','oreb','dreb','reb','ast','stl',
                                    'blk','tov','pf','pm'],inplace=True)

    df_player_adv_cleaned = df_adv_stats.copy()
    df_player_adv_cleaned.drop(columns=['month','day','year','team','HvA',
                                        'opp','wl','min','offrtg','defrtg',
                                        'netrtg','astpct','ast_to','ast_ratio',
                                        'orebpct','drebpct','rebpct','to_ratio',
                                        'efgpct','tspct','usgpct','pace','pie'],inplace=True)
    
    df_player_final = pd.merge(df_player_cleaned, df_player_adv_cleaned, on='date')

    return df_player_final
#%%
def get_opp_team_stats(seconds, url):
    driver_team_adv = webdriver.Chrome()
    driver_team_adv.get(url)
    time.sleep(seconds)
    
    inf_adjust_season_selector = '//select[@name="Season"]/option[@label="2018-19"]'
    driver_team_adv.find_element_by_xpath(inf_adjust_season_selector).click()
    time.sleep(seconds)
    
    inf_adjust_seasontype_selector = '//select[@name="SeasonType"]/option[@label="Regular Season"]'
    driver_team_adv.find_element_by_xpath(inf_adjust_seasontype_selector).click()
    time.sleep(seconds+6)
    
    # show all games
    driver_team_adv.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select/option[1]').click()
    time.sleep(seconds)
    
    table_team_adv = driver_team_adv.find_element_by_class_name('nba-stat-table__overflow')
    
    team_stats_adv = []

    for line_id, lines in enumerate(table_team_adv.text.split('\n')):
        if (line_id != 0 and line_id != 1):
            team_stats_adv.append( [i for i in lines.split(' ')] )
    
    driver_team_adv.close()
    
    df_team_adv = pd.DataFrame({'team': [i[0] for i in team_stats_adv],
                            'HvA': [i[2] for i in team_stats_adv],
                            'opp': [i[3] for i in team_stats_adv],
                            'date': [datetime.strptime(i[4],'%m/%d/%Y') for i in team_stats_adv],
                            'wl': [i[5] for i in team_stats_adv],
                            'min': [int(i[6]) for i in team_stats_adv],
                            'offrtg': [float(i[7]) for i in team_stats_adv],
                            'defrtg': [float(i[8]) for i in team_stats_adv],
                            'netrtg': [float(i[9]) for i in team_stats_adv],
                            'astpct': [float(i[10]) for i in team_stats_adv],
                            'ast_to': [float(i[11]) for i in team_stats_adv],
                            'ast_ratio': [float(i[12]) for i in team_stats_adv],
                            'orebpct': [float(i[13]) for i in team_stats_adv],
                            'drebpct': [float(i[14]) for i in team_stats_adv],
                            'rebpct': [float(i[15]) for i in team_stats_adv],
                            'tovpct': [float(i[16]) for i in team_stats_adv],
                            'efgpct': [float(i[17]) for i in team_stats_adv],
                            'tspct': [float(i[18]) for i in team_stats_adv],
                            'pace': [float(i[19]) for i in team_stats_adv],
                            'pie': [float(i[20]) for i in team_stats_adv],
                      })
    
    return df_team_adv
#%%
def clean_opp_team_stats(df_team_adv, roll_window):
    team_names = list(df_team_adv['team'].unique())
    cols_team_adv_SMA_list = ['offrtg','defrtg','netrtg','pace','pie']
    df_team_final = pd.DataFrame()

    for team in team_names:
        df_team = df_team_adv[df_team_adv['team'] == team].copy()
        df_team.sort_values(by=['date'],inplace=True)
        df_team = df_team.reset_index(drop=True)
        for col in cols_team_adv_SMA_list:
            new_col = col + '_SMA_opp'
            df_team[str(new_col)] = df_team[col].rolling(window=roll_window).mean().shift(1)

        df_team.drop(['HvA','opp','wl','min','offrtg','defrtg','netrtg','astpct','ast_to',
                      'ast_ratio','orebpct','drebpct','rebpct','tovpct','efgpct','tspct',
                      'pace','pie'], axis=1, inplace=True)
        df_team.rename(columns={'team':'opp'}, inplace=True)
        df_team_final=df_team_final.append(df_team)
    
    df_team_final.reset_index(drop=True,inplace=True)

    return df_team_final
#%%
def combine_player_opp_team(df_player,df_team):
    df_player_final = df_player.copy()
    df_player_final = df_player_final.merge(df_team, how='left',on=['date','opp'])
    df_player_final = df_player_final.dropna()
    df_player_final.reset_index(drop=True,inplace=True)
    
    return df_player_final
#%%
seconds = 2
n = 100
roll_window = 3
#%%
url_leaders = 'https://stats.nba.com/players/traditional/'
leaders = get_top_players(seconds, url_leaders)
#with open('data/leaders_dict.pickle', 'wb') as handle:
#    pickle.dump(leaders, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('data/leaders_dict.pickle', 'rb') as handle:
#    leaders = pickle.load(handle)
#%%
top_players = []
for key in leaders:
    top_players.append(key)
#%%
top_n_players = top_players[:n]
top_n_players
#%%
url_oppteam = 'https://stats.nba.com/teams/boxscores-advanced/'
df_team_adv = get_opp_team_stats(seconds, url_oppteam)
df_team_final = clean_opp_team_stats(df_team_adv,roll_window)
#with open('data/df_team_final.pickle', 'wb') as to_write:
#    pickle.dump(data/df_team_final, to_write)
#with open('data/df_team_final.pickle','rb') as read_file:
#    df_team_final = pickle.load(read_file)
#%%
df_all = pd.DataFrame()
count = 1

for player in top_n_players:
    print(count, player)
    url_playerprofile = 'https://stats.nba.com/player/'+str(leaders[player])+'/boxscores-traditional/'
    df_game_log = get_game_log(seconds, url_playerprofile)
    df_game_log_cleaned = clean_game_log(df_game_log, player, roll_window)
    url_playerprofile_adv = 'https://stats.nba.com/player/'+str(leaders[player])+'/boxscores-advanced/'
    df_adv_stats = get_advanced_stats(seconds, url_playerprofile_adv)
    df_adv_stats_cleaned = clean_adv_stats(df_adv_stats, roll_window)
    df_player_final = combine_game_log_adv_stats(df_game_log_cleaned,df_adv_stats_cleaned)

    df_player_final = combine_player_opp_team(df_player_final,df_team_final)
    df_all = df_all.append(df_player_final,sort=False)
    df_all.reset_index(drop=True,inplace=True)
    count += 1
df_all
#%%
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'df_all_'+timestamp
df_all.to_csv(r'data/'+filename+'.csv')
with open('data/'+filename+'.pickle', 'wb') as to_write:
    pickle.dump(df_all, to_write)
#%%
#df_all['days_rest'] = df_all.groupby(['player'])['date'].diff().dt.days
#df_all.to_csv(r'data/df_all.csv')
#with open('data/df_all.pickle', 'wb') as to_write:
#    pickle.dump(df_all, to_write)
#with open('data/df_all.pickle','rb') as read_file:
#    df_all = pickle.load(df_all)

#%%
#%%