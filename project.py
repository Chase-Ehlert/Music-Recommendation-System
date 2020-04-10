import pandas as pd
import operator
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from dataset import createDataset
import random

def showGenreClusterDistribution(clusterGenreCorrelationDataframe):
    clusterDistribution = {}
    for index, row in clusterGenreCorrelationDataframe.iterrows():
        clusterNumber = row['genreN']
        songGenre = row['genreT']
        if songGenre in clusterDistribution:
            clusterDistribution[songGenre].append(clusterNumber)
        else:
            clusterDistribution[songGenre] = [clusterNumber]
    return clusterDistribution

def clusterGenrePopularity(clusterGenreDistribution):
    clusterLabels = {}
    counter = 0
    while counter < 10:
        clusterLabels[counter] = {}
        clusterLabels[counter]['pop'] = 0
        clusterLabels[counter]['dance and electronica'] = 0
        clusterLabels[counter]['punk'] = 0
        clusterLabels[counter]['soul and reggae'] = 0
        clusterLabels[counter]['folk'] = 0
        clusterLabels[counter]['metal'] = 0
        clusterLabels[counter]['hip-hop'] = 0
        clusterLabels[counter]['classic pop and rock'] = 0
        clusterLabels[counter]['classical'] = 0
        clusterLabels[counter]['jazz and blues'] = 0
        counter += 1
    counter = 0
    for genre in clusterGenreDistribution:
        clusterList = clusterGenreDistribution[genre]
        for cluster in clusterList:
            clusterLabels[int(cluster)][genre] += 1
    return clusterLabels

def createClusterLabels(clusterPopularity):
    clusterLabeled = {}
    for cluster in clusterPopularity:
        clusterLabeled[cluster] = max(clusterPopularity[cluster].items(), key=operator.itemgetter(1))[0]
    return clusterLabeled

def convertClustarLabels(dataframe, labelsDictionary):
    convertedDataframe = dataframe.replace(labelsDictionary)
    return convertedDataframe

def removeTestingSongsFromDataset(dataset, sampleDataset):
    for index, row in sampleDataset.iterrows():
        dataset = dataset[dataset.track_id != row['track_id']]
    return datasetMinusSamples

def songsToRecommend(leftoverDataset, kmeans, clusteredDataset, favoriteSongIds):
    listOfSongIndexes = []
    dataframeWithFavoriteSongs = pd.DataFrame(columns=list(clusteredDataset))
    favoriteSongs = pd.DataFrame(columns=list(clusteredDataset))
    for index, row in leftoverDataset.iterrows():
        if row['track_id'] in favoriteSongIds:
            songIndex = leftoverDataset[leftoverDataset['track_id']==row['track_id']].index.item()
            listOfSongIndexes.append(songIndex)
            favoriteSongs = favoriteSongs.append(pd.Series(row, index=favoriteSongs.columns, name=songIndex))
    totalDataframe = [clusteredDataset, favoriteSongs]
    dataframeWithFavoriteSongs = pd.concat(totalDataframe)

    features = dataframeWithFavoriteSongs[timbreFeaturesAndGenre]
    timbreFeaturesOnly = timbreFeaturesAndGenre[:-1]
    kmeans.fit(features[timbreFeaturesOnly])
    features.loc[:, 'genre'] = kmeans.labels_

    genreClusterLabelsToRecommend = []
    for index, row in features.iterrows():
        if index in listOfSongIndexes:
            genreClusterLabelsToRecommend.append(row['genre'].astype(int))
    
    songsToRecommend = []
    features.drop(features.tail(5).index,inplace=True)
    for index, row in features.iterrows():
        if row['genre'] in genreClusterLabelsToRecommend:
            songsToRecommend.append(index)

    recommendedSongIndexes = []
    for i in range(5):
        recommendedSongIndexes.append(random.choice(songsToRecommend))
    
    recommendedSongs = pd.DataFrame(columns = list(leftoverDataset))
    for index, row in leftoverDataset.iterrows():
        if index in recommendedSongIndexes:
            recommendedSongs = recommendedSongs.append(pd.Series(row, index=recommendedSongs.columns, name=index))

    return recommendedSongs[['genre', 'track_id', 'artist_name', 'title']]

timbreFeaturesAndGenre = ['avg_timbre1','avg_timbre2','avg_timbre3','avg_timbre4', 'avg_timbre5',
                                'avg_timbre6','avg_timbre7','avg_timbre8', 'avg_timbre9','avg_timbre10',
                                'avg_timbre11','avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3',
                                'var_timbre4', 'var_timbre5','var_timbre6','var_timbre7','var_timbre8',
                                'var_timbre9','var_timbre10','var_timbre11','var_timbre12', 'genre']

datasetMinusSamples, testingDataset = createDataset()
datasetMinusSamples = removeTestingSongsFromDataset(datasetMinusSamples, testingDataset)
formattedDataset = datasetMinusSamples[['genre', 'track_id', 'artist_name', 'title', 'avg_timbre1']]
formattedDataset.to_csv('testingDataSet.csv', sep='\t')

featureVector = testingDataset[timbreFeaturesAndGenre]
kMeans = KMeans(n_clusters=10)
timbreFeatures = timbreFeaturesAndGenre[:-1]
kMeans.fit(featureVector[timbreFeatures])
featureVector.loc[:, 'genre'] = kMeans.labels_

genreNumbers = featureVector[['genre']]
genreTitles = testingDataset[['genre']]
genreNumbers['genre'] = genreNumbers['genre'].astype(str)

clusterRelationshipDataframe = pd.DataFrame()
clusterRelationshipDataframe['genreN'] = genreNumbers['genre']
clusterRelationshipDataframe['genreT'] = genreTitles['genre']

songClusters = showGenreClusterDistribution(clusterRelationshipDataframe)

clusterDistribution = {}
clusterDistribution = clusterGenrePopularity(songClusters)

clusterLabelsDictionary = createClusterLabels(clusterDistribution)

convertedFeatureDataframe = convertClustarLabels(featureVector[['genre']], clusterLabelsDictionary)

for key in clusterLabelsDictionary:
    print(key, clusterLabelsDictionary[key])
print('\n')
confusionMatrix = pd.DataFrame(confusion_matrix(convertedFeatureDataframe[['genre']], testingDataset[['genre']]))
print(confusionMatrix)

volunteerMusicDan = ['TRRWENG128F42867D8', 'TRASNUX128F425EAF2', 'TRGOMQC128F427E291', 'TRRUKPM128F42663B3', 'TRRDJKK128F147C7C2']
volunteerMusicJohn = ['TRGPDHJ128F14527D5', 'TRJQQWN128E078F20A', 'TRIJUZX128F149BDD4', 'TRWQRUZ128F1452AC6', 'TRUPKQZ128F9306CC6']
volunteerMusicSami = ['TRNXMNM128F427DB8C', 'TRFBQSA128F92DF83F', 'TRSBBOA128F145B794', 'TRXATRF128F4292C64', 'TRRWENG128F42867D8']

recommendation = songsToRecommend(datasetMinusSamples, kMeans, testingDataset[timbreFeaturesAndGenre], volunteerMusicSami)
recommendation.to_csv('recommendation.csv', sep='\t')