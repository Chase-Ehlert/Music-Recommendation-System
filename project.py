import pandas as pd
import operator
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from dataset import createDataset

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

timbreFeaturesAndGenre = ['avg_timbre1','avg_timbre2','avg_timbre3','avg_timbre4', 'avg_timbre5',
                                'avg_timbre6','avg_timbre7','avg_timbre8', 'avg_timbre9','avg_timbre10',
                                'avg_timbre11','avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3',
                                'var_timbre4', 'var_timbre5','var_timbre6','var_timbre7','var_timbre8',
                                'var_timbre9','var_timbre10','var_timbre11','var_timbre12', 'genre']

datasetMinusSamples, testingDataset = createDataset()
dataSetMinusSamples = removeTestingSongsFromDataset(datasetMinusSamples, testingDataset)
datasetMinusSamples.to_csv('testingDataSet.csv', sep='\t')

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

