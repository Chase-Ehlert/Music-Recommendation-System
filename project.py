import pandas as pd
import plotly.express as px
import operator
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

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

pd.options.mode.chained_assignment = None  # default='warn'

dataset = pd.read_csv('msd_genre_dataset0.csv', header=0)

classicPopAndRock = dataset.iloc[0:23895]
punk = dataset.iloc[23895:27095]
folk = dataset.iloc[27095:40287]
pop = dataset.iloc[40287:41904]
danceAndElectronica = dataset.iloc[41904:46839]
metal = dataset.iloc[46839:48942]
jazz = dataset.iloc[48942:53276]
classical = dataset.iloc[53276:55150]
hipHop = dataset.iloc[55150:55584]
soulAndReggae = dataset.iloc[55584:59600]

timbreFeaturesAndGenre = ['avg_timbre1','avg_timbre2','avg_timbre3','avg_timbre4', 'avg_timbre5',
                            'avg_timbre6','avg_timbre7','avg_timbre8', 'avg_timbre9','avg_timbre10',
                            'avg_timbre11','avg_timbre12', 'var_timbre1', 'var_timbre2', 'var_timbre3',
                            'var_timbre4', 'var_timbre5','var_timbre6','var_timbre7','var_timbre8',
                            'var_timbre9','var_timbre10','var_timbre11','var_timbre12', 'genre']

datasetFrames = [classicPopAndRock.head(10), punk.head(10), folk.head(10), pop.head(10),
                 danceAndElectronica.head(10), metal.head(10), jazz.head(10), classical.head(10),
                 hipHop.head(10), soulAndReggae.head(10)]

datasetSample = pd.concat(datasetFrames)
shuffledDatasetSample = shuffle(datasetSample)

featureVector = shuffledDatasetSample[timbreFeaturesAndGenre]

kMeans = KMeans(n_clusters=10)
timbreFeatures = timbreFeaturesAndGenre[:-1]
kMeans.fit(featureVector[timbreFeatures])
featureVector.loc[:, 'genre'] = kMeans.labels_

genreNumbers = featureVector[['genre']]
genreTitles = shuffledDatasetSample[['genre']]
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
confusionMatrix = pd.DataFrame(confusion_matrix(convertedFeatureDataframe[['genre']], shuffledDatasetSample[['genre']]))
print(confusionMatrix)

