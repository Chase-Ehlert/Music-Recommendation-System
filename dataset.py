import pandas as pd
from sklearn.utils import shuffle

def createDataset():
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

    datasetFrames = [classicPopAndRock.head(10), punk.head(10), folk.head(10), pop.head(10),
                    danceAndElectronica.head(10), metal.head(10), jazz.head(10), classical.head(10),
                    hipHop.head(10), soulAndReggae.head(10)]

    datasetSample = pd.concat(datasetFrames)
    return dataset, datasetSample