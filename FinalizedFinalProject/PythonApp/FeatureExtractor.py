import numpy as np
import librosa
import os

import pandas as pd


def getSourcePath(path):
	#feature extraction method
	def extract_feature(file_name):
	    X, sample_rate = librosa.load(file_name)
	    onset_env = librosa.onset.onset_strength(X, sr=sample_rate)
	    #beats per minute(tempo)
	    tempo = np.mean(librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate).T,axis=0)
	    #captures the specific characteristics of sinhala music
	    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=1).T,axis=0)
	    #extracts the harmonic element
	    y_harmonic = np.mean(librosa.effects.harmonic(X, margin=3.0).T,axis=0)
	    return tempo,mfccs,y_harmonic

	#parsing the file system to extract features & labels
	def parse_audio_files(dirname):
		labels = []
		labels = np.array([labels])
		ext_features = []
		ext_features = np.array([ext_features])
		for dirs, subdir, files in os.walk(dirname):
			for file in files:
				tempo, mfccs, y_harmonic = extract_feature(os.path.join(dirs, file))
				features = np.hstack((mfccs, tempo, y_harmonic))
				ext_features = np.append(ext_features, np.array(features))
				labels = np.append(labels, np.array(file.split("@")[0]))
		return np.array(ext_features), np.array(labels, dtype=str)

	dirname=path
	#recording the features total data
	total_features,labels= parse_audio_files(dirname)

	total_features=total_features.reshape(total_features.shape[0]/3,3)

	x=pd.concat([pd.DataFrame(labels),pd.DataFrame(total_features)],axis=1)
	x=pd.DataFrame(x)

	x.to_csv('FeatureHolder.csv',index=False,header=None)
	return "Return Success"
