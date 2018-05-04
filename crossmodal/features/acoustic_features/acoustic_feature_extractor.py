
from ac_types.mfccs import mfccs
from ac_types.short_time_energy import short_time_energy as ste
from ac_types.pitch import pitch
from ac_types.hnr import hnr
from ac_types.zcr import zcr


class Acoustic_feature_extractor:

    features_expected = []

    def __init__(self, featureslist ):
        for feature in featureslist:
            if ( feature != "MFCC" and feature != "ZCR" and feature != "PITCH" and feature != "SHORT_TIME_ENERGY" and feature != "HNR"):
                raise ValueError('Not such a feature supported: '+ str(feature)  )

        self.features_expected = featureslist


    """
    this function takes a numpy array (representing the wav file) and the relevant frequency and returns a list of features.
    The order of features in the list returned is in relevant order with those in ListofFeatures
    """
    def feature_extract(self,wav,freq):
        features = []

        for feature in self.features_expected:
            if( feature == "MFCC"):
                mfcc_features = mfccs(sound = wav,sr = freq)
                features.append(mfcc_features)
                #print("MFCC Features:")
                #print(mfcc_features)
            elif( feature == "ZCR" ):
                zcr_features = zcr(sound=wav,sampling_freq =freq)
                features.append(zcr_features)
                #print("ZCR Features:")
                print(zcr_features)
            elif( feature == "PITCH" ):
                pitch_features = pitch(sound=wav,sr=freq)
            elif( feature == "SHORT_TIME_ENERGY" ):
                short_time_energy_features = ste(wav,freq)
                features.append(short_time_energy_features)
                #print("STE Features:")
                #print(short_time_energy_features)
            elif( feature == "HNR" ):
                hnr_features = hnr(wav,freq)
                features.append(hnr_features)
                #print("HNR Features:")
                #print(hnr_features)
            else:
                raise ValueError('Not such a feature supported: '+ str(feature)  )
        return features


def dummy():
        ex = Acoustic_feature_extractor(["MFCC","ZCR","HNR","PITCH","SHORT_TIME_ENERGY"])
        import pandas as pd
        from librosa.core import load
        audio_df = pd.read_pickle('../../parsers/audio_data.pkl')
        wavpath =  audio_df["WAV Path"][3]
        wav,freq = load(wavpath)
        print(wav.shape)
        ex.feature_extract(wav,freq)

dummy()
