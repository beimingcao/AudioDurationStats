import os
import glob
import librosa
from pyannote.audio import Pipeline

def stats_duration(args):

    vad = False
    out_dir = './'
    audio_path = args.audio_dir
    wav_path_list = glob.glob(audio_path + '/*.wav')
    wav_path_list.sort()
    access_token = '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'
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token = access_token)
    out_path = os.path.join(out_dir, os.path.basename(wav_path_list[0])[:-8] + '_duration.txt')
    f = open(out_path, 'w')
    tt = []
    for wav_path in wav_path_list:
        _t = librosa.get_duration(filename = wav_path)
        if vad == True:
            try:
                diarization = pipeline(wav_path, num_speakers=1)
                i = 0
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if i == 0:
                        _start = turn.start
                    i+=1
                _end = turn.end
                t = _end - _start
            except:
                t = _t
        else:
            t = _t
        tt.append(t)
        filename = os.path.basename(wav_path)[:-4]
        f.write(filename)
        f.write('\t')
        f.write(str(t))
        f.write('\n')
    f.write('Total')
    f.write('\t')
    f.write(str(sum(tt)))
    f.write('\n')
    f.write('mean')
    f.write('\t')
    f.write(str(sum(tt)/len(tt)))
    f.close()

if __name__ == '__main__':
    import argparse
    temp = '/home/beiming/Desktop/INTERSPEECH2021_BACKUP/DL001/WAV'
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default = temp)
    args = parser.parse_args()
    stats_duration(args)
