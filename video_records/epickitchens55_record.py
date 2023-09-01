from .video_record import VideoRecord


class EpicKitchens55_VideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return self._series['start_frame'] - 1

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def fps(self):
        return 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def tags(self):
        return self._series['all_nouns']

    @property
    def verb(self):
        return self._series['verb']

    @property
    def label(self):
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}

    @property
    def metadata(self):
        return {'uid': self._index}

