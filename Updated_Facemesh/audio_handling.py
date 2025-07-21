import numpy as np
import av

class AudioFrameHandler:
    def __init__(self, sound_file_path):
        self.sound_file_path = sound_file_path
        self.audio_file = av.open(sound_file_path)
        self.audio_stream = self.audio_file.streams.audio[0]
        self.chunk_size = 1024  # Adjust chunk size if necessary
        self.alarm_segments = []
        self.current_segment = 0
        self.prepare_audio()

    def prepare_audio(self):
        """Prepare the alarm sound file into chunks."""
        # Read the audio file and chop it into chunks
        audio_data = self.audio_file.decode(self.audio_stream)[0].to_ndarray()
        num_chunks = int(np.ceil(len(audio_data) / self.chunk_size))

        self.alarm_segments = np.array_split(audio_data, num_chunks)

    def process(self, frame, play_sound):
        """Process each audio frame depending on the play_sound flag."""
        if play_sound:
            if self.current_segment >= len(self.alarm_segments):
                self.current_segment = 0  # Loop the alarm sound

            # Return the current segment of the alarm sound
            segment = self.alarm_segments[self.current_segment]
            self.current_segment += 1

            # Adjust the segment length to match the frame length
            if len(segment) < len(frame.to_ndarray()):
                segment = np.pad(segment, (0, len(frame.to_ndarray()) - len(segment)), 'constant')

            return av.AudioFrame.from_ndarray(segment, layout='mono')

        else:
            # Dampens the entire input sound wave to provide a silence effect
            frame_data = frame.to_ndarray()
            frame_data *= 0.0  # This effectively mutes the audio
            return av.AudioFrame.from_ndarray(frame_data, layout='mono')
