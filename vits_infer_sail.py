import soundfile
import os
import argparse
import numpy as np
import math
import sophon.sail as sail
from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin
import time
import logging
logging.basicConfig(level=logging.INFO)


class VITS:
    def __init__(
        self,
        args,
    ):
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.max_length = self.input_shape[1]

        self.inference_time = 0.0
        self.sample_rate = 16000
        self.stage_factor = 900.0


    def init(self):
        self.inference_time = 0.0


    def estimate_silence_threshold(self, audio, sample_rate, duration=0.1):
        """
        Estimate the threshold of silence in an audio signal by calculating
        the average energy of the first 'duration' seconds of the audio.

        Args:
            audio: numpy array of audio data.
            sample_rate: the sample rate of the audio data.
            duration: duration (in seconds) of the initial audio to consider for silence.

        Returns:
            The estimated silence threshold.
        """
        # Calculate the number of samples to consider
        num_samples = int(sample_rate * duration)

        # Calculate the energy of the initial segment of the audio
        initial_energy = np.mean(np.abs(audio[-num_samples:]))

        # Return the average energy as the threshold
        return initial_energy


    def remove_silence_from_end(self, audio, sample_rate, threshold=0.01, frame_length=512):
        """
        Removes silence from the end of an audio signal using a specified energy threshold.
        If no threshold is provided, it estimates one based on the initial part of the audio.

        Args:
            audio: numpy array of audio data.
            sample_rate: the sample rate of the audio data.
            threshold: amplitude threshold to consider as silence. If None, will be estimated.
            frame_length: number of samples to consider in each frame.

        Returns:
            The audio signal with end silence removed.
        """
        if threshold is None:
            threshold = self.estimate_silence_threshold(audio, sample_rate)

        # Calculate the energy of audio by frame
        energies = [np.mean(np.abs(audio[i:i+frame_length])) for i in range(0, len(audio), frame_length)]

        # Find the last frame with energy above the threshold
        for i, energy in enumerate(reversed(energies)):
            if energy > threshold:
                last_non_silent_frame = len(energies) - i - 1
                break
        else:
            # In case the whole audio is below the threshold
            return np.array([])

        # Calculate the end index of the last non-silent frame
        end_index = (last_non_silent_frame + 1) * frame_length

        # Return the trimmed audio
        return audio[:end_index]


    def split_text_near_punctuation(self, text, max_length):
        # Define punctuation marks where the text can be split
        punctuation = "。！？，、；：,."
        # Initialize a list to hold the split text segments
        split_texts = []

        # Continue splitting the text until the remaining text is shorter than max_length
        while len(text) > max_length:
            # Assume we need to split at the max_length, then search left for the nearest punctuation
            split_pos = max_length
            # Search left for the nearest punctuation
            while split_pos > 0 and text[split_pos] not in punctuation:
                split_pos -= 1

            # If no punctuation is found to the left, split at the original max_length
            if split_pos == 0:
                split_pos = max_length

            # Split the text and add to the list
            split_texts.append(text[:split_pos + 1])
            # Update the remaining text
            text = text[split_pos + 1:].lstrip()

        # Add the remaining text segment
        split_texts.append(text)
        return split_texts

    def preprocess(self, x: np.ndarray):
        x = np.expand_dims(x, axis=0) if x.ndim == 1 else x
        # Initialize an empty list to collect output tensors
        x_segments = []
        for i in range(0, x.shape[1], self.max_length):
            # Extract a sequence of length `self.max_length` from x
            x_segment = x[:, i:i + self.max_length]

            # If the segment is shorter than `self.max_length`, pad it
            if x_segment.shape[1] < self.max_length:
                padding_size = self.max_length - x_segment.shape[1]
                x_segment = np.pad(x_segment, [(0, 0), (0, padding_size)], mode='constant', constant_values=0)

            x_segments.append(x_segment)
        return x_segments


    def inference(self, x_segments: list[np.ndarray]):
        # Initialize an empty list to collect output tensors
        outputs = []
        for i in range(len(x_segments)):
            # Extract a sequence of length `self.max_length` from x
            start_time = time.time()
            input_data = {self.input_name: x_segments[i]}
            output_data = self.net.process(self.graph_name, input_data)
            self.inference_time += time.time() - start_time
            y_max, y_segment = output_data.values()

            if i == len(x_segments) - 1:
                # This is the last segment, cast slice
                y_segment = y_segment[:math.ceil(y_max[0] / self.stage_factor * len(y_segment))]
                y_segment = self.remove_silence_from_end(y_segment, self.sample_rate)

            # Collect the output
            outputs.append(y_segment)

        # Concatenate all output segments along the sequence dimension
        y = np.concatenate(outputs, axis=-1)
        return y

    def __call__(self, x: np.ndarray):
        """
        Args:
          x:
            A int32 array of shape (1, 128) or (1, 512)
        """
        # self.set_stage(x)
        x_segments = self.preprocess(x)
        y = self.inference(x_segments)
        return y


def main():
    parser = argparse.ArgumentParser(
        description='Inference code for bert vits models')
    parser.add_argument('--bmodel', type=str, default='vits-chinese_f16.bmodel', help='path of bmodel')
    parser.add_argument('--text_file', type=str, default='vits_infer_item.txt', help='path of text')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    vits = VITS(args)

    tts_front = VITS_PinYin(None, None, hasBert=False)
    results_path = "./results/"
    os.makedirs(results_path, exist_ok=True)

    n = 0
    total_time = time.time()
    fo = open(args.text_file, "r+", encoding='utf-8')
    vits.init()
    while True:
        try:
            item = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if not item:
            break

        # cut log items
        split_items = vits.split_text_near_punctuation(item, int(vits.max_length / 2))
        output_audio =[]
        for split_item in split_items:
            logging.info(split_item)
            phonemes, _ = tts_front.chinese_to_phonemes(split_item)
            input_ids = cleaned_text_to_sequence(phonemes)
            x = np.array(input_ids, dtype=np.int32)
            output_audio.append(vits(x))
        n += 1

        audio_path = f"{results_path}sail_{n}.wav"
        soundfile.write(
            audio_path, np.concatenate(output_audio, axis=-1), vits.sample_rate)
        logging.info("save audio to {}".format(audio_path))

    fo.close()
    total_time = time.time() - total_time

    # calculate speed
    logging.info("------------------ Predict Time Info ----------------------")
    inference_time = vits.inference_time
    logging.info("text nums: {}, inference_time(ms): {:.2f}".format(n, inference_time * 1000))
    logging.info("text nums: {}, total_time(ms): {:.2f}".format(n, total_time * 1000))


if __name__ == "__main__":
    main()
