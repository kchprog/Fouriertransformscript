"""
    Python script for graphing sounds as Fourier transforms, made by Kevin Chen (utorid: chenke94)
    for PHY207. Programmed with the assistance of GPT=4 and Claude-3 LLMs.
"""

import os
import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
from scipy.signal import find_peaks



GLOBAL_currentCanvas = None

def freq_to_note(frequency):
    """
    Converts a frequency to a musical note name. Reference:
    https://en.wikipedia.org/wiki/Piano_key_frequencies
    https://stackoverflow.com/questions/64505024/turning-frequencies-into-notes-in-python

    """
    A4 = 440

    noteNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    ## Note on the keyboard; the MIDI offset for A4 is 49 by convention
    note_number = 12 * math.log(frequency / A4, 2) + 49
    note_number = round(note_number)

    ## Get the name of the note
    note = (note_number - 1 ) % 12
    note = noteNames[note]
    
    ## Get the octave
    octave = (note_number + 8 ) // 12
    
    return note + str(octave)


def plot_fourier_transform(audioArray, sampleRate):
    """
    Returns a matplotlib figure plotting out the fourier transform of a given audio signal

    INPUT:
        audioArray: a numpy list representing the sound (2D or 1D, converted to 1D afterward)
        sampleRate: the sample rate of the audio clip, expected to be 44100 (standard)
    
    OUTPUTL
        a matplotlib figure with the fourier spectrum of the signal represented
    """
    ## Flatten the audio array
    if audioArray.ndim == 2 and audioArray.shape[1] == 2:
        # This averages the two channels to create a mono signal
        audioArray = audioArray.mean(axis=1)
    elif audioArray.ndim == 2 and audioArray.shape[1] == 1:
        # This converts a 2-D array with one column into a 1-D array
        audioArray = audioArray.flatten()
    elif audioArray.ndim != 1:
        raise ValueError('`audioArray` must be a 1-D array or a 2-D array with one or two columns.')

    ## Perform the Fourier Transform using numpy built in functions lol
    fft_result = np.fft.fft(audioArray)
    N = len(fft_result)
    frequencies = np.fft.fftfreq(N, 1/sampleRate)

    ## As this is a 'real signal', we only take the positive parts of the spectrum
    fft_result = fft_result[:N//2]
    frequencies = frequencies[:N//2]

    ## MIN distance between peaks (to stop us from picking clustered peaks)

    min_distance = 1000
    min_prominence = 0.3
    ## Find peaks
    peaks, _ = find_peaks(np.abs(fft_result), prominence=min_prominence, distance=min_distance)


    ## Create a figure
    fig = Figure(figsize=(6, 4))
    subfig = fig.add_subplot(1, 1, 1)

    ## Plot the spectrum
    subfig.plot(frequencies, np.abs(fft_result))
    subfig.set_xlabel('Frequency (Hertz)')
    subfig.set_ylabel('Relative Amplitude')
    subfig.set_title('Fourier Frequency Spectrum')

    ## restrict the x-axis to contain only frequencies we're expecting
    subfig.set_xlim(0, 10000)

    ## Annotate the top 3 peaks (rough way to find harmonics / major constituents)
    ## Get the top peaks by amplitude
    top_peaks = peaks[np.argsort(np.abs(fft_result[peaks]))[-6:]]

    texts = []
    count = 1
    print("Notable peaks, in order of decreasing amplitude:")
    highestPeakAmp = np.abs(fft_result[top_peaks[-1]])

    for peak in reversed(top_peaks):
        peak_freq = frequencies[peak]
        peak_amp = np.abs(fft_result[peak])

        size = 8 if peak_amp > (highestPeakAmp / 3) else 6
        print("Peak {}: Frequency {}, equivalent to note {}".format(count, peak_freq, freq_to_note(peak_freq)))
        texts.append(subfig.annotate(f'{peak_freq:.2f} Hz, {freq_to_note(peak_freq)}', 
                                    xy=(peak_freq, peak_amp), 
                                    xytext=(3, 4), 
                                    textcoords='offset points', 
                                    fontsize=size,
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.1')))

        count += 1
    # Automatically adjust the positions of the annotations
    return fig


def select_file():
    """Selects the file, then presents the fourier transform plot in the tkinter window"""

    global GLOBAL_currentCanvas

    file_path = filedialog.askopenfilename(filetypes=[('Video Files', '*.mp4'), ('Audio Files', '*.mp3')])

    if file_path:

        ## Clear the canvas whenever we call select_file()
        if GLOBAL_currentCanvas is not None:
            GLOBAL_currentCanvas.get_tk_widget().destroy()

        file_extension = os.path.splitext(file_path)[1]

        if file_extension.lower() == '.mp4':
            # Extract audio from the video file
            video = mp.VideoFileClip(file_path)
            audio = video.audio
            temp_audio_path = os.path.join(os.getcwd(), 'fourier_temp.wav')
            audio.write_audiofile(temp_audio_path, codec='pcm_s16le')

            ## Loads in the audio file into the temporary new file, then get the sound over the whole video
            ## as individual instantaneous samples

            audio_data = mp.AudioFileClip('fourier_temp.wav')
            samples =  list(audio_data.iter_frames())
            audio_array = np.array(samples)

            ## gets the sample rate of the audio
            sample_rate = audio_data.fps

            ## do fourier transform and get the figure
            fig = plot_fourier_transform(audio_array, sample_rate)

            GLOBAL_currentCanvas = FigureCanvasTkAgg(fig, master=root)
            GLOBAL_currentCanvas.draw()
            GLOBAL_currentCanvas.get_tk_widget().pack()

            audio_data.close()
            audio.close()
            video.close()
            os.remove('fourier_temp.wav')

        elif file_extension.lower() == '.mp3':
            audio_data = mp.AudioFileClip(file_path)
            samples =  list(audio_data.iter_frames())
            audio_array = np.array(samples)

            sample_rate = audio_data.fps

            print("SAMPLE RATE", sample_rate)

            fig = plot_fourier_transform(audio_array, sample_rate)

            GLOBAL_currentCanvas = FigureCanvasTkAgg(fig, master=root)
            GLOBAL_currentCanvas.draw()
            GLOBAL_currentCanvas.get_tk_widget().pack()

            audio_data.close()

        else:
            tk.messagebox.showerror("Error", "Unsupported file format!")

## Initialize and create the main window
root = tk.Tk()
root.title('Audio Fourier Transform')

## Button to select the video.audio file
button = tk.Button(root, text='Select File (.mp4, .mp3 supported!)', command=select_file)
button.pack(pady=20)

# Start the GUI event loop
root.mainloop()